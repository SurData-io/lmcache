# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future, TimeoutError
from typing import Any, Callable, List, Optional, Sequence, Set
import asyncio
import inspect
import threading
import time

# First Party
from lmcache import torch_device_type
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.exceptions import IrrecoverableException
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.connector import CreateConnector
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.naive_serde import CreateSerde

logger = init_logger(__name__)


class RemoteBackend(StorageBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: Optional[LocalCPUBackend],
        dst_device: str = torch_device_type,
        plugin_name: Optional[str] = None,
    ):
        super().__init__(dst_device=dst_device)
        self.put_tasks: Set[CacheEngineKey] = set()
        self.lock = threading.Lock()

        self.plugin_name = plugin_name

        # Determine if we're using legacy remote_url or new plugin-based approach
        if plugin_name is not None:
            # Using plugin-based approach
            self.remote_url = f"plugin://{plugin_name}"
            logger.info("Creating RemoteBackend for plugin: %s", plugin_name)
        else:
            # Legacy remote_url approach
            if config.remote_url is None:
                raise ValueError(
                    "remote_url must be provided when not using plugin_name"
                )
            self.remote_url = config.remote_url

        self.local_cpu_backend = local_cpu_backend

        self.loop = loop
        self.config = config
        self.metadata = metadata

        # Re-establish connection only when the connection
        # has been lost for 10 secs
        self.connection: Optional[RemoteConnector] = None
        self.min_reconnect_interval = 10
        self.failure_time = -1000000.0
        # First CreateConnector success; later reconnects may leave connection=None.
        self._connect_ever_ok = False
        # Must exist before init_connection(): CreateConnector then calls
        # _try_attach_preload_provider. Missing this attr used to be caught as
        # a connection failure, the live SurData connector was discarded, and
        # HealthMonitor reconnect hit surdata_sdk_init bad state (rc=-6).
        self._preload_coordinator = None
        self.init_connection()

        assert config.remote_serde is not None
        self.serializer, self.deserializer = CreateSerde(
            config.remote_serde, metadata, config
        )

        # Precompute MLA mode status
        self._mla_worker_id_as0_mode = (
            config.get_extra_config_value(
                "remote_enable_mla_worker_id_as0", metadata.use_mla
            )
            and metadata.use_mla
            and metadata.world_size > 1
            and metadata.worker_id != 0
        )
        logger.info("metadata=%s", metadata)
        logger.info(
            "Connected to remote storage at %s, remote_mla_worker_id_as_0 mode: %s",
            self.remote_url,
            self._mla_worker_id_as0_mode,
        )

        # TODO(Jiayi): If we want to have cache admission policies,
        # we must make decision (whether to send or not) at the local side

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

        # NOTE: Health monitoring is now handled at the LMCacheEngine level
        # through HealthMonitor. RemoteBackend no longer manages its own
        # health monitoring. The HealthMonitor in LMCacheEngine will
        # register RemoteBackendHealthCheck for each RemoteBackend.

        self._get_blocking_failed_count = 0
        self._put_failed_count = 0

        self._setup_metrics()

    def set_preload_coordinator(self, coordinator) -> None:
        self._preload_coordinator = coordinator
        self._try_attach_preload_provider()

    def _try_attach_preload_provider(self) -> None:
        coordinator = getattr(self, "_preload_coordinator", None)
        if coordinator is None or self.connection is None:
            return
        from lmcache.v1.preload_lifecycle import find_provider

        provider = find_provider({"Remote": self})
        if provider is None:
            return
        coordinator.attach_provider(provider)

    def _setup_metrics(self) -> None:
        prometheus_logger = PrometheusLogger.GetOrCreate(
            self.metadata,
            config=self.config,
        )
        prometheus_logger.remote_put_task_num.set_function(lambda: len(self.put_tasks))
        prometheus_logger.get_blocking_failed_count.set_function(
            lambda: self._get_blocking_failed_count
        )
        prometheus_logger.put_failed_count.set_function(lambda: self._put_failed_count)

    def __str__(self):
        return self.__class__.__name__

    def init_connection(self):
        # Initialize connection
        if self.connection is not None:
            return
        if (time.time() - self.failure_time) < self.min_reconnect_interval:
            logger.warning(
                "Connection will not be re-established yet "
                "since it has not been long enough since "
                "the last failure"
            )
            return
        url = self.remote_url
        try:
            # Determine the URL to use for connection
            if self.plugin_name is not None:
                # Using plugin-based approach
                # Create a virtual URL that the adapter can recognize
                url = f"plugin://{self.plugin_name}"
                logger.info("Creating connector for plugin: %s", self.plugin_name)
            else:
                # Legacy remote_url approach
                if self.config.remote_url is None:
                    raise ValueError(
                        "remote_url must be provided when not using plugin_name"
                    )
                url = self.config.remote_url

            self.connection = CreateConnector(
                url,
                self.loop,
                self.local_cpu_backend,
                self.config,
                self.metadata,
                plugin_name=self.plugin_name,
            )
            if self.connection is None:
                raise RuntimeError(f"CreateConnector returned None for {url}")
            self._connect_ever_ok = True
            logger.info("Connection initialized/re-established at %s", url)
            try:
                self._try_attach_preload_provider()
            except Exception as attach_err:
                # Keep the connector. Attach is retried from
                # set_preload_coordinator; dropping it here re-inits SDK.
                logger.warning(
                    "Preload provider attach deferred after connect: %s",
                    attach_err,
                )
        except IrrecoverableException:
            logger.error("Irrecoverable error during connection initialization")
            raise
        except Exception as e:
            self.connection = None
            if not self._connect_ever_ok:
                raise IrrecoverableException(
                    f"Failed to initialize remote connection at {url}: {e}"
                ) from e
            with self.lock:
                self.failure_time = time.time()
            logger.warning("Failed to initialize/re-establish remote connection: %s", e)

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        if self.connection is None:
            logger.warning("Connection is None in contains, returning False")
            return False

        # For MLA worker id as 0 mode, use worker_id 0
        if self._mla_worker_id_as0_mode:
            key = key.with_new_worker_id(0)

        try:
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is self.loop:
                logger.error(
                    "RemoteBackend.contains called on StorageManager loop; "
                    "returning False to avoid deadlock"
                )
                return False
            if self.config.extra_config is not None and self.config.extra_config.get(
                "use_exists_sync", False
            ):
                return self.connection.exists_sync(key)
            future = asyncio.run_coroutine_threadsafe(
                self.connection.exists(key), self.loop
            )
            try:
                return future.result(self.config.blocking_timeout_secs)
            except TimeoutError:
                future.cancel()
                logger.warning(
                    "RemoteBackend.contains timed out after %s s",
                    self.config.blocking_timeout_secs,
                )
                return False
        except Exception as e:
            logger.warning("Remote connection failed in contains: %s", e)
            logger.warning("Returning False")
            return False

    def _remap_keys_mla(self, keys: List[CacheEngineKey]) -> List[CacheEngineKey]:
        if self._mla_worker_id_as0_mode:
            return [key.with_new_worker_id(0) for key in keys]
        return keys

    async def async_exists_mask(self, keys: List[CacheEngineKey]) -> List[bool]:
        n = len(keys)
        false_mask = [False] * n
        if n == 0:
            return []
        if self.connection is None:
            logger.warning("Connection is None in async_exists_mask, returning False")
            return false_mask
        probe_keys = self._remap_keys_mla(keys)

        async def _probe() -> List[bool]:
            connector_mask = getattr(self.connection, "async_exists_mask", None)
            if inspect.iscoroutinefunction(connector_mask):
                raw = await connector_mask(probe_keys)
            else:
                raw = await asyncio.gather(
                    *(self.connection.exists(k) for k in probe_keys)
                )
            if not isinstance(raw, (list, tuple)) or len(raw) != n:
                raise ValueError("async_exists_mask shape mismatch")
            out: List[bool] = []
            for item in raw:
                if type(item) is not bool:
                    raise ValueError("async_exists_mask non-bool entry")
                out.append(item)
            return out

        try:
            return await asyncio.wait_for(
                _probe(), self.config.blocking_timeout_secs
            )
        except Exception as e:
            logger.warning("Remote connection failed in async_exists_mask: %s", e)
            return false_mask

    def batched_contains(
        self,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        if self.connection is None:
            logger.warning("Connection is None in batched_contains, returning 0")
            return 0

        if not self.connection.support_batched_contains():
            return super().batched_contains(keys, pin)

        if self._mla_worker_id_as0_mode:
            keys = [key.with_new_worker_id(0) for key in keys]

        try:
            return self.connection.batched_contains(keys)
        except Exception as e:
            logger.warning("Remote connection failed in batched_contains: %s", e)
            return 0

    def prefix_contains_tokens(self, token_ids, n_keys: int = -1) -> int:
        if self.connection is None:
            return 0
        fn = getattr(self.connection, "prefix_contains_tokens", None)
        if not callable(fn):
            return 0
        try:
            return int(fn(token_ids, n_keys) or 0)
        except TypeError:
            return int(fn(token_ids) or 0)
        except Exception as e:
            logger.warning("Remote prefix_contains_tokens failed: %s", e)
            return 0

    def lookup_prefix_hit(self, token_ids, n_keys: int = -1):
        if self.connection is None:
            return 0, None
        fn = getattr(self.connection, "lookup_prefix_hit", None)
        if callable(fn):
            try:
                return fn(token_ids, n_keys)
            except TypeError:
                return fn(token_ids)
        blocks = self.prefix_contains_tokens(token_ids, n_keys)
        return blocks, None

    def kvb_store_memory_objs(self, token_ids, memory_objs) -> int:
        if self.connection is None:
            raise RuntimeError("Connection is None in kvb_store_memory_objs")
        fn = getattr(self.connection, "kvb_store_memory_objs", None)
        if not callable(fn):
            raise RuntimeError("connection missing kvb_store_memory_objs")
        return int(fn(token_ids, memory_objs) or 0)

    def kvb_get_prefix(self, memory_objs, hit=None) -> int:
        if self.connection is None:
            raise RuntimeError("Connection is None in kvb_get_prefix")
        fn = getattr(self.connection, "kvb_get_prefix", None)
        if not callable(fn):
            raise RuntimeError("connection missing kvb_get_prefix")
        return int(fn(memory_objs, hit) or 0)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.lock:
            return key in self.put_tasks

    def put_callback(self, future: Future, key: CacheEngineKey):
        with self.lock:
            self.put_tasks.discard(key)
        try:
            future.result()
        except Exception as e:
            self._put_failed_count += 1
            logger.error("Put task failed for key %s: %s", key, e)

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Future:
        """
        Submit a put task to store KV cache to remote storage asynchronously.

        :param on_complete_callback: Optional callback invoked after the remote
            write completes. Callback exceptions are caught and logged.
        """

        def create_immediate_empty_future() -> Future:
            f: Future = Future()
            f.set_result(None)
            return f

        if self.connection is None:
            logger.warning("Connection is None in submit_put_task, returning None")
            return create_immediate_empty_future()

        # If MLA worker id as 0 mode is enabled, skip put tasks
        if self._mla_worker_id_as0_mode:
            return create_immediate_empty_future()

        if self.exists_in_put_tasks(key):
            return create_immediate_empty_future()

        memory_obj.ref_count_up()

        with self.lock:
            self.put_tasks.add(key)

        compressed_memory_obj = self.serializer.serialize(memory_obj)
        memory_obj.ref_count_down()

        def put_done_callback(f: Future) -> None:
            self.put_callback(f, key)
            if on_complete_callback is not None:
                try:
                    on_complete_callback(key)
                except Exception as e:
                    logger.warning("on_complete_callback failed for key %s: %s", key, e)

        # NOTE: No need to do error handling here
        # since the `future` is never waited
        future = asyncio.run_coroutine_threadsafe(
            self.connection.put(key, compressed_memory_obj), self.loop
        )
        future.add_done_callback(put_done_callback)
        return future

    def _gds_blocking_host_put(self) -> bool:
        if str(getattr(self.config, "kv_io_mode", "dram") or "dram").lower() != "gds":
            return False
        return not bool(getattr(self.config, "async_store_d2h_enabled", False))

    def batched_put_callback(self, future: Future, keys: List[CacheEngineKey]):
        """
        Callback function for batched put tasks.
        """
        with self.lock:
            self.put_tasks.difference_update(keys)
        try:
            future.result()
        except Exception as e:
            self._put_failed_count += 1
            logger.error(
                "Batched put task failed for %s keys: %s", len(keys), e
            )

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        Submit batched put tasks to store KV caches to remote storage.

        :param on_complete_callback: Optional callback invoked once per key
            after that key's write completes (not once per batch).
        """
        if self.connection is None:
            logger.warning(
                "Connection is None in batched_submit_put_task, returning None"
            )
            return
        if self.connection.support_batched_put():
            if self._mla_worker_id_as0_mode:
                return

            # First, increment reference counts for all objects
            for memory_obj in memory_objs:
                memory_obj.ref_count_up()

            compressed_memory_objs = []
            try:
                for memory_obj in memory_objs:
                    compressed_memory_objs.append(self.serializer.serialize(memory_obj))
            finally:
                # Always decrement reference counts for all objects,
                # regardless of whether serialization succeeded or failed
                for memory_obj in memory_objs:
                    memory_obj.ref_count_down()

            def batched_done_callback(f: Future) -> None:
                self.batched_put_callback(f, list(keys))
                # Invoke per-key callback for each key in the batch
                if on_complete_callback is not None:
                    for key in keys:
                        try:
                            on_complete_callback(key)
                        except Exception as e:
                            logger.warning(
                                "on_complete_callback failed for key %s: %s", key, e
                            )

            future = asyncio.run_coroutine_threadsafe(
                self.connection.batched_put(keys, compressed_memory_objs),  # type: ignore
                self.loop,
            )
            future.add_done_callback(batched_done_callback)
            if self._gds_blocking_host_put():
                try:
                    future.result(timeout=self.config.blocking_timeout_secs)
                except TimeoutError:
                    future.cancel()
                    logger.error(
                        "GDS batched put timed out after %s s",
                        self.config.blocking_timeout_secs,
                    )
                    raise
        else:
            for key, memory_obj in zip(keys, memory_objs, strict=False):
                self.submit_put_task(
                    key, memory_obj, on_complete_callback=on_complete_callback
                )

    @_lmcache_nvtx_annotate
    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Blocking get function.
        """
        # Check if local_cpu_backend is available (required for memory allocation)
        if self.local_cpu_backend is None:
            logger.warning(
                "local_cpu_backend is None in get_blocking "
                "(likely scheduler role), returning None"
            )
            return None

        if self.connection is None:
            logger.warning("Connection is None in get_blocking, returning None")
            return None
        # For MLA worker id as 0 mode, use worker_id 0
        if self._mla_worker_id_as0_mode:
            key = key.with_new_worker_id(0)
        t1 = time.perf_counter()
        future = asyncio.run_coroutine_threadsafe(self.connection.get(key), self.loop)

        try:
            memory_obj = future.result(self.config.blocking_timeout_secs)
        except Exception as e:
            if isinstance(e, TimeoutError):
                logger.warning("get blocking timeout, trigger cancel the future task")
                future.cancel()
            logger.warning("Error occurred in get_blocking: %s, return None", e)
            memory_obj = None

        t2 = time.perf_counter()
        self.stats_monitor.update_interval_remote_time_to_get_sync((t2 - t1) * 1000)
        if memory_obj is None:
            self._get_blocking_failed_count += 1
            return None
        decompressed_memory_obj = self.deserializer.deserialize(memory_obj)
        t3 = time.perf_counter()
        logger.debug(
            "Get takes %.6f msec, deserialization takes %.6f msec",
            (t2 - t1) * 1000,
            (t3 - t2) * 1000,
        )
        return decompressed_memory_obj

    @property
    def get_blocking_failed_count(self):
        return self._get_blocking_failed_count

    @property
    def put_failed_count(self):
        return self._put_failed_count

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        # Check if local_cpu_backend is available (required for memory allocation)
        if self.local_cpu_backend is None:
            logger.warning(
                "local_cpu_backend is None in batched_get_blocking "
                "(likely scheduler role), returning None list"
            )
            return [None] * len(keys)

        if self.connection is None:
            logger.warning("Connection is None in batched_get_blocking, returning None")
            return [None] * len(keys)

        # For MLA worker id as 0 mode, use worker_id 0
        if self._mla_worker_id_as0_mode:
            keys = [key.with_new_worker_id(0) for key in keys]

        t1 = time.perf_counter()
        # batched get
        if self.connection.support_batched_get():
            future = asyncio.run_coroutine_threadsafe(
                self.connection.batched_get(keys), self.loop
            )
            try:
                memory_objs = future.result(self.config.blocking_timeout_secs)
            except Exception as e:
                if isinstance(e, TimeoutError):
                    logger.warning(
                        "batched get blocking timeout, trigger cancel the future task"
                    )
                    future.cancel()
                else:
                    logger.warning(
                        "Error occurred in batched_get_blocking: %s, "
                        "returning None list",
                        e,
                    )
                memory_objs = [None] * len(keys)
        else:
            remote_backend_individual_get_stats: dict[
                CacheEngineKey, dict[str, float]
            ] = {}
            retrieve_stats = self.stats_monitor.get_current_retrieve_stats()
            if retrieve_stats is not None:
                retrieve_stats.detailed_metrics[
                    "remote_backend_individual_get_stats"
                ] = remote_backend_individual_get_stats

            futures = [
                asyncio.run_coroutine_threadsafe(self.connection.get(key), self.loop)
                for key in keys
            ]
            memory_objs = []
            failed = False
            for fut in futures:
                if not failed:
                    try:
                        memory_obj = fut.result(self.config.blocking_timeout_secs)
                    except Exception as e:
                        failed = True
                        if isinstance(e, TimeoutError):
                            logger.warning(
                                "get blocking timeout, trigger cancel the future task"
                            )
                            fut.cancel()
                        else:
                            logger.warning(
                                "Error occurred in get_blocking: %s, returning None", e
                            )
                        memory_obj = None
                    memory_objs.append(memory_obj)
                else:
                    memory_objs.append(None)
                    fut.cancel()

        t2 = time.perf_counter()
        duration = t2 - t1
        self.stats_monitor.update_interval_remote_time_to_get_sync(duration * 1000)

        retrieve_stats = self.stats_monitor.get_current_retrieve_stats()
        if retrieve_stats is not None:
            retrieve_stats.detailed_metrics[
                "remote_backend_batched_get_blocking_time"
            ] = (
                retrieve_stats.detailed_metrics.get(
                    "remote_backend_batched_get_blocking_time", 0.0
                )
                + duration
            )
        decompressed_memory_objs: list[Optional[MemoryObj]] = []
        error_happened = False
        for memory_obj in memory_objs:
            if memory_obj is None:
                error_happened = True
                decompressed_memory_objs.append(None)
            else:
                decompressed_memory_objs.append(
                    self.deserializer.deserialize(memory_obj)
                )
        if error_happened:
            self._get_blocking_failed_count += 1

        assert len(decompressed_memory_objs) == len(keys), (
            f"keys length: {len(keys)}, "
            f"decompressed memory objs length: {len(decompressed_memory_objs)}"
        )
        return decompressed_memory_objs

    async def support_batched_async_contains(self) -> bool:
        return (
            self.connection is not None
            and self.connection.support_batched_async_contains()
        )

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        if self.connection is None:
            logger.warning("Connection is None in batched_async_contains, returning 0")
            return 0
        if self._mla_worker_id_as0_mode:
            keys = [key.with_new_worker_id(0) for key in keys]

        try:
            assert self.connection.support_batched_async_contains(), (
                f"Connector {self.connection} does not support batched async contains"
            )
            # warning, this timeout will not actually stop the
            # scheduler from waiting for the result
            return await asyncio.wait_for(
                self.connection.batched_async_contains(lookup_id, keys, pin),
                self.config.blocking_timeout_secs,
            )
        except asyncio.TimeoutError:
            logger.warning("batched_async_contains timed out")
            return 0
        except Exception as e:
            logger.warning("Error occurred in batched_async_contains: %s", e)
            return 0

    async def support_batched_get_non_blocking(self) -> bool:
        return (
            self.connection is not None
            and self.connection.support_batched_get_non_blocking()
        )

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> List[MemoryObj]:
        # Check if local_cpu_backend is available (required for memory allocation)
        if self.local_cpu_backend is None:
            logger.warning(
                "local_cpu_backend is None in batched_get_non_blocking "
                "(likely scheduler role), returning empty list"
            )
            return []

        if self.connection is None:
            logger.warning(
                "Connection is None in batched_get_non_blocking, returning empty list"
            )
            return []
        try:
            # warning, this timeout will not actually stop the
            # scheduler from waiting for the result
            get_fn = self.connection.batched_get_non_blocking
            try:
                coro = get_fn(lookup_id, keys, transfer_spec)
            except TypeError:
                coro = get_fn(lookup_id, keys)
            return await asyncio.wait_for(
                coro,
                self.config.blocking_timeout_secs,
            )
        except asyncio.TimeoutError:
            logger.warning("batched_get_non_blocking timed out")
            return []
        except Exception as e:
            logger.warning("Error occurred in batched_get_non_blocking: %s", e)
            return []

    def pin(self, key: CacheEngineKey) -> bool:
        logger.debug(
            "Remote backend does not support pin. "
            "This method is a no-op and will return True."
        )
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        logger.debug(
            "Remote backend does not support unpin. "
            "This method is a no-op and will return True."
        )
        return True

    def remove(self, key, force=True):
        if self.connection is None:
            logger.warning("Connection is None in remove, returning False")
            return False

        try:
            return self.connection.remove_sync(key)
        except Exception as e:
            logger.exception(
                "Failed to remove key %s from remote backend, error: %s", key, e
            )
            return False

    def get_allocator_backend(self):
        assert self.local_cpu_backend is not None, (
            "local_cpu_backend is required for get_allocator_backend, "
            "should not be called in scheduler role"
        )
        return self.local_cpu_backend

    def close(self):
        try:
            assert self.connection is not None
            future = asyncio.run_coroutine_threadsafe(
                self.connection.close(), self.loop
            )
            future.result()
            logger.info("Remote backend closed.")
        except Exception as e:
            logger.warning("Error occurred when closing remote connection: %s", e)
