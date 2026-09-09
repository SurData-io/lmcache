# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from concurrent.futures import Future
from typing import (
    TYPE_CHECKING,
    Any,
    Coroutine,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)
import asyncio
import functools
import inspect
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.observability import PrometheusLogger
from lmcache.utils import (
    CacheEngineKey,
    _lmcache_nvtx_annotate,
    start_loop_in_thread_with_exceptions,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import (
    EventManager,
    EventStatus,
    EventType,
    release_prefetch_result,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend import CreateStorageBackends, is_cuda_worker
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
    StorageBackendInterface,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.preload_lifecycle import (
    PreloadLifecycleCoordinator,
    PreloadWork,
    find_provider,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker
    from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
        LMCacheAsyncLookupServer,
    )

logger = init_logger(__name__)


async def gather_loading_results_with_keys(
    loading_tasks: Sequence[asyncio.Future],
    loading_task_keys: Sequence[Sequence[CacheEngineKey]],
) -> list[list[tuple[CacheEngineKey, MemoryObj]]]:
    """Wait for all tier GET tasks. Failed tiers become empty results."""
    loading_results = await asyncio.gather(*loading_tasks, return_exceptions=True)
    paired: list[list[tuple[CacheEngineKey, MemoryObj]]] = []
    for keys, results in zip(loading_task_keys, loading_results, strict=False):
        if isinstance(results, BaseException):
            logger.error("prefetch tier GET failed: %s", results)
            paired.append([])
            continue
        paired.append(list(zip(keys, results, strict=False)))
    return paired


# Helper function to get the class name of the backend
def get_backend_cname(backend: StorageBackendInterface) -> str:
    return backend.__class__.__name__


# Helper function to allocate and copy memory objects between D and H
def allocate_and_copy_objects(
    allocator_backend: AllocatorBackendInterface,
    keys: Sequence[CacheEngineKey],
    src_memory_objs: list[MemoryObj],
    stream: Any,
) -> tuple[Sequence[CacheEngineKey], list[MemoryObj]]:
    """
    Allocate the memory objects and copy the data from src_memory_objs to
    the newly allocated memory objects

    Args:
        allocator_backend: the allocator backend to allocate the new memory
          objects
        keys: the cache engine keys corresponding to the memory objects
        src_memory_objs: the memory objects to copy from
        stream: the device-specific GPU stream to run the copy in
            (e.g., torch_dev.Stream on CUDA or XPU)

    Returns:
        - list of cache engine keys that corresponds to the memory objects
          that has been successfully allocated
        - list of the memory objects that has been successfully allocated
    """
    allocated_objects = []
    for key, src_memory_obj in zip(keys, src_memory_objs, strict=False):
        if allocator_backend.contains(key):
            continue
        memory_obj = allocator_backend.allocate(
            src_memory_obj.get_shape(),
            src_memory_obj.get_dtype(),
            fmt=src_memory_obj.meta.fmt,
            eviction=True,
            busy_loop=False,
        )

        if memory_obj is None:
            break

        if memory_obj.tensor is None:
            # This should not happen with current implementation,
            # but handle it defensively to avoid memory leak
            logger.warning(
                "Allocated MemoryObj has None tensor, this is unexpected. "
                "Releasing the memory object."
            )
            memory_obj.ref_count_down()
            break

        with torch_dev.stream(stream):
            memory_obj.tensor.copy_(src_memory_obj.tensor, non_blocking=True)
        allocated_objects.append(memory_obj)

    if stream is not None:
        stream.synchronize()
    return keys[: len(allocated_objects)], allocated_objects


class WeightedSemaphore:
    def __init__(self, chunk_budget: int):
        # it is physically impossible to have more fragmentation than 50%
        # when all of the chunks are of the same size (save_unfull_chunk=False)
        # so we can safely allocate half of the chunk budget for concurrent requests
        self._concurrent_budget_cap = chunk_budget // 2
        self._chunk_budget_cap = chunk_budget
        self._current_chunks = self._concurrent_budget_cap
        self._cond = asyncio.Condition()

    async def acquire(self, n: int = 1) -> None:
        if n > self._chunk_budget_cap:
            raise ValueError(
                f"Trying to acquire {n} chunks, "
                f"Cannot acquire more than {self._chunk_budget_cap} chunks"
                "Please set the max local cpu size to a larger value"
            )

        async with self._cond:
            logger.debug("WeightedSemaphore: Attempting to acquire %d chunks", n)
            if n <= self._concurrent_budget_cap:
                await self._cond.wait_for(lambda: self._current_chunks >= n)
                self._current_chunks -= n
            else:
                # Oversized case: require exclusive access
                await self._cond.wait_for(
                    lambda: self._current_chunks == self._concurrent_budget_cap
                )
                # Reserve everything
                self._current_chunks = 0
            logger.debug(
                "WeightedSemaphore: Acquired %d chunks, remaining chunks: %d",
                n,
                self._current_chunks,
            )

    async def release(self, n: int = 1) -> None:
        async with self._cond:
            if n <= self._concurrent_budget_cap:
                self._current_chunks += n
            else:
                self._current_chunks = self._concurrent_budget_cap
            self._cond.notify_all()


class AsyncMultiSerializer:
    """
    Prevent race conditions where multiple batched_get's cause the local CPU
    backend to allocate memory objects in parallel and get deadlocked.
    Make the assumption that the save_unfull_chunk is False so that we
    can assume that we can always use 50% of the given memory
    """

    def __init__(
        self,
        allocator_backend: AllocatorBackendInterface,
        loop: asyncio.AbstractEventLoop,
    ):
        self.chunk_budget = allocator_backend.calculate_chunk_budget()
        self._sem = WeightedSemaphore(self.chunk_budget)
        self.loop = loop

    async def run(
        self,
        coro_fn: Coroutine[Any, Any, Any],
        num_chunks: int,
    ) -> Any:
        await self._sem.acquire(num_chunks)
        try:
            return await coro_fn
        finally:
            await self._sem.release(num_chunks)


class AsyncSingleSerializer:
    """
    Prevent race conditions in a naive way by forcing each request that
    is passed through to be serialized
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        # lazy init in run
        self.lock: Optional[asyncio.Lock] = None

    async def run(self, coro_fn: Coroutine[Any, Any, Any], *args, **kwargs) -> Any:
        # we need to lazily initialize the lock to
        # place it on the calling event loop
        if self.lock is None:
            self.lock = asyncio.Lock()
        async with self.lock:  # type: ignore
            return await coro_fn


AsyncSerializer = Union[AsyncSingleSerializer, AsyncMultiSerializer]


# TODO: extend this class to implement caching policies and eviction policies
class StorageManager:
    """
    The StorageManager is responsible for managing the storage backends.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        event_manager: EventManager,
        lmcache_worker: Optional["LMCacheWorker"] = None,
        async_lookup_server: Optional["LMCacheAsyncLookupServer"] = None,
    ):
        self.config = config
        self.metadata = metadata
        self.loop = asyncio.new_event_loop()

        self.thread = threading.Thread(
            target=start_loop_in_thread_with_exceptions,
            args=(self.loop,),
            name="storage-manager-event-loop",
        )
        self.thread.start()

        self.storage_backends: OrderedDict[str, StorageBackendInterface] = OrderedDict()
        self.manager_lock = threading.Lock()
        self.lmcache_worker = lmcache_worker

        # Use the unified create path so that init and
        # dynamic creation share the same logic.
        self.create_backends()
        self.preload_coordinator = PreloadLifecycleCoordinator(self.loop)
        extra = getattr(config, "extra_config", None) or {}
        stall_ms = extra.get("surdata.preload_head_stall_timeout_ms", 600000)
        try:
            self.preload_coordinator.stall_timeout_s = max(0.0, float(stall_ms) / 1000.0)
        except (TypeError, ValueError):
            self.preload_coordinator.stall_timeout_s = 600.0
        for backend in self.storage_backends.values():
            setter = getattr(backend, "set_preload_coordinator", None)
            if callable(setter):
                setter(self.preload_coordinator)
        try:
            self.preload_coordinator.attach_provider(
                find_provider(self.storage_backends)
            )
        except RuntimeError:
            logger.exception("preload lifecycle provider ABI mismatch")
            raise

        # the backend used for actual storage
        self.non_allocator_backends = self.get_non_allocator_backends()

        self.enable_pd = config.enable_pd

        self.allocator_backend = None
        if metadata.role != "scheduler":
            self.allocator_backend = self._get_allocator_backend(config)

        self.local_cpu_backend = self.storage_backends.get("LocalCPUBackend", None)

        self.instance_id = config.lmcache_instance_id
        self.worker_id = metadata.worker_id

        self.event_manager = event_manager
        self._lookup_acked: set[str] = set()

        self.async_lookup_server: Optional["LMCacheAsyncLookupServer"] = (
            async_lookup_server
        )
        self.async_serializer: Optional[AsyncSerializer] = None

        # The GPU stream for internal copies during put
        if is_cuda_worker(metadata):
            self.internal_copy_stream = torch_dev.Stream()
        else:
            self.internal_copy_stream = None

        # freeze mode: only use local_cpu backend for retrieval
        self._freeze = False
        self._freeze_lock = threading.RLock()

        # Backend bypass mode: skip specific backends during health check failures
        self._bypassed_backends: set[str] = set()
        self._bypass_lock = threading.RLock()

        if not self.enable_pd and self.config.enable_async_loading:
            assert self.allocator_backend is not None
            self.async_serializer = AsyncSingleSerializer(self.loop)

        self._setup_metrics()

    def _setup_metrics(self) -> None:
        prometheus_logger = PrometheusLogger.GetOrCreate(
            self.metadata,
            config=self.config,
        )

        metric_map = {
            "storage_events_ongoing_count": EventStatus.ONGOING,
            "storage_events_done_count": EventStatus.DONE,
            "storage_events_not_found_count": EventStatus.NOT_FOUND,
        }

        for metric_name, status in metric_map.items():
            metric = getattr(prometheus_logger, metric_name)
            metric.set_function(
                lambda s=status: self.event_manager.get_events_count_by_status(
                    EventType.LOADING, s
                )
            )

    def _get_allocator_backend(
        self, config: LMCacheEngineConfig
    ) -> AllocatorBackendInterface:
        if self.enable_pd:
            allocator_backend = self.storage_backends["PDBackend"]
        elif "MaruBackend" in self.storage_backends:
            if "LocalCPUBackend" in self.storage_backends:
                allocator_backend = self.storage_backends["LocalCPUBackend"]
            else:
                allocator_backend = self.storage_backends["MaruBackend"]
        else:
            allocator_backend = self.storage_backends["LocalCPUBackend"]
        assert isinstance(allocator_backend, AllocatorBackendInterface)
        return allocator_backend

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction=True,
        busy_loop=True,
    ) -> Optional[MemoryObj]:
        """
        Allocate memory object with memory allocator.
        Use LRU evictor if eviction is enabled.
        """
        # TODO (Jiayi): We might need to pre-allocate and management
        # disk in a similar way as CPU.
        assert self.allocator_backend is not None
        return self.allocator_backend.allocate(
            shapes, dtypes, fmt, eviction=eviction, busy_loop=busy_loop
        )

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction=True,
        busy_loop=True,
    ) -> Optional[list[MemoryObj]]:
        """
        Batched allocate memory object with memory allocator.
        Use LRU evictor if eviction is enabled.
        """
        # TODO (Jiayi): We might need to pre-allocate and management
        # disk in a similar way as CPU.
        if self.allocator_backend is None:
            raise RuntimeError("Allocator backend not available for scheduler role")
        return self.allocator_backend.batched_allocate(
            shapes, dtypes, batch_size, fmt, eviction=eviction, busy_loop=busy_loop
        )

    def put(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        location: Optional[str] = None,
    ) -> None:
        """
        Non-blocking function to put the memory object into the storages.
        Do not store if the same object is being stored (handled here by
        storage manager) or has been stored (handled by storage backend).
        """
        raise RuntimeError(
            "StorageManager.put is deprecated and should not be called anymore"
        )

    def batched_put(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
        location: Optional[str] = None,
    ) -> None:
        """
        Non-blocking function to batched put the memory objects into the
        storage backends.
        Do not store if the same object is being stored (handled here by
        storage manager) or has been stored (handled by storage backend).
        """
        # The dictionary from backend cname to objects and keys
        obj_dict: dict[
            str,
            tuple[Sequence[CacheEngineKey], list[MemoryObj]],
        ] = {}
        if self.allocator_backend is None:
            # For scheduler role, no allocator backend available
            raise RuntimeError("Batched put not available for scheduler role")
        obj_dict[get_backend_cname(self.allocator_backend)] = (
            keys,
            memory_objs,
        )

        for backend_name, backend in self.storage_backends.items():
            if location and backend_name != location:
                continue
            # Skip bypassed backends
            with self._bypass_lock:
                if backend_name in self._bypassed_backends:
                    continue

            allocator_backend = backend.get_allocator_backend()
            cname = get_backend_cname(allocator_backend)
            if cname not in obj_dict:
                new_keys, new_objs = allocate_and_copy_objects(
                    allocator_backend, keys, memory_objs, self.internal_copy_stream
                )
                obj_dict[cname] = (new_keys, new_objs)

            # NOTE: the handling of exists_in_put_tasks
            # is done in the backend
            ks, objs = obj_dict[cname]
            backend.batched_submit_put_task(ks, objs, transfer_spec=transfer_spec)

        for cname, (ks, objs) in obj_dict.items():
            for memory_obj in objs:
                memory_obj.ref_count_down()

    def get(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """
        Blocking function to get the memory object from the storages.
        """

        # Search all backends for blocking get
        for backend_name, backend in self.get_active_storage_backends(location):
            # TODO(Jiayi): need to make sure all memory_objs returned
            # are allocated by the allocator backend.
            memory_obj = backend.get_blocking(key)
            if memory_obj:
                if (
                    backend_name not in ["LocalCPUBackend", "PDBackend", "MaruBackend"]
                    and "LocalCPUBackend" in self.storage_backends
                ):
                    local_cpu_backend = self.storage_backends["LocalCPUBackend"]
                    assert isinstance(local_cpu_backend, LocalCPUBackend)
                    local_cpu_backend.submit_put_task(key, memory_obj)
                return memory_obj

        return None

    def get_non_blocking(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[Future]:
        """
        Non-blocking function to get the memory object from the storages.
        """
        # TODO (Jiayi): incorporate prefetching here

        # Search all backends for non-blocking get
        for backend_name, backend in self.get_active_storage_backends(location):
            # NOTE(Jiayi): bypass the allocator for now
            task = backend.get_non_blocking(key)
            if task:
                # TODO (Jiayi): add write-back logic here
                return task
        return None

    def batched_get(
        self,
        keys: List[CacheEngineKey],
        location: Optional[str] = None,
    ) -> List[Optional[MemoryObj]]:
        """
        Blocking function to get the memory objects from the storages.
        """
        # TODO (ApostaC): remove the nested optional here
        for backend_name, storage_backend in self.get_active_storage_backends(location):
            memory_objs = storage_backend.batched_get_blocking(keys)
            if memory_objs:
                # Align with single-key `get()` logic:
                # auto-write remote data to local CPU cache
                if (
                    backend_name not in ["LocalCPUBackend", "PDBackend", "MaruBackend"]
                    and "LocalCPUBackend" in self.storage_backends
                    and None not in memory_objs
                ):
                    logger.debug(
                        "Storing %s objects from %s to LocalCPUBackend",
                        len(keys),
                        backend_name,
                    )
                    local_cpu_backend = self.storage_backends["LocalCPUBackend"]
                    assert isinstance(local_cpu_backend, LocalCPUBackend)
                    # Type cast: Safe (we verified no Nones above)
                    # `batched_submit_put_task` expects list[MemoryObj]
                    # TODO (lisiG9): Refactor this write-back logic into caching
                    #  policy module
                    memory_objs_no_none = cast(List[MemoryObj], memory_objs)
                    local_cpu_backend.batched_submit_put_task(keys, memory_objs_no_none)
                return memory_objs
        return [None] * len(keys)

    def layerwise_batched_get(
        self,
        keys: List[List[CacheEngineKey]],
        location: Optional[str] = None,
    ) -> Generator[Future, None, None]:
        """
        Non-blocking function to get the memory objects into the storages
        in a layerwise manner.
        Do not store if the same object is being stored (handled here by
        storage manager) or has been stored (handled by storage backend).

        :param List[List[CacheEngineKey]] keys: The keys to get. The first
            dimension corresponds to the number of layers, and the second
            dimension corresponds to the number of chunks.

        :return: A generator that yields a future for each layer.
        """
        if location is None:
            location = "LocalCPUBackend"
        for keys_multi_chunk in keys:
            # Retrieve all chunks for one layer
            backend = self.storage_backends[location]
            # TODO(Jiayi): need to make async loading and layerwise compatible
            coro = backend.batched_get_non_blocking("fake_lookup_id", keys_multi_chunk)
            task = asyncio.run_coroutine_threadsafe(coro, self.loop)
            yield task

    def prefetch_single_done_callback(
        self,
        future: asyncio.Future,
        keys: list[CacheEngineKey],
        backend_name: str,
    ) -> None:
        """
        Callback function when a single prefetch task
        (i.e., prefetching from a single backend) is done.
        """
        # TODO(Jiayi): support write-back policy here
        pass

    def prefetch_all_done_callback(
        self,
        task: asyncio.Future,
        lookup_id: str,
        cum_chunk_lengths_total: list[int],
        tier_expected_chunks: list[int],
        keys_per_chunk: int = 1,
    ) -> None:
        """
        Callback function when all prefetch tasks
        (i.e., prefetching from all backends for the entire request) are done.

        :param int keys_per_chunk: Number of storage keys per logical chunk
            (``num_layers`` for layerwise mode, ``1`` otherwise). Used to
            convert each tier's per-key result count back to chunk units.
        """
        assert self.async_lookup_server is not None
        try:
            self.event_manager.update_event_status(
                EventType.LOADING, lookup_id, status=EventStatus.DONE
            )
            res = task.result()
        except Exception:
            logger.exception(
                "prefetch gather failed lookup_id=%s; LOADING marked DONE with error",
                lookup_id,
            )
            self.event_manager.claim_done_event(
                EventType.LOADING, lookup_id, only_if_aborted=True
            )
            return

        claimed = self.event_manager.claim_done_event(
            EventType.LOADING, lookup_id, only_if_aborted=True
        )
        if claimed is not None:
            release_prefetch_result(res)
            logger.info(
                "LOADING aborted lookup id %s released prefetch result",
                lookup_id,
            )
            return
        if (
            self.event_manager.get_event_status(EventType.LOADING, lookup_id)
            == EventStatus.NOT_FOUND
        ):
            # cleanup_memory_objs already claimed the DONE event.
            return

        # Calculate total retrieved chunks across all tiers based on actual results
        # from batched_get_non_blocking, not the batched_async_contains results.
        # This handles the case where chunks may be evicted between contains check
        # and actual retrieval.
        #
        # Example: chunk_size=256, 7 chunks total (1792 tokens) across 3 tiers
        #   cum_chunk_lengths_total = [0, 256, 512, 768, 1024, 1280, 1536, 1792]
        #   tier_expected_chunks = [3, 2, 2]  # Tier 0: 3, Tier 1: 2, Tier 2: 2
        #
        #   Chunks:
        #   [0 1 2 3 4 5 6]
        #   |-----|          <--- stored in Tier0, tier_expected_chunks[0]==3
        #         |---|      <--- stored in Tier1, tier_expected_chunks[1]==2
        #             |---|  <--- stored in Tier2, tier_expected_chunks[2]==2
        #
        # Case 1: All chunks retrieved successfully
        #   [0 1 2 3 4 5 6]
        #   |-----|          <--- Tier0: retrieved 3 chunks (obj0, obj1, obj2)
        #         |---|      <--- Tier1: retrieved 2 chunks (obj3, obj4)
        #             |---|  <--- Tier2: retrieved 2 chunks (obj5, obj6)
        #   res = [[obj0, obj1, obj2], [obj3, obj4], [obj5, obj6]]
        #   total_retrieved_chunks = 7
        #   retrieved_length = cum_chunk_lengths_total[7] = 1792
        #
        # Case 2: Tier 1 only got 1 chunk (eviction), Tier 2 got all 2 chunks
        #   [0 1 2 3 4 5 6]
        #   |-----|          <--- Tier0: retrieved 3 chunks (obj0, obj1, obj2)
        #         |-|X|      <--- Tier1: retrieved 1 chunk (obj3), missing obj4
        #             |---|  <--- Tier2: retrieved 2 chunks (obj5, obj6) - IGNORED
        #   res = [[obj0, obj1, obj2], [obj3], [obj5, obj6]]
        #   total_retrieved_chunks = 4 (stop at tier 1, tier 2 chunks ignored)
        #   retrieved_length = cum_chunk_lengths_total[4] = 1024
        #   Note: Even though tier 2 successfully retrieved 2 chunks, they are
        #   not counted because tier 1 has a gap, breaking prefix continuity.
        #
        # Case 3: Tier 0 only got 2 chunks (eviction), other tiers got all
        #   [0 1 2 3 4 5 6]
        #   |---|X|          <--- Tier0: retrieved 2 chunks (obj0, obj1), missing obj2
        #         |---|      <--- Tier1: retrieved 2 chunks (obj3, obj4) - IGNORED
        #             |---|  <--- Tier2: retrieved 2 chunks (obj5, obj6) - IGNORED
        #   res = [[obj0, obj1], [obj3, obj4], [obj5, obj6]]
        #   total_retrieved_chunks = 2 (stop at tier 0, all subsequent ignored)
        #   retrieved_length = cum_chunk_lengths_total[2] = 512
        total_retrieved_chunks = 0
        for tier_idx, tier_result in enumerate(res):
            # `tier_result` is a list of (key, mem_obj) pairs, one per
            # storage key. With layerwise on, each logical chunk maps to
            # keys_per_chunk per-layer keys, so divide to get chunk count.
            # We round down so a partially-retrieved chunk (e.g., one
            # layer evicted) is treated as a miss.
            actual_chunks = len(tier_result) // keys_per_chunk
            expected_chunks = tier_expected_chunks[tier_idx]
            total_retrieved_chunks += actual_chunks

            # Release the tail rounded off by actual_chunks; else staging buffer leaks.
            tail_start = actual_chunks * keys_per_chunk
            for _, mem_obj in tier_result[tail_start:]:
                mem_obj.ref_count_down()

            # If a tier retrieved fewer chunks than expected, we stop counting
            # because subsequent chunks are not contiguous
            if actual_chunks < expected_chunks:
                # Release all chunks in subsequent tiers since they won't be used
                for subsequent_tier in res[tier_idx + 1 :]:
                    for _, mem_obj in subsequent_tier:
                        mem_obj.ref_count_down()
                break

        retrieved_length = cum_chunk_lengths_total[total_retrieved_chunks]
        logger.info(
            "LOADING done lookup id %s retrieved length %s (lookup ACK already sent)",
            lookup_id,
            retrieved_length,
        )

    def _local_cpu_backend_name(self) -> str:
        return "LocalCPUBackend"

    def _coerce_bool_mask(self, mask: Any, n: int) -> Optional[List[bool]]:
        if not isinstance(mask, (list, tuple)) or len(mask) != n:
            return None
        out: List[bool] = []
        for item in mask:
            if type(item) is not bool:
                return None
            out.append(item)
        return out

    def _backend_has_async_exists_mask(self, backend: StorageBackendInterface) -> bool:
        fn = getattr(backend, "async_exists_mask", None)
        return inspect.iscoroutinefunction(fn)

    async def _backend_exists_mask(
        self,
        backend_name: str,
        backend: StorageBackendInterface,
        keys: List[CacheEngineKey],
    ) -> List[bool]:
        n = len(keys)
        false_mask = [False] * n
        if n == 0:
            return []
        if self._backend_has_async_exists_mask(backend):
            try:
                raw = await backend.async_exists_mask(keys)
            except Exception:
                logger.exception(
                    "async_exists_mask failed backend=%s", backend_name
                )
                return false_mask
            coerced = self._coerce_bool_mask(raw, n)
            if coerced is None:
                logger.error(
                    "async_exists_mask returned invalid mask backend=%s "
                    "expected_len=%s",
                    backend_name,
                    n,
                )
                return false_mask
            return coerced
        if backend_name == self._local_cpu_backend_name():
            return [bool(backend.contains(k, False)) for k in keys]

        def _sync_all() -> List[bool]:
            return [bool(backend.contains(k, False)) for k in keys]

        try:
            return await asyncio.to_thread(_sync_all)
        except Exception:
            logger.exception(
                "sync exists mask failed backend=%s", backend_name
            )
            return false_mask

    async def _union_prefix_owners_async(
        self,
        keys: List[CacheEngineKey],
        keys_per_chunk: int,
        search_range: Optional[List[str]] = None,
        token_ids=None,
    ) -> List[str]:
        """Per-chunk any-tier prefix. Stop at the first chunk missing on all tiers."""
        if keys_per_chunk < 1:
            raise ValueError(f"keys_per_chunk must be >= 1, got {keys_per_chunk}")
        n_chunks = len(keys) // keys_per_chunk
        backends = list(self.get_active_storage_backends(search_range=search_range))
        if not backends or n_chunks == 0:
            return []
        cpu_name = self._local_cpu_backend_name()
        remote_prefix: dict[str, int] = {}
        mask_backends = []
        for backend_name, backend in backends:
            prefix_fn = getattr(backend, "prefix_contains_tokens", None)
            if backend_name != cpu_name and callable(prefix_fn):
                if token_ids is None:
                    remote_prefix[backend_name] = 0
                else:
                    try:
                        remote_prefix[backend_name] = int(
                            prefix_fn(token_ids, n_chunks) or 0
                        )
                    except TypeError:
                        remote_prefix[backend_name] = int(prefix_fn(token_ids) or 0)
                continue
            mask_backends.append((backend_name, backend))
        masks = []
        if mask_backends:
            masks = await asyncio.gather(
                *[
                    self._backend_exists_mask(name, backend, keys)
                    for name, backend in mask_backends
                ]
            )
        owners: List[str] = []
        for ci in range(n_chunks):
            start = ci * keys_per_chunk
            end = start + keys_per_chunk
            cpu_hit = False
            other: Optional[str] = None
            for (backend_name, _backend), mask in zip(
                mask_backends, masks, strict=False
            ):
                if not all(mask[start:end]):
                    continue
                if backend_name == cpu_name:
                    cpu_hit = True
                elif other is None:
                    other = backend_name
            if not cpu_hit:
                for backend_name, _backend in backends:
                    if backend_name == cpu_name:
                        continue
                    if ci < int(remote_prefix.get(backend_name, 0) or 0) and other is None:
                        other = backend_name
            if cpu_hit:
                owners.append(cpu_name)
            elif other is not None:
                owners.append(other)
            else:
                break
        return owners

    async def _pin_backend_keys(
        self,
        backend_name: str,
        backend: StorageBackendInterface,
        keys: List[CacheEngineKey],
    ) -> None:
        if not keys:
            return
        if self._backend_has_async_exists_mask(backend):
            for k in keys:
                backend.pin(k)
            return

        def _pin_sync() -> None:
            for k in keys:
                backend.contains(k, True)

        try:
            await asyncio.to_thread(_pin_sync)
        except Exception:
            logger.exception("async pin failed backend=%s", backend_name)

    async def _pin_owned_keys_async(
        self,
        keys: List[CacheEngineKey],
        owners: List[str],
        keys_per_chunk: int,
        pin: bool,
    ) -> None:
        if not pin:
            return
        grouped = self._keys_by_owner(keys, owners, keys_per_chunk)
        tasks = []
        for owner, owner_keys in grouped.items():
            if owner == "PDBackend":
                continue
            backend = self.storage_backends.get(owner)
            if backend is None or not owner_keys:
                continue
            tasks.append(self._pin_backend_keys(owner, backend, owner_keys))
        if tasks:
            await asyncio.gather(*tasks)

    def _key_exists_unpinned(self, backend: StorageBackendInterface, key: CacheEngineKey) -> bool:
        return bool(backend.contains(key, False))

    def _chunk_exists_unpinned(
        self,
        backend: StorageBackendInterface,
        chunk_keys: List[CacheEngineKey],
    ) -> bool:
        return all(self._key_exists_unpinned(backend, k) for k in chunk_keys)

    def _union_prefix_owners(
        self,
        keys: List[CacheEngineKey],
        keys_per_chunk: int,
        search_range: Optional[List[str]] = None,
        token_ids=None,
    ) -> List[str]:
        """Per-chunk any-tier prefix. Stop at the first chunk missing on all tiers."""
        if keys_per_chunk < 1:
            raise ValueError(f"keys_per_chunk must be >= 1, got {keys_per_chunk}")
        n_chunks = len(keys) // keys_per_chunk
        backends = list(self.get_active_storage_backends(search_range=search_range))
        owners: List[str] = []
        cpu_name = self._local_cpu_backend_name()
        remote_prefix: dict[str, int] = {}
        for backend_name, backend in backends:
            if backend_name == cpu_name:
                continue
            prefix_fn = getattr(backend, "prefix_contains_tokens", None)
            if callable(prefix_fn) and token_ids is not None:
                try:
                    remote_prefix[backend_name] = int(
                        prefix_fn(token_ids, n_chunks) or 0
                    )
                except TypeError:
                    remote_prefix[backend_name] = int(prefix_fn(token_ids) or 0)
                continue
            if callable(prefix_fn) and token_ids is None:
                remote_prefix[backend_name] = 0
                continue
            batched = getattr(backend, "batched_contains", None)
            if not callable(batched):
                continue
            try:
                remote_prefix[backend_name] = int(batched(keys) or 0)
            except TypeError:
                remote_prefix[backend_name] = int(batched(keys, False) or 0)
        for ci in range(n_chunks):
            chunk_keys = keys[ci * keys_per_chunk : (ci + 1) * keys_per_chunk]
            cpu_hit = False
            other: Optional[str] = None
            for backend_name, backend in backends:
                if backend_name == cpu_name:
                    if self._chunk_exists_unpinned(backend, chunk_keys):
                        cpu_hit = True
                    continue
                if backend_name in remote_prefix:
                    if ci < remote_prefix[backend_name] and other is None:
                        other = backend_name
                    continue
                if not self._chunk_exists_unpinned(backend, chunk_keys):
                    continue
                if other is None:
                    other = backend_name
            if cpu_hit:
                owners.append(cpu_name)
            elif other is not None:
                owners.append(other)
            else:
                break
        return owners

    def _pin_owned_keys(
        self,
        keys: List[CacheEngineKey],
        owners: List[str],
        keys_per_chunk: int,
        pin: bool,
    ) -> None:
        if not pin:
            return
        for ci, owner in enumerate(owners):
            if owner == "PDBackend":
                continue
            backend = self.storage_backends.get(owner)
            if backend is None:
                continue
            chunk_keys = keys[ci * keys_per_chunk : (ci + 1) * keys_per_chunk]
            for k in chunk_keys:
                backend.contains(k, True)

    def _keys_by_owner(
        self,
        keys: List[CacheEngineKey],
        owners: List[str],
        keys_per_chunk: int,
    ) -> Dict[str, List[CacheEngineKey]]:
        grouped: Dict[str, List[CacheEngineKey]] = {}
        for ci, owner in enumerate(owners):
            grouped.setdefault(owner, []).extend(
                keys[ci * keys_per_chunk : (ci + 1) * keys_per_chunk]
            )
        return grouped

    def _send_lookup_ack(self, lookup_id: str, num_hit_tokens: int) -> None:
        if lookup_id in self._lookup_acked:
            return
        self._lookup_acked.add(lookup_id)
        if self.async_lookup_server is not None:
            logger.info(
                "Responding to scheduler for lookup id %s with contains hits %s",
                lookup_id,
                num_hit_tokens,
            )
            self.async_lookup_server.send_response_to_scheduler(
                lookup_id, num_hit_tokens
            )

    async def async_lookup_and_prefetch(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        cum_chunk_lengths: list[int],
        search_range: Optional[list[str]] = None,
        pin: bool = False,
        keys_per_chunk: int = 1,
        token_ids=None,
    ) -> None:
        """
        Perform asynchronous lookup and prefetching across all storage backends.

        :param str lookup_id: The unique id (e.g., request id) for the request.
        :param list[CacheEngineKey] keys: The keys to lookup and prefetch.
        :param list[int] cum_chunk_lengths: The cumulative token lengths of the chunks.
            This is a list where cum_chunk_lengths[i] represents the total number of
            tokens from chunk 0 to chunk i-1 (inclusive).
            Example: If chunk_size=256 and we have 3 chunks:
                - chunk 0: 256 tokens (tokens 0-255)
                - chunk 1: 256 tokens (tokens 256-511)
                - chunk 2: 128 tokens (tokens 512-639)
            Then cum_chunk_lengths = [0, 256, 512, 640]
            Note: len(cum_chunk_lengths) = (len(keys) // keys_per_chunk) + 1
        :param Optional[list[str]] search_range: The range of storage backends
        to search in. Should be a subset of ["LocalCPUBackend",
        "LocalDiskBackend"] for now. If None, search in all backends.
        :param bool pin: Whether to pin the keys.
        :param int keys_per_chunk: Number of storage keys per logical chunk.
            For non-layerwise mode this is 1 (one key per chunk). For
            layerwise mode the caller passes num_layers, since each chunk
            is stored as num_layers per-layer keys (LayerCacheEngineKey).
            All chunk-level accounting below is derived from
            num_hit_keys // keys_per_chunk so that hit counts and
            cum_chunk_lengths indexing stay in chunk units even though the
            backend operates on per-layer keys.

        Probe each logical chunk on every active backend (exists mask, no pin).
        Union any-tier hits into a contiguous prefix, stop at the first
        all-miss chunk. LocalCPU wins when both tiers have the chunk. GET
        uses the sparse owner key set so CPU=suffix + Remote=prefix still
        reconstitutes the full prefix.

        ``kv_io_mode=gds`` skips remote host GET at lookup; the worker
        page-direct path reads RAW_PAGES with ``slot_mapping``.
        """

        if keys_per_chunk < 1:
            raise ValueError(f"keys_per_chunk must be >= 1, got {keys_per_chunk}")
        self._lookup_acked.discard(lookup_id)
        if len(keys) % keys_per_chunk != 0:
            raise ValueError(
                f"len(keys)={len(keys)} is not a multiple of "
                f"keys_per_chunk={keys_per_chunk}"
            )
        orig_keys = keys
        cum_chunk_lengths_total = cum_chunk_lengths[:]
        owners = await self._union_prefix_owners_async(
            orig_keys, keys_per_chunk, search_range, token_ids=token_ids
        )
        num_total_hit_chunks = len(owners)
        if pin:
            await self._pin_owned_keys_async(
                orig_keys, owners, keys_per_chunk, pin=True
            )

        loading_tasks = []
        tier_expected_chunks = []
        loading_task_keys: list[list[CacheEngineKey]] = []
        grouped = self._keys_by_owner(orig_keys, owners, keys_per_chunk)
        backend_by_name = {
            name: backend
            for name, backend in self.get_active_storage_backends(
                search_range=search_range
            )
        }
        for backend_name, backend_keys in grouped.items():
            backend = backend_by_name.get(backend_name)
            if backend is None or not backend_keys:
                continue
            n_chunks = len(backend_keys) // keys_per_chunk
            if n_chunks == 0:
                continue
            if (
                str(getattr(self.config, "kv_io_mode", "dram")) == "gds"
                and backend_name != self._local_cpu_backend_name()
            ):
                logger.info(
                    "GDS skip lookup-time host GET lookup_id=%s backend=%s "
                    "chunks=%s (worker page-direct)",
                    lookup_id,
                    backend_name,
                    n_chunks,
                )
                continue
            tier_expected_chunks.append(n_chunks)
            loading_task_keys.append(list(backend_keys))

            assert self.async_serializer is not None, (
                "Async serializer must be initialized via post_init before using "
                "async_lookup_and_prefetch."
            )
            work = PreloadWork(
                lookup_id=lookup_id,
                generation=self.preload_coordinator.next_generation(lookup_id),
                keys=list(backend_keys),
                n_keys=len(backend_keys),
                backend_name=backend_name,
            )
            serializer = self.async_serializer

            async def _gated_get(
                _backend=backend,
                _lookup_id=lookup_id,
                _work=work,
                _serializer=serializer,
                _n_chunks=n_chunks,
            ):
                async def _factory(w: PreloadWork):
                    n = w.granted_keys if w.granted_keys > 0 else len(_work.keys)
                    use_keys = _work.keys[:n]
                    return await _serializer.run(
                        _backend.batched_get_non_blocking(
                            _lookup_id,
                            use_keys,
                            {
                                "cum_chunk_lengths": cum_chunk_lengths_total[
                                    : _n_chunks + 1
                                ],
                                "preload_generation": w.generation,
                            },
                        ),
                        n,
                    )

                return await self.preload_coordinator.run_when_reserved(
                    _work, _factory
                )

            loading_task = asyncio.create_task(_gated_get())
            loading_task.add_done_callback(
                functools.partial(
                    self.prefetch_single_done_callback,
                    keys=orig_keys,
                    backend_name=backend_name,
                )
            )
            loading_tasks.append(loading_task)

        if num_total_hit_chunks == 0:
            self._send_lookup_ack(lookup_id, 0)
            return

        hit_tokens = cum_chunk_lengths_total[num_total_hit_chunks]
        self._send_lookup_ack(lookup_id, hit_tokens)
        if not loading_tasks:
            return

        # gather_with_keys() here make a pair of (key, memory_obj) for each chunk
        # in each tier. The all_done result's layout is like following and
        # will be processed in _async_process_tokens_internal()
        # Tier 0:
        #  Tuple(loading_task_keys[0][0] : MemoryObj0)
        #  Tuple(loading_task_keys[0][1] : MemoryObj1)
        # Tier 1:
        #  Tuple(loading_task_keys[1][0] : MemoryObj2)
        #  Tuple(loading_task_keys[1][1] : MemoryObj3)
        async def gather_with_keys() -> list[list[tuple[CacheEngineKey, MemoryObj]]]:
            return await gather_loading_results_with_keys(
                loading_tasks, loading_task_keys
            )

        all_done = asyncio.create_task(gather_with_keys())
        # Register the event before adding the callback to avoid race conditions
        self.event_manager.add_event(
            EventType.LOADING,
            lookup_id,
            all_done,
        )

        all_done.add_done_callback(
            lambda future: self.prefetch_all_done_callback(
                future,
                lookup_id,
                cum_chunk_lengths_total,
                tier_expected_chunks,
                keys_per_chunk=keys_per_chunk,
            )
        )

    def _on_lookup_prefetch_future_done(
        self, lookup_id: str, future: Future
    ) -> None:
        try:
            future.result()
        except Exception:
            logger.exception(
                "async_lookup_and_prefetch failed lookup_id=%s", lookup_id
            )
            try:
                self.event_manager.update_event_status(
                    EventType.LOADING, lookup_id, status=EventStatus.DONE
                )
            except Exception:
                pass
            self._send_lookup_ack(lookup_id, 0)

    def set_hot_cache(self, enabled: bool) -> None:
        """
        Dynamically enable or disable the hot cache on LocalCPUBackend.

        When disabled, the existing hot cache entries will be cleared
        and no new data will be written to the hot cache.

        Args:
            enabled: True to enable hot cache, False to disable
        """
        backend = self.local_cpu_backend
        if not isinstance(backend, LocalCPUBackend):
            logger.warning("Cannot set hot_cache: LocalCPUBackend not available")
            return

        if not enabled:
            backend.clear()
        backend.use_hot = enabled
        logger.info("LocalCPUBackend hot_cache set to %s", enabled)

    def is_hot_cache_enabled(self) -> bool:
        """
        Get the current hot cache status of LocalCPUBackend.

        Returns:
            bool: True if hot cache is enabled, False otherwise
        """
        backend = self.local_cpu_backend
        if not isinstance(backend, LocalCPUBackend):
            return False
        return backend.use_hot

    def set_freeze(self, enabled: bool) -> None:
        """
        Set freeze mode.

        When enabled, only local_cpu backend will be used for retrieval.
        """
        with self._freeze_lock:
            self._freeze = enabled
        logger.info("StorageManager freeze mode set to %s", enabled)

    def is_frozen(self) -> bool:
        """
        Get freeze mode status.

        Returns:
            bool: True if freeze mode is enabled, False otherwise
        """
        with self._freeze_lock:
            return self._freeze

    def set_backend_bypass(self, backend_name: str, bypassed: bool) -> None:
        """
        Set bypass mode for a specific backend.

        When a backend is bypassed:
        - It will be skipped during contains/put/get operations
        - This is typically used when a health check fails with LOCAL_CPU fallback

        Args:
            backend_name: The name of the backend to bypass (e.g., "RemoteBackend")
            bypassed: True to bypass, False to restore normal operation
        """
        with self._bypass_lock:
            if bypassed:
                self._bypassed_backends.add(backend_name)
                logger.info("StorageManager: Backend %s is now bypassed", backend_name)
            else:
                self._bypassed_backends.discard(backend_name)
                logger.info(
                    "StorageManager: Backend %s bypass removed, "
                    "restored to normal operation",
                    backend_name,
                )

    def is_backend_bypassed(self, backend_name: str) -> bool:
        """
        Check if a backend is currently bypassed.

        Args:
            backend_name: The name of the backend to check

        Returns:
            bool: True if the backend is bypassed, False otherwise
        """
        with self._bypass_lock:
            return backend_name in self._bypassed_backends

    def get_bypassed_backends(self) -> List[str]:
        """
        Get the list of currently bypassed backend names.

        Returns:
            List[str]: List of bypassed backend names
        """
        with self._bypass_lock:
            return list(self._bypassed_backends)

    def get_all_backend_names(self) -> List[str]:
        """
        Get the list of all registered backend names.

        Returns:
            List[str]: List of all backend names
        """
        return list(self.storage_backends.keys())

    def contains(
        self,
        key: CacheEngineKey,
        search_range: Optional[List[str]] = None,
        pin: bool = False,
    ) -> Optional[str]:
        """
        Check whether the key exists in the storage backend.

        :param CacheEngineKey key: The key to check.

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of ["LocalCPUBackend",
        "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param bool pin: Whether to pin the key.

        return: True if the key exists in the specified storage backends.
        """

        for backend_name, backend in self.get_active_storage_backends(
            search_range=search_range
        ):
            # NOTE(Jiayi): We do not pin for PDBackend
            pin_in_backend = pin if backend_name != "PDBackend" else False

            if backend.contains(key, pin_in_backend):
                return backend_name

        return None

    def batched_contains(
        self,
        keys: List[CacheEngineKey],
        search_range: Optional[List[str]] = None,
        pin: bool = False,
        token_ids=None,
    ) -> tuple[int, dict]:
        """
        Check whether the key exists in the storage backend.

        :param List[CacheEngineKey] keys: The keys to check.

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of ["LocalCPUBackend",
        "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param bool pin: Whether to pin the key.

        return: Return hit chunks and block mapping by any-tier prefix union.
        """
        owners = self._union_prefix_owners(
            keys, keys_per_chunk=1, search_range=search_range, token_ids=token_ids
        )
        if pin:
            self._pin_owned_keys(keys, owners, 1, pin=True)
        block_mapping = self._keys_by_owner(keys, owners, 1)
        return len(owners), block_mapping

    def get_block_mapping(
        self,
        chunk_infos: List[Tuple[CacheEngineKey, int, int]],
        token_ids=None,
    ) -> Dict[str, List[Tuple[CacheEngineKey, int, int]]]:
        """
        Get block mapping for the given chunk infos, works by prefix match.

        :param List[Tuple[CacheEngineKey, int, int]] chunk_infos:
        List of chunk infos, each tuple contains (key, begin, end)

        :        return: Dict[str, List[Tuple[CacheEngineKey, int, int]]]:
        Block mapping for the given chunk infos, each key is the backend name,
        each value is a list of chunk infos owned by that backend (sparse).
        """
        keys = [chunk_info[0] for chunk_info in chunk_infos]
        owners = self._union_prefix_owners(
            keys, keys_per_chunk=1, token_ids=token_ids
        )
        block_mapping: Dict[str, List[Tuple[CacheEngineKey, int, int]]] = {}
        for ci, owner in enumerate(owners):
            block_mapping.setdefault(owner, []).append(chunk_infos[ci])
        return block_mapping

    def touch_cache(self) -> None:
        for backend_name, backend in self.storage_backends.items():
            if backend_name == "LocalCPUBackend" or backend_name == "LocalDiskBackend":
                backend.touch_cache()

    def remove(
        self,
        key: CacheEngineKey,
        locations: Optional[List[str]] = None,
    ) -> int:
        """
        Remove the key and the corresponding cache in the specified
        locations.

        :param CacheEngineKey key: The key to remove.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `remove` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `remove` in all backends.

        return: Total number of removed caches in the specified
        storage backends.
        """

        num_removed = 0
        for backend_name, backend in self.storage_backends.items():
            # TODO(Jiayi): need to handle remove in non-cpu backends
            if locations is None or backend_name in locations:
                num_removed += backend.remove(key)

        return num_removed

    def batched_remove(
        self,
        keys: List[CacheEngineKey],
        locations: Optional[List[str]] = None,
    ) -> int:
        """
        Batched remove the keys and the corresponding cache in the specified
        locations.

        :param List[CacheEngineKey] keys: The keys to remove.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `remove` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `remove` in all backends.

        return: Total number of removed caches in the specified
        storage backends.
        """
        num_removed = 0
        for backend_name, backend in self.storage_backends.items():
            if locations is None or backend_name in locations:
                num_removed += backend.batched_remove(keys)

        return num_removed

    def batched_unpin(
        self,
        keys: List[CacheEngineKey],
        locations: Optional[List[str]] = None,
    ) -> None:
        """
        Unpin the keys in the specified locations.

        :param List[CacheEngineKey] keys: The keys to unpin.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `unpin` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `unpin` in all backends.
        """
        for backend_name, backend in self.storage_backends.items():
            if locations is None or backend_name in locations:
                for key in keys:
                    backend.unpin(key)

    def clear(
        self,
        locations: Optional[List[str]] = None,
    ) -> int:
        """
        Clear all caches in the specified locations.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `clear` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `clear` in all backends.

        return: Total number of cleared tokens in the specified
        storage backends.
        """

        num_cleared_tokens = 0
        for backend_name, backend in self.storage_backends.items():
            # TODO(Jiayi): need to handle remove in non-cpu backends
            if locations is None or backend_name in locations:
                if hasattr(backend, "clear"):
                    num_cleared_tokens += backend.clear()
                else:
                    logger.warning(
                        "Storage backend %s does not support "
                        "clear operation. Skipping.",
                        backend_name,
                    )

        return num_cleared_tokens

    def memcheck(self) -> bool:
        """
        Check the integrity of the underlying storage backend's
        memory allocators

        Returns:
            True if everything is good otherwise False
        """
        for backend in self.storage_backends.values():
            if not isinstance(backend, AllocatorBackendInterface):
                continue
            if not backend.get_memory_allocator().memcheck():
                return False
        return True

    def get_active_storage_backends(
        self,
        location: Optional[str] = None,
        search_range: Optional[List[str]] = None,
    ) -> Generator[Tuple[str, StorageBackendInterface], None, None]:
        """
        Get the active storage backends based on freeze mode, bypass mode, and filters.

        :param Optional[str] location: If specified, only yield backends
            matching this exact name.
        :param Optional[List[str]] search_range: If specified, only yield
            backends whose names are in this list.

        :return: Generator of (backend_name, backend) tuples.
        """
        for backend_name, backend in self.storage_backends.items():
            # In freeze mode, only use local_cpu backend
            with self._freeze_lock:
                if self._freeze and backend_name != "LocalCPUBackend":
                    continue
            # Skip bypassed backends
            with self._bypass_lock:
                if backend_name in self._bypassed_backends:
                    continue
            if location and backend_name != location:
                continue
            if search_range and backend_name not in search_range:
                continue
            yield backend_name, backend

    def get_non_allocator_backends(self) -> List[str]:
        """
        Get the names of the actual storage backends. Some backends,
        such as LocalCPUBackend and PDBackend, in some cases, only
        serve as a backend for allocation.
        """
        storage_names = []
        for backend_name, backend in self.storage_backends.items():
            if "LocalCPUBackend" == backend_name and not self.config.local_cpu:
                # if local_cpu is False, means LocalCPUBackend is only a allocator
                continue
            if "PDBackend" == backend_name and backend.pd_config.role == "sender":  # type: ignore
                # if pd_config.role is sender, means PDBackend is only a allocator
                continue
            storage_names.append(backend_name)
        return storage_names

    def list_backends(self) -> Dict[str, str]:
        """
        List all active storage backends.

        Returns:
            Dict mapping backend name to its class name.
        """
        with self.manager_lock:
            return {
                name: type(backend).__name__
                for name, backend in self.storage_backends.items()
            }

    def close_backend(self, backend_name: str) -> bool:
        """
        Close and remove a specific storage backend by name.

        The backend will be closed and removed from the internal
        dict so that no stale references remain.

        Args:
            backend_name: The name of the backend to close.

        Returns:
            True if the backend was found and closed, False
            otherwise.
        """
        with self.manager_lock:
            backend = self.storage_backends.get(backend_name)
            if backend is None:
                logger.warning(
                    "Backend %s not found, cannot close",
                    backend_name,
                )
                return False

            try:
                logger.info("Closing backend: %s", backend_name)
                backend.close()
            except Exception:
                logger.exception("Error closing backend %s", backend_name)

            del self.storage_backends[backend_name]

            # Update derived references
            self.non_allocator_backends = self.get_non_allocator_backends()
            if backend_name == "LocalCPUBackend":
                self.local_cpu_backend = None
            logger.info("Backend %s closed and removed", backend_name)
            return True

    def create_backends(self) -> Dict[str, str]:
        """
        Create new storage backends based on current config.

        Backends that are already present will be skipped
        **before** instantiation so that no unnecessary
        resources are allocated.  This allows callers to close
        a subset of backends, update config via ``/conf``,
        and then call this method to bring up only the missing
        backends.

        Returns:
            Dict mapping newly created backend name to its
            class name.
        """
        with self.manager_lock:
            existing_names = set(self.storage_backends)
            new_backends = CreateStorageBackends(
                self.config,
                self.metadata,
                self.loop,
                dst_device=(
                    torch_device_type if is_cuda_worker(self.metadata) else "cpu"
                ),
                lmcache_worker=self.lmcache_worker,
                skip_backends=existing_names,
                existing_backends=self.storage_backends,
            )

            created: Dict[str, str] = {}
            for name, backend in new_backends.items():
                self.storage_backends[name] = backend
                created[name] = type(backend).__name__
                logger.info(
                    "Created backend: %s (%s)",
                    name,
                    created[name],
                )

            # Refresh derived references
            self.non_allocator_backends = self.get_non_allocator_backends()
            cpu = self.storage_backends.get("LocalCPUBackend")
            if cpu is not None:
                self.local_cpu_backend = cpu

            return created

    def recreate_backend(self, backend_name: str) -> Dict[str, str]:
        """
        Close a backend and recreate it from current config.

        This is an atomic close-then-create operation that
        combines :meth:`close_backend` and :meth:`create_backends`
        into a single step.

        Args:
            backend_name: Name of the backend to recreate
                (e.g. ``RemoteBackend``).

        Returns:
            Dict mapping newly created backend name to its
            class name.

        Raises:
            KeyError: If *backend_name* does not exist.
        """
        with self.manager_lock:
            backend = self.storage_backends.get(backend_name)
            if backend is None:
                raise KeyError("Backend %s not found" % backend_name)

            # --- close ---
            try:
                logger.info("Closing backend: %s", backend_name)
                backend.close()
            except Exception:
                logger.exception("Error closing backend %s", backend_name)
            del self.storage_backends[backend_name]

            # --- create ---
            existing_names = set(self.storage_backends)
            new_backends = CreateStorageBackends(
                self.config,
                self.metadata,
                self.loop,
                dst_device=(
                    torch_device_type if is_cuda_worker(self.metadata) else "cpu"
                ),
                lmcache_worker=self.lmcache_worker,
                skip_backends=existing_names,
                existing_backends=self.storage_backends,
            )

            created: Dict[str, str] = {}
            for name, be in new_backends.items():
                self.storage_backends[name] = be
                created[name] = type(be).__name__
                logger.info(
                    "Recreated backend: %s (%s)",
                    name,
                    created[name],
                )

            # Refresh derived references
            self.non_allocator_backends = self.get_non_allocator_backends()
            cpu = self.storage_backends.get("LocalCPUBackend")
            if cpu is not None:
                self.local_cpu_backend = cpu
            elif backend_name == "LocalCPUBackend":
                self.local_cpu_backend = None

            return created

    def cancel_request(self, req_id: str) -> None:
        """
        Cancel an in-flight or pending request.

        Delegates to all storage backends. Backends that track per-request
        state will cancel the request; others will no-op.

        :param str req_id: The request identifier to cancel.
        :return: None
        """
        if getattr(self, "preload_coordinator", None) is not None:
            self.preload_coordinator.on_invalidated(req_id)
        for backend in self.storage_backends.values():
            backend.cancel_request(req_id)

    def close(self):
        logger.info("Closing StorageManager...")
        if getattr(self, "preload_coordinator", None) is not None:
            self.preload_coordinator.close()

        # Close all backends
        for name, backend in self.storage_backends.items():
            try:
                logger.info("Closing storage backend: %s", name)
                backend.close()
                logger.info("Storage backend %s closed successfully", name)
            except Exception as e:
                logger.error("Error closing backend %s: %s", name, e)

        # Stop event loop
        try:
            if self.loop.is_running():
                logger.info("Stopping event loop...")
                self.loop.call_soon_threadsafe(self.loop.stop)
                logger.info("Event loop stop signaled")
        except Exception as e:
            logger.error("Error stopping event loop: %s", e)

        # Wait for thread with timeout
        if self.thread.is_alive():
            logger.info("Waiting for storage manager thread to finish...")
            self.thread.join(timeout=10.0)

            if self.thread.is_alive():
                logger.warning(
                    "Storage manager thread did not terminate within 10s timeout. "
                    "Proceeding with shutdown anyway."
                )
            else:
                logger.info("Storage manager thread terminated successfully")
        else:
            logger.info("Storage manager thread already stopped")

        logger.info("Storage manager closed.")
