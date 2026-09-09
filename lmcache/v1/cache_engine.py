# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.health_monitor.base import HealthMonitor

# Standard
import asyncio
import concurrent.futures
import ctypes
import gc
import multiprocessing
import threading
import time

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.observability import LMCacheStatsLogger, LMCStatsMonitor
from lmcache.usage_telemetry import InitializeUsageContext
from lmcache.utils import (
    CacheEngineKey,
    CacheStoreEvent,
    _lmcache_nvtx_annotate,
    compress_slot_mapping,
    convert_tokens_to_list,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gds_store_policy import (
    GDS_STORE_OVERLOAD_IMMEDIATE_DROP,
)
from lmcache.v1.kvb_v2 import (
    PAYLOAD_ENCODING_CANONICAL_KV2LTD,
    SURDATA_SDK_KV_LAYOUT_SPLIT_REGION,
    compute_profile,
    physical_page_format_id,
)
from lmcache.v1.page_direct_xfer import build_raw_page_record, profile_as_sdk_dict
from lmcache.v1.event_manager import (
    EventManager,
    EventStatus,
    EventType,
    release_prefetch_result,
)
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface, gds_dma_len
from lmcache.v1.gpu_connector.utils import assert_layerwise_gpu_connector
from lmcache.v1.hidden_state_store import HiddenStateStore
from lmcache.v1.memory_allocators.cu_file_memory_allocator import CuFileMemoryAllocator
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
    PagedTensorMemoryAllocator,
)
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.platform import current_device_spec
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.system_detection import NUMADetector, NUMAMapping
from lmcache.v1.token_database import (
    ChunkedTokenDatabase,
    SegmentTokenDatabase,
    TokenDatabase,
)

logger = init_logger(__name__)

_GDS_UNWRAP_MAX_DEPTH = 8


def unwrap_gds_remote_connector(conn: Any, max_depth: int = _GDS_UNWRAP_MAX_DEPTH):
    """Return the innermost connector that exposes GDS put/get.

    ``CreateConnector`` wraps the real adapter in ``InstrumentedRemoteConnector``.
    Walk ``getWrappedConnector()`` (capped to avoid cycles) to the innermost
    object; succeed only if that object itself has ``submit_gds_object_put``
    or ``submit_gds_object_get``. Wrapper forwarding must not count as a hit.
    """
    current = conn
    seen: set[int] = set()
    for _ in range(max(1, int(max_depth))):
        if current is None:
            return None
        cid = id(current)
        if cid in seen:
            break
        seen.add(cid)
        getter = getattr(current, "getWrappedConnector", None)
        if not callable(getter):
            break
        try:
            nxt = getter()
        except Exception:
            break
        if nxt is None or nxt is current:
            break
        current = nxt
    if current is not None and (
        hasattr(current, "submit_gds_object_put")
        or hasattr(current, "submit_gds_object_get")
    ):
        return current
    return None


def _token_ids_u32(tokens) -> List[int]:
    if tokens is None:
        return []
    if hasattr(tokens, "detach"):
        tokens = tokens.detach().cpu().reshape(-1).tolist()
    return [int(t) & 0xFFFFFFFF for t in tokens]


def _require_cpu_slot_mapping(slot_mapping: Any) -> None:
    """Reject CUDA slot_mapping before page-direct view enumeration.

    ``get_page_direct_views`` reads slot ids on the host. A CUDA tensor here
    would force a synchronous D2H per chunk. Callers must pass CPU mapping.

    Args:
        slot_mapping: Token-to-slot ids.

    Raises:
        ValueError: If ``slot_mapping`` is missing or not on CPU.
    """
    if slot_mapping is None:
        raise ValueError("get_page_direct_views requires CPU slot_mapping")
    is_cuda = bool(getattr(slot_mapping, "is_cuda", False))
    dev_type = getattr(getattr(slot_mapping, "device", None), "type", None)
    if is_cuda or (dev_type is not None and dev_type != "cpu"):
        raise ValueError(
            "get_page_direct_views requires CPU slot_mapping, got device=%s"
            % (getattr(slot_mapping, "device", None),)
        )


# Type aliases for processed chunks
# (cache_key, memory_obj, start_index, end_index)
ProcessedChunk = Tuple[CacheEngineKey, MemoryObj, int, int]
# (list of processed chunks, total kv size)
ProcessTokensInternalResult = Tuple[List[ProcessedChunk], int]


class ReadyStoreEvent:
    """Stand-in CUDA event that is already complete (sync fallback)."""

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None


@dataclass
class PendingStore:
    """In-flight async D2H store.

    Host ``MemoryObj``s stay referenced here until ``complete_store`` or
    ``abandon_store``. After D2H, they must stay pinned until the native
    object-put future completes (SurData adapter unpins).
    """

    req_id: str
    keys: List[CacheEngineKey]
    memory_objs: List[MemoryObj]
    event: Optional[Any] = None
    error: Optional[BaseException] = None
    transfer_spec: Any = None
    tot_kv_size: int = 0
    tot_token_num: int = 0
    num_to_store_tokens: int = 0
    store_stats: Any = None
    submitted: bool = False
    gpu_block_count: int = 0
    gds_deferred_gpu_blocks: int = 0
    kwargs_keepalive: Dict[str, Any] = field(default_factory=dict)
    gds_leases: List[Any] = field(default_factory=list)
    put_future: Any = None
    gds_state: str = ""
    generation: int = 0
    block_ids: tuple = ()
    drop_put: bool = False
    quarantined: bool = False
    gpu_released: bool = False
    t_issue: float = 0.0
    t_event_ready: float = 0.0
    gds_fallback_d2h: bool = False
    gds_keys: List[CacheEngineKey] = field(default_factory=list)
    gds_objs: List[MemoryObj] = field(default_factory=list)
    requested_chunks: int = 0
    accepted_chunks: int = 0
    dropped_chunks: int = 0
    gds_put_error: Optional[BaseException] = None
    host_put_error: Optional[BaseException] = None
    d2h_admitted_bytes: int = 0
    d2h_admitted: int = 0
    token_ids: List[int] = field(default_factory=list)
    pd_leases: List[Tuple[int, int]] = field(default_factory=list)
    pd_keepalive: List[Any] = field(default_factory=list)
    direct_permit: int = 0
    store_reject_reason: str = ""
    t_gds_queue: float = 0.0
    t_native_first_submit: float = 0.0
    t_native_last_accept: float = 0.0
    t_native_done: float = 0.0
    t_kv_quarantine: float = 0.0
    t_cpu_prep_ms: float = 0.0
    t_cuda_enqueue_ms: float = 0.0
    t_native_submit_ms: float = 0.0


@dataclass
class PendingLoad:
    """In-flight async H2D retrieve.

    Host ``MemoryObj``s stay pinned until ``complete_load`` or ``abandon_load``.
    """

    req_id: str
    chunks: List[Any]
    event: Optional[Any] = None
    error: Optional[BaseException] = None
    retrieve_stats: Any = None
    ret_mask: Optional[torch.Tensor] = None
    kwargs_keepalive: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    generation: int = 0
    block_ids: tuple[int, ...] = ()
    gather_future: Any = None
    h2d_issued: bool = False
    # Explicit transfer state. h2d_issued is kept for compatibility.
    xfer_state: str = "QUEUED_FOR_L1"
    gds_leases: List[Any] = field(default_factory=list)
    gather_retries: int = 0
    t_h2d_issue: float = 0.0
    gds_state: str = ""
    t_gds_submit: float = 0.0
    t_dma_done: float = 0.0
    gds_read_fut: Any = None
    pd_header_hosts: List[Any] = field(default_factory=list)
    pd_header_event: Any = None
    pd_page_views: List[Any] = field(default_factory=list)
    pd_spans: List[tuple[int, int]] = field(default_factory=list)
    pd_hit: Any = None
    pd_prefix: int = 0
    pd_adapter: Any = None


XFER_QUEUED_FOR_L1 = "QUEUED_FOR_L1"
XFER_GATHERING = "GATHERING"
XFER_L1_READY = "L1_READY"
XFER_H2D_IN_FLIGHT = "H2D_IN_FLIGHT"
XFER_GDS_IN_FLIGHT = "GDS_IN_FLIGHT"
XFER_GPU_SCATTER_IN_FLIGHT = "GPU_SCATTER_IN_FLIGHT"
XFER_COMPLETED = "COMPLETED"
XFER_CANCELLING = "CANCELLING"
XFER_RELEASED = "RELEASED"

GDS_WAIT_LEASE = "WAIT_LEASE"
GDS_GATHER_SUBMITTED = "GATHER_SUBMITTED"
GDS_PUT_SUBMITTED = "PUT_SUBMITTED"
GDS_PUT_DONE = "PUT_DONE"
GDS_PUT_FAILED = "PUT_FAILED"
GDS_LOOKUP_READY = "LOOKUP_READY"
GDS_SUBMITTED = "GDS_SUBMITTED"
GDS_DMA_DONE = "GDS_DMA_DONE"
GDS_HBM_ASSEMBLED = "HBM_ASSEMBLED"
GDS_COMPLETE = "COMPLETE"


class GdsPipelineMetrics:
    """Hot-path-light GDS retrieve counters (no histograms on the DMA path)."""

    def __init__(self) -> None:
        self.submit_to_dma_us: List[int] = []
        self.dma_to_hbm_us: List[int] = []
        self.queue = 0
        self.inflight = 0
        self.admission_wait = 0
        self.sdk_cuda_error = 0
        self.watchdog_timeout = 0
        self.load_spec_abort = 0
        self.store_direct_chunks = 0
        self.store_d2h_chunks = 0
        self.store_slot_starved = 0
        self.store_dropped_chunks = 0
        self.store_future_error = 0
        self.staging_lease_wait_count = 0
        self.staging_lease_wait_us = 0
        self.staging_lease_starved = 0
        self.d2h_admit_or_allocate_wait_count = 0
        self.d2h_admit_or_allocate_wait_us = 0
        self.d2h_budget_exhausted = 0
        self.d2h_alloc_fail = 0
        self.direct_permit_rejected = 0
        self.page_direct_unavailable = 0
        self.page_direct_emergency_d2h = 0
        self.gds_cache_register_fail = 0
        self.kvb_gpu_future_error = 0
        self.host_put_backlog = 0
        self.host_put_wait_us: List[int] = []
        self.store_cpu_prep_us: List[int] = []
        self.store_cuda_enqueue_us: List[int] = []
        self.store_native_submit_us: List[int] = []

    def _note_us_list(self, samples: List[int], us: int) -> None:
        if us < 0:
            return
        samples.append(int(us))
        if len(samples) > 4096:
            del samples[: len(samples) - 2048]

    def note_store_split(
        self, cpu_prep_us: int, cuda_enqueue_us: int, native_submit_us: int
    ) -> None:
        """Record one GDS store's EngineCore split (CPU / CUDA enqueue / native)."""
        self._note_us_list(self.store_cpu_prep_us, cpu_prep_us)
        self._note_us_list(self.store_cuda_enqueue_us, cuda_enqueue_us)
        self._note_us_list(self.store_native_submit_us, native_submit_us)

    def note_submit_dma(self, us: int) -> None:
        if us >= 0:
            self.submit_to_dma_us.append(int(us))
            if len(self.submit_to_dma_us) > 4096:
                self.submit_to_dma_us = self.submit_to_dma_us[-2048:]

    def note_dma_hbm(self, us: int) -> None:
        if us >= 0:
            self.dma_to_hbm_us.append(int(us))
            if len(self.dma_to_hbm_us) > 4096:
                self.dma_to_hbm_us = self.dma_to_hbm_us[-2048:]

    def snapshot(self) -> Dict[str, Any]:
        def _p99(samples: List[int]) -> int:
            if not samples:
                return 0
            ordered = sorted(samples)
            idx = min(len(ordered) - 1, int(round(0.99 * (len(ordered) - 1))))
            return int(ordered[idx])

        def _p50(samples: List[int]) -> int:
            if not samples:
                return 0
            ordered = sorted(samples)
            return int(ordered[len(ordered) // 2])

        return {
            "queue": self.queue,
            "inflight": self.inflight,
            "admission_wait": self.admission_wait,
            "sdk_cuda_error": self.sdk_cuda_error,
            "watchdog_timeout": self.watchdog_timeout,
            "load_spec_abort": self.load_spec_abort,
            "store_direct_chunks": self.store_direct_chunks,
            "store_d2h_chunks": self.store_d2h_chunks,
            "store_slot_starved": self.store_slot_starved,
            "store_dropped_chunks": self.store_dropped_chunks,
            "store_future_error": self.store_future_error,
            "staging_lease_wait_count": self.staging_lease_wait_count,
            "staging_lease_wait_us": self.staging_lease_wait_us,
            "staging_lease_starved": self.staging_lease_starved,
            "d2h_admit_or_allocate_wait_count": self.d2h_admit_or_allocate_wait_count,
            "d2h_admit_or_allocate_wait_us": self.d2h_admit_or_allocate_wait_us,
            "d2h_budget_exhausted": self.d2h_budget_exhausted,
            "d2h_alloc_fail": self.d2h_alloc_fail,
            "direct_permit_rejected": self.direct_permit_rejected,
            "page_direct_unavailable": self.page_direct_unavailable,
            "page_direct_emergency_d2h": self.page_direct_emergency_d2h,
            "gds_cache_register_fail": self.gds_cache_register_fail,
            "kvb_gpu_future_error": self.kvb_gpu_future_error,
            "host_put_backlog": self.host_put_backlog,
            "host_put_wait_p99_us": _p99(self.host_put_wait_us),
            "host_put_wait_n": len(self.host_put_wait_us),
            "submit_to_dma_p99_us": _p99(self.submit_to_dma_us),
            "dma_to_hbm_p99_us": _p99(self.dma_to_hbm_us),
            "submit_to_dma_n": len(self.submit_to_dma_us),
            "dma_to_hbm_n": len(self.dma_to_hbm_us),
            "store_cpu_prep_p50_us": _p50(self.store_cpu_prep_us),
            "store_cuda_enqueue_p50_us": _p50(self.store_cuda_enqueue_us),
            "store_native_submit_p50_us": _p50(self.store_native_submit_us),
            "store_split_n": len(self.store_cpu_prep_us),
        }


class CacheEngineEndSignal:
    pass


class LMCacheEngine:
    """The main class for the cache engine.

    When storing the KV caches into the cache engine, it takes GPU KV
    caches from the serving engine and convert them into MemoryObjs that
    resides in the CPU. The MemoryObjs are then being stored into the
    StorageBackends in an asynchronous manner.

    When retrieving the KV caches from the cache engine, it fetches the
    MemoryObjs from the StorageBackends and convert them into GPU KV caches
    by GPUConnectors specialized for the serving engine.

    It also supports prefetching the KV caches from the StorageBackends.
    It relies on the StorageBackends to manage the requests of prefetching
    and real retrieval and avoid the conflicts.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        token_database: TokenDatabase,
        gpu_connector: Optional[GPUConnectorInterface],
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ):
        logger.info("Creating LMCacheEngine with config: %s", config)
        self.config = config
        self.metadata = metadata
        self.token_database = token_database
        self.gpu_connector = gpu_connector
        self.broadcast_fn = broadcast_fn
        self.broadcast_object_fn = broadcast_object_fn
        # save_only_first_rank only works when use mla
        self.save_only_first_rank = (
            self.config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )

        if self.save_only_first_rank and self.gpu_connector is not None:
            self.broadcast_stream = (
                self.gpu_connector.load_stream
                if hasattr(self.gpu_connector, "load_stream")
                else torch_dev.Stream()
            )

        # Transient GPU staging for MLA TP broadcast (at most the in-flight
        # chunk). Empty after retrieve returns; async keepalive holds the
        # last chunk until complete_load.
        self._leader_gpu_substitute_objs: List[MemoryObj] = []

        self.enable_controller = config.enable_controller

        # NOTE: Unix systems use fork by default
        multiprocessing.set_start_method("spawn", force=True)

        # avoid circular import
        # First Party
        from lmcache.v1.cache_controller import LMCacheWorker

        self.lmcache_worker: Optional[LMCacheWorker] = None
        lmcache_worker_ids = config.get_lmcache_worker_ids(
            metadata.use_mla, metadata.world_size
        )
        # lmcache_worker_ids is empty means start on all workers
        if (
            self.enable_controller
            and self.metadata.role != "scheduler"
            and (not lmcache_worker_ids or metadata.worker_id in lmcache_worker_ids)
        ):
            self.lmcache_worker = LMCacheWorker(config, metadata, self)
        else:
            self.lmcache_worker = None
            logger.info(
                "LMCacheWorker is not initialized (related configs: "
                "enable_controller: %s, role: %s, worker_id: %d, worker_ids: %s).",
                self.enable_controller,
                self.metadata.role,
                self.metadata.worker_id,
                lmcache_worker_ids,
            )

        self.async_loading = config.enable_async_loading
        self.event_manager = EventManager()
        self._gds_metrics = GdsPipelineMetrics()
        self._gds_d2h_lock = threading.Condition()
        self._gds_d2h_bytes = 0
        self._gds_d2h_inflight = 0
        self._gds_direct_lock = threading.Lock()
        self._gds_direct_inflight = 0
        self._pd_profile = None
        self._pd_header_pin_lock = threading.Lock()
        self._pd_header_pins: List[torch.Tensor] = []

        self.use_layerwise = config.use_layerwise

        # TODO: support save_only_first_rank when use layerwise
        # if use_layerwise is True, all ranks will initialize the storage_manager
        # if save_only_first_rank is False, all ranks will initialize
        # the storage_manager
        # if save_only_first_rank is True, only the first rank and
        # lookup server workers will initialize the storage_manager
        self.storage_manager: Optional[StorageManager] = None

        # KV events
        self.kv_events_enabled = False
        self.kv_events_enabled = config.enable_kv_events
        if self.kv_events_enabled:
            self.kv_events: List[CacheStoreEvent] = []
            logger.info("KV events are enabled.")
        else:
            logger.info("KV events are disabled.")

        # HACK: remove this in the future
        # NOTE (Jiayi): This is currently used to support
        # dropping the kv cache from the buffer in PD backend
        # at decoder.
        self.remove_after_retrieve = config.enable_pd and config.pd_role == "receiver"

        # asymmetric store/retrieve location can be specified
        # this is typically used (but not limited) in PD system
        self.store_location = config.store_location
        self.retrieve_locations = config.retrieve_locations

        self.num_layers = metadata.kv_shape[0]
        self.fmt = None
        if self.use_layerwise:
            if metadata.use_mla:
                self.fmt = MemoryFormat.KV_MLA_FMT
            elif config.enable_blending:
                self.fmt = MemoryFormat.KV_2TD
            else:
                self.fmt = MemoryFormat.KV_T2D
        if metadata.use_mla:
            self.fmt = MemoryFormat.KV_MLA_FMT

        # NOTE(ApostaC): we haven't support lookup-cache yet
        self.lookup_cache: dict[CacheEngineKey, Any] = {}

        # lookup_id -> {location -> [pinned keys]}
        self.lookup_pins: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.abandoned_loads: List[PendingLoad] = []
        self.abandoned_stores: List[PendingStore] = []

        InitializeUsageContext(config, metadata)
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        # Initialize PinMonitor singleton with config
        PinMonitor.GetOrCreate(config, metadata)

        self.post_inited = False

        # Flag to control KVCache Check logging (can be toggled via API)
        self.kvcache_check_log_enabled = False

        gc.collect()
        if not config.py_enable_gc:
            gc.disable()

        # Health monitor reference (injected by LMCacheManager)
        self._health_monitor: Optional["HealthMonitor"] = None

        # Flag to indicate if initialization failed (irrecoverable error)
        self._init_failed = False

        # Hidden-state cache (logically separate from KV; lives on its own
        # pinned pool). Bound to storage_manager in post_init for coupled
        # eviction. None when disabled in config.
        self.hidden_state_store: Optional[HiddenStateStore] = None
        if config.enable_hidden_state_cache:
            self.hidden_state_store = HiddenStateStore(config, token_database)

    def set_health_monitor(self, health_monitor: "HealthMonitor") -> None:
        """
        Set the health monitor reference.

        This is called by LMCacheManager after creating the HealthMonitor
        to inject the reference into the engine.

        Args:
            health_monitor: The HealthMonitor instance from LMCacheManager
        """
        self._health_monitor = health_monitor

    def is_healthy(self) -> bool:
        """
        Check if the LMCache system is healthy.

        This method returns False if:
        - Initialization failed (irrecoverable error)
        - HealthMonitor reports unhealthy

        If no health monitor is set and initialization succeeded,
        it returns True (assume healthy).

        Returns:
            bool: True if healthy, False otherwise
        """
        if self._init_failed:
            return False
        if self._health_monitor is not None:
            return self._health_monitor.is_healthy()
        return True

    def _get_req_id(self, kwargs: dict) -> str:
        """Extracts request ID from kwargs for logging."""
        return kwargs.get("req_id", "unspecified")

    def mark_init_failed(self, reason: str = "") -> None:
        """
        Mark the engine as having failed initialization.

        This is called by LMCacheManager when an irrecoverable error occurs
        during initialization or post_init. Once marked, is_healthy() will
        always return False, causing the system to fall back to recomputation.

        Args:
            reason: Optional reason string for logging
        """
        self._init_failed = True
        if reason:
            logger.error("LMCacheEngine marked as init failed: %s", reason)
        else:
            logger.error("LMCacheEngine marked as init failed")

    def post_init(self, **kwargs) -> None:
        if not self.post_inited:
            logger.info("Post initializing LMCacheEngine")
            lookup_server_worker_ids = self.config.get_lookup_server_worker_ids(
                self.metadata.use_mla, self.metadata.world_size
            )
            if (
                self.lmcache_worker is not None
                or self.use_layerwise
                or not self.save_only_first_rank
                or self.metadata.is_first_rank()
                or len(lookup_server_worker_ids) == 0
                or self.metadata.worker_id in lookup_server_worker_ids
            ):
                logger.info(
                    "Initialize storage manager on rank %d, "
                    "use layerwise: %s,"
                    "save only first rank: %s",
                    self.metadata.worker_id,
                    self.use_layerwise,
                    self.save_only_first_rank,
                )
                async_lookup_server = kwargs.get("async_lookup_server", None)
                self.storage_manager = StorageManager(
                    self.config,
                    self.metadata,
                    event_manager=self.event_manager,
                    lmcache_worker=self.lmcache_worker,
                    async_lookup_server=async_lookup_server,
                )
                if self.hidden_state_store is not None:
                    self.hidden_state_store.bind_storage_manager(self.storage_manager)
                self._maybe_prewarm_gds_pool()
            self.post_inited = True

    def freeze(self, enabled: bool) -> None:
        """
        Set the freeze mode for the cache engine.

        When freeze mode is enabled:
        - All store operations will be skipped (no new data stored)
        - Only local_cpu backend will be used for retrieval
        - No admit/evict messages will be generated
        This protects the local_cpu hot cache from changes.

        Args:
            enabled (bool): Whether to enable freeze mode
        """
        if self.storage_manager is not None:
            self.storage_manager.set_freeze(enabled)

    def is_frozen(self) -> bool:
        """
        Get the current freeze mode status.

        Returns:
            bool: True if freeze mode is enabled, False otherwise
        """
        if self.storage_manager is not None:
            return self.storage_manager.is_frozen()
        return False

    def set_hot_cache(self, enabled: bool) -> None:
        """
        Dynamically enable or disable the LocalCPUBackend hot cache.

        When disabled, the existing hot cache entries will be cleared
        and no new data will be written to the hot cache.

        Args:
            enabled (bool): Whether to enable hot cache
        """
        if self.storage_manager is not None:
            self.storage_manager.set_hot_cache(enabled)

    def is_hot_cache_enabled(self) -> bool:
        """
        Get the current hot cache status of LocalCPUBackend.

        Returns:
            bool: True if hot cache is enabled, False otherwise
        """
        if self.storage_manager is not None:
            return self.storage_manager.is_hot_cache_enabled()
        return False

    def prepare_store(
        self,
        tokens: Optional[Union[torch.Tensor, list[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[PendingStore]:
        """Allocate host MemoryObjs and chunk keys without D2H or put.

        Returns None when the store is skipped (unhealthy, frozen, passive,
        or no chunks allocated).
        """
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping store operation")
            return None

        assert self.gpu_connector is not None, (
            "gpu_connector is required for store operation"
        )

        if self._is_passive():
            logger.debug("rank=%d ignore store", self.metadata.worker_id)
            return None

        assert self.storage_manager is not None

        req_id = self._get_req_id(kwargs)
        num_to_store_tokens = 0

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        elif tokens is not None:
            num_to_store_tokens = len(tokens)
        elif hashes is not None:
            assert offsets is not None, (
                "Offsets should be set when hashes are provided during store"
            )
            num_to_store_tokens = sum(offsets)
            kwargs["slot_mapping"] = torch.tensor(
                kwargs["slot_mapping"], dtype=torch.long, device=torch_device_type
            )

        assert tokens is not None or hashes is not None, (
            "Either 'tokens' or 'hashes' must be provided."
        )

        self._log_kvcache_for_check(
            operation="Store",
            kwargs=kwargs,
            token_count=num_to_store_tokens,
            require_req_id=False,
        )

        if self.is_frozen():
            logger.debug(
                "Freeze mode enabled, skipping store operation for %d tokens",
                num_to_store_tokens,
            )
            return None

        store_stats = self.stats_monitor.on_store_request(num_to_store_tokens)

        starts: List[int] = []
        ends: List[int] = []
        keys: List[CacheEngineKey] = []
        memory_objs: List[MemoryObj] = []
        allocate_l1_explicit = "allocate_l1" in kwargs
        allocate_l1 = bool(kwargs.pop("allocate_l1", True))
        if (
            str(getattr(self.config, "kv_io_mode", "dram")) == "gds"
            and not allocate_l1_explicit
        ):
            allocate_l1 = False

        tot_kv_size = 0
        tot_token_num = 0
        requested_chunks = 0
        accepted_chunks = 0
        dropped_chunks = 0
        l1_exhausted = False

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        with store_stats.profile_process_tokens():
            prev_key = 0
            for start, end, key in self.token_database.process_tokens(
                tokens,
                hashes,
                offsets,
                mask,
                request_configs=request_configs,
            ):
                assert isinstance(key, CacheEngineKey)
                num_tokens = end - start
                kv_shapes = self.metadata.get_shapes(num_tokens)
                kv_dtypes = self.metadata.get_dtypes()

                requested_chunks += 1
                if allocate_l1:
                    if l1_exhausted:
                        dropped_chunks += 1
                        logger.warning(
                            "Local cpu memory under pressure so skipping "
                            "remaining KV chunk start=%s end=%s "
                            "(wanted=%s allocated=%s dropped=%s)",
                            start,
                            end,
                            requested_chunks,
                            accepted_chunks,
                            dropped_chunks,
                        )
                        continue
                    memory_obj = self.storage_manager.allocate(
                        kv_shapes,
                        kv_dtypes,
                        busy_loop=self.config.get_extra_config_value(
                            "force_store_wait", False
                        ),
                        fmt=self.fmt,
                    )
                    if memory_obj is None:
                        l1_exhausted = True
                        dropped_chunks += 1
                        logger.warning(
                            "Local cpu memory under pressure so"
                            " choosing to store only "
                            " %d total chunks of KV cache "
                            "(wanted=%s allocated=%s dropped=%s).",
                            len(memory_objs),
                            requested_chunks,
                            accepted_chunks,
                            dropped_chunks,
                        )
                        continue
                    tot_kv_size += memory_obj.get_size()
                    memory_objs.append(memory_obj)
                    accepted_chunks += 1
                else:
                    tot_kv_size += 0
                    accepted_chunks += 1

                starts.append(start)
                ends.append(end)
                keys.append(key)
                tot_token_num += num_tokens

                if self.kv_events_enabled:
                    stored_event = CacheStoreEvent(
                        block_hashes=[key.chunk_hash],
                        parent_block_hash=None if start == 0 else prev_key,
                        token_ids=[],
                        block_size=num_tokens,
                        lora_id=None,
                        medium="cpu",
                        lora_name=None,
                    )
                    if tokens is not None:
                        stored_event.token_ids = convert_tokens_to_list(
                            tokens,
                            start,
                            end,
                        )
                        if isinstance(tokens, torch.Tensor):
                            stored_event.medium = tokens.device
                    elif hashes is not None:
                        stored_event.token_ids = hashes[start : end + 1]
                    logger.debug(
                        (
                            "Added kv cache event '%s' to kv cache events queue"
                            % stored_event
                        )
                    )
                    self.kv_events.append(stored_event)
                    prev_key = key.chunk_hash

        if allocate_l1 and not memory_objs:
            if dropped_chunks:
                metrics = getattr(self, "_gds_metrics", None)
                if metrics is not None:
                    metrics.store_dropped_chunks += dropped_chunks
            return None
        if not allocate_l1 and not keys:
            return None

        pending = PendingStore(
            req_id=req_id,
            keys=keys,
            memory_objs=memory_objs,
            transfer_spec=kwargs.get("transfer_spec", None),
            tot_kv_size=tot_kv_size,
            tot_token_num=tot_token_num,
            num_to_store_tokens=num_to_store_tokens,
            store_stats=store_stats,
            kwargs_keepalive={"starts": starts, "ends": ends},
            requested_chunks=requested_chunks,
            accepted_chunks=accepted_chunks,
            dropped_chunks=dropped_chunks,
            token_ids=_token_ids_u32(tokens),
        )
        return pending

    def issue_store_async(
        self,
        tokens: Optional[Union[torch.Tensor, list[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[PendingStore]:
        """Issue async GPU gather for DRAM host D2H or GDS GPU bounce.

        Adapter must not branch on kv_io_mode; GDS lease/cuFile stay here.
        """
        if str(getattr(self.config, "kv_io_mode", "dram")) == "gds":
            return self.issue_store_gds(
                tokens=tokens,
                hashes=hashes,
                offsets=offsets,
                mask=mask,
                **kwargs,
            )
        return self.issue_store_d2h(
            tokens=tokens,
            hashes=hashes,
            offsets=offsets,
            mask=mask,
            **kwargs,
        )

    def issue_store_d2h(
        self,
        tokens: Optional[Union[torch.Tensor, list[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[PendingStore]:
        """Issue D2H on the store stream and return a PendingStore.

        Does not wait for the CUDA event and does not submit object put.
        """
        pending = self.prepare_store(
            tokens=tokens,
            hashes=hashes,
            offsets=offsets,
            mask=mask,
            **kwargs,
        )
        if pending is None:
            return None

        starts = pending.kwargs_keepalive["starts"]
        ends = pending.kwargs_keepalive["ends"]
        store_stats = pending.store_stats
        assert self.gpu_connector is not None

        with store_stats.profile_from_gpu():
            try:
                # Only V2 overrides batched_from_gpu_async with a no-sync
                # CUDA event. V3 / layerwise / other connectors inherit the
                # interface fallback, which runs batched_from_gpu then a
                # ready event (explicit sync fallback).
                batched_async = getattr(
                    self.gpu_connector, "batched_from_gpu_async", None
                )
                if callable(batched_async):
                    pending.event = batched_async(
                        pending.memory_objs, starts, ends, **kwargs
                    )
                else:
                    self.gpu_connector.batched_from_gpu(
                        pending.memory_objs, starts, ends, **kwargs
                    )
                    pending.event = ReadyStoreEvent()
            except Exception as exc:
                pending.error = exc
                logger.exception(
                    "[req_id=%s] issue_store_d2h failed; dropping store",
                    pending.req_id,
                )
                return pending

        if pending.event is None:
            pending.event = ReadyStoreEvent()
        pending.t_issue = time.perf_counter()
        pending.generation = int(kwargs.get("generation", 0) or 0)
        pending.block_ids = tuple(kwargs.get("block_ids", ()) or ())
        return pending

    def complete_store(self, pending: PendingStore) -> None:
        """Submit batched_put after the D2H event is ready.

        Pins host MemoryObjs so L1 cannot recycle them until the SurData
        adapter unpins after the native object-put future completes.
        """
        if pending.submitted:
            return
        pending.submitted = True

        if pending.error is not None or getattr(pending, "drop_put", False):
            self.abandon_store(pending, wait_event=True, already_submitted=True)
            return

        kv_io_mode = str(getattr(self.config, "kv_io_mode", "dram") or "dram")
        if kv_io_mode == "gds" and (
            pending.put_future is not None
            or pending.gds_leases
            or getattr(pending, "pd_leases", None)
        ):
            harvest = self._harvest_gds_store(pending)
            if harvest == "pending":
                pending.submitted = False
                return
            if harvest == "failed" and not pending.memory_objs:
                pending.gds_state = GDS_PUT_FAILED
                logger.error(
                    "[req_id=%s] GDS put failed with no host subset: %s",
                    pending.req_id,
                    pending.gds_put_error,
                )
                self.abandon_store(pending, wait_event=True, already_submitted=True)
                return
            if not pending.memory_objs:
                pending.gds_state = GDS_PUT_DONE
                pending.submitted = True
                if pending.store_stats is not None:
                    self.stats_monitor.on_store_finished(
                        pending.store_stats,
                        pending.tot_token_num,
                    )
                return
            pending.gds_state = GDS_PUT_DONE

        if not pending.memory_objs:
            return

        assert self.storage_manager is not None
        store_stats = pending.store_stats
        for memory_obj in pending.memory_objs:
            memory_obj.pin()

        try:
            with store_stats.profile_put():
                adapter = None
                backends = getattr(self.storage_manager, "storage_backends", None) or {}
                committed = None
                if pending.token_ids:
                    for backend in backends.values():
                        kvb = getattr(backend, "kvb_store_memory_objs", None)
                        if callable(kvb):
                            adapter = backend
                            committed = int(
                                kvb(pending.token_ids, pending.memory_objs) or 0
                            )
                            break
                if adapter is None:
                    self.storage_manager.batched_put(
                        pending.keys,
                        pending.memory_objs,
                        transfer_spec=pending.transfer_spec,
                        location=self.store_location,
                    )
                else:
                    if committed < 1:
                        raise RuntimeError("kvb_store_memory_objs committed 0")
                    if committed < len(pending.memory_objs):
                        pending.keys = pending.keys[:committed]
                        pending.memory_objs = pending.memory_objs[:committed]
                        ends = list(
                            pending.kwargs_keepalive.get("host_ends")
                            or pending.kwargs_keepalive.get("ends")
                            or []
                        )
                        if ends and committed <= len(ends):
                            pending.tot_token_num = int(ends[committed - 1])
                        pending.accepted_chunks = committed
        except Exception as exc:
            pending.host_put_error = exc
            metrics = getattr(self, "_gds_metrics", None)
            if metrics is not None:
                metrics.store_future_error += 1
            self._release_store_d2h_admit(pending)
            # SurData adapter _batched_put_host releases complete_store pins in
            # its finally once the put coroutine finishes (including failures).
            pending.submitted = False
            raise

        self._release_store_d2h_admit(pending)
        self.stats_monitor.on_store_finished(
            store_stats,
            pending.tot_token_num,
        )
        tot_time = store_stats.time_to_store()
        tot_kv_size = pending.tot_kv_size
        host_put_keys = len(pending.memory_objs)
        logger.info(
            "[req_id=%s] Stored %d out of total %d tokens "
            "(host_put_keys=%s). "
            "size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s; "
            "offload_time: %.4f ms, put_time: %.4f ms",
            pending.req_id,
            pending.tot_token_num,
            pending.num_to_store_tokens,
            host_put_keys,
            tot_kv_size / 1024**3,
            tot_time * 1000,
            tot_kv_size / tot_time / 1024**3 if tot_time > 0 else 0,
            (store_stats.process_tokens_time + store_stats.from_gpu_time) * 1000,
            store_stats.put_time * 1000,
        )

    def _event_is_ready(self, event) -> bool:
        if event is None:
            return True
        query = getattr(event, "query", None)
        if callable(query):
            return bool(query())
        return False

    def abandon_store(
        self,
        pending: PendingStore,
        wait_event: bool = True,
        already_submitted: bool = False,
    ) -> None:
        """Drop an in-flight store without blocking the compute path."""
        if pending.submitted and not already_submitted:
            return
        if not self._event_is_ready(pending.event):
            self.abandoned_stores.append(pending)
            return
        fut = getattr(pending, "put_future", None)
        if fut is not None:
            done = getattr(fut, "done", None)
            if callable(done) and not done():
                self.abandoned_stores.append(pending)
                return
        pending.submitted = True
        self._release_gds_leases(pending)
        self._release_store_d2h_admit(pending)
        for memory_obj in pending.memory_objs:
            try:
                memory_obj.ref_count_down()
            except Exception:
                logger.exception(
                    "[req_id=%s] abandon_store ref_count_down failed",
                    pending.req_id,
                )
        pending.memory_objs = []
        pending.keys = []
        pending.gds_keys = []
        pending.gds_objs = []
        pending.kwargs_keepalive = {}

    def poll_abandoned_stores(self) -> List[PendingStore]:
        still: List[PendingStore] = []
        done: List[PendingStore] = []
        for pending in self.abandoned_stores:
            if self._event_is_ready(pending.event):
                pending.submitted = False
                self.abandon_store(pending, wait_event=False, already_submitted=False)
                done.append(pending)
            else:
                still.append(pending)
        self.abandoned_stores = still
        return done

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store(
        self,
        tokens: Optional[Union[torch.Tensor, list[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Store the tokens/hashes and mask into the cache engine.

        Synchronous wrapper: issue D2H, wait for the event, then complete put.
        Non-vLLM callers keep the original blocking semantics.

        :param Optional[torch.Tensor] tokens: The tokens of the corresponding KV caches.

        :param Optional[List[int]] hashes: The hashes of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        pending = (
            self.issue_store_gds(
                tokens=tokens,
                hashes=hashes,
                offsets=offsets,
                mask=mask,
                **kwargs,
            )
            if str(getattr(self.config, "kv_io_mode", "dram")) == "gds"
            else self.issue_store_d2h(
                tokens=tokens,
                hashes=hashes,
                offsets=offsets,
                mask=mask,
                **kwargs,
            )
        )
        if pending is None:
            return
        if pending.error is not None:
            self.abandon_store(pending, wait_event=True)
            raise pending.error
        if pending.event is not None:
            pending.event.synchronize()
        fut = getattr(pending, "put_future", None)
        if fut is not None:
            waiter = getattr(fut, "result", None)
            if callable(waiter):
                waiter()
        self.complete_store(pending)

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[None, None, None]:
        """
        Store the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields None. In the first iteration, the
            generator allocates the memory objects for all layers and moves
            the KV cache of the first layer from GPU to CPU. In the next
            iterations, it moves the KV cache of layer i from GPU to the memory
            objects (on CPU) and puts the memory objects of layer i-1 to the
            storage backends. In the last iteration, it puts the memory objects
            of the last layer to the storage backends.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping store_layer operation")
            return

        assert self.storage_manager is not None
        assert self.gpu_connector is not None, (
            "gpu_connector is required for store_layer operation"
        )

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        else:
            num_to_store_tokens = len(tokens)

        # KVCache Check logging
        self._log_kvcache_for_check(
            operation="Layerwise store",
            kwargs=kwargs,
            token_count=num_to_store_tokens,
            require_req_id=True,
        )

        monitor_req_id = self.stats_monitor.on_store_request(num_to_store_tokens)

        # Check if freeze mode is enabled
        if self.is_frozen():
            logger.debug(
                "Freeze mode enabled, skipping store_layer for %d tokens",
                num_to_store_tokens,
            )
            # Still need to yield to avoid StopIteration
            for layer_id in range(self.num_layers):
                yield
            return

        starts = []
        ends = []
        keys = []
        memory_objs = []
        tot_token_num = 0
        kv_dtype = self.metadata.kv_dtype
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        prev_key = 0
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, mask=mask, request_configs=request_configs
        ):
            assert isinstance(key, CacheEngineKey)

            keys_multi_layer = key.split_layers(self.num_layers)
            # Only check the first layer
            if self.storage_manager.contains(
                keys_multi_layer[0], self.retrieve_locations
            ):
                continue

            # Allocate the memory object
            num_tokens = end - start
            kv_shape_single_layer = self.gpu_connector.get_shape(num_tokens)

            memory_objs_multi_layer = self.storage_manager.batched_allocate(
                kv_shape_single_layer,
                kv_dtype,
                batch_size=self.num_layers,
                fmt=self.fmt,
                busy_loop=self.config.get_extra_config_value("force_store_wait", False),
            )

            if memory_objs_multi_layer is None:
                logger.warning(
                    "Local cpu memory under pressure so"
                    " choosing to not store the KV cache."
                )
                break

            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)
            memory_objs.append(memory_objs_multi_layer)
            tot_token_num += num_tokens

            # Create KV event
            if self.kv_events_enabled and tokens is not None:
                stored_event = CacheStoreEvent(
                    block_hashes=[key.chunk_hash],
                    parent_block_hash=None if start == 0 else prev_key,
                    token_ids=[],
                    block_size=num_tokens,
                    lora_id=None,
                    medium="cpu",
                    lora_name=None,
                )
                if tokens is not None:
                    stored_event.token_ids = convert_tokens_to_list(
                        tokens,
                        start,
                        end,
                    )
                    if isinstance(tokens, torch.Tensor):
                        stored_event.medium = tokens.device
                logger.debug(
                    "Added kv cache event '%s' to kv cache events queue",
                    stored_event,
                )
                self.kv_events.append(stored_event)
                prev_key = key.chunk_hash

        if keys:
            # Transpose the keys and memory objects into layer major format
            memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]
            keys = [list(row) for row in zip(*keys, strict=False)]

            # Calculate total KV size for logging
            tot_kv_size = sum(
                mo.get_size() for layer_objs in memory_objs for mo in layer_objs
            )

            assert_layerwise_gpu_connector(self.gpu_connector)

            t_start = time.perf_counter()
            mem_obj_generator = self.gpu_connector.batched_from_gpu(
                memory_objs, starts, ends, **kwargs
            )

            next(mem_obj_generator)

            for layer_id in range(self.num_layers):
                yield
                next(mem_obj_generator)
                self.storage_manager.batched_put(
                    keys[layer_id], memory_objs[layer_id], location=self.store_location
                )

            tot_time = time.perf_counter() - t_start
            logger.info(
                "[req_id=%s] Stored %d out of total %d tokens. "
                "size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s",
                req_id,
                tot_token_num,
                len(tokens),
                tot_kv_size / 1024**3,
                tot_time * 1000,
                tot_kv_size / tot_time / 1024**3 if tot_time > 0 else 0,
            )
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield

        self.stats_monitor.on_store_finished(monitor_req_id, tot_token_num)
        yield

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve the KV caches from the cache engine. And put the retrieved
        KV cache to the serving engine via the GPU connector.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables). ``sync=False`` issues H2D without waiting
            and holds MemoryObjs until ``complete_load``.

        :return: the boolean mask indicating which tokens are retrieved. The
            length of the mask should be the same as the tokens. On CPU.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        sync = bool(kwargs.pop("sync", True))
        self._last_pending_load = None
        self._last_cancelled_load = None
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping retrieve operation")
            return torch.zeros(len(tokens), dtype=torch.bool)

        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve operation"
        )

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        tot_kv_size = 0

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)

        # KVCache Check logging
        self._log_kvcache_for_check(
            operation="retrieve",
            kwargs=kwargs,
            token_count=num_required_tokens,
            require_req_id=True,
        )

        retrieve_stats = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        reordered_chunks: List[ProcessedChunk] = []
        gds_leases: List[Any] = []
        if not self._is_passive():
            with retrieve_stats.profile_process_tokens():
                if str(getattr(self.config, "kv_io_mode", "dram")) == "gds":
                    reordered_chunks, tot_kv_size, gds_leases = (
                        self._gds_process_tokens_internal(
                            tokens,
                            mask,
                            ret_mask,
                            **kwargs,
                        )
                    )
                elif self.async_loading:
                    reordered_chunks, tot_kv_size = self._async_process_tokens_internal(  # noqa: E501
                        tokens,
                        mask,
                        ret_mask,
                        **kwargs,
                    )
                else:
                    reordered_chunks, tot_kv_size = self._process_tokens_internal(
                        tokens,
                        mask,
                        ret_mask,
                        **kwargs,
                    )

        # MLA TP (world_size>1): NCCL broadcast is required. world_size==1
        # has no peer — skip the leader .to(cuda) substitute and H2D from
        # CPU L1 straight into paged KV (debug / DP replica TP=1).
        if self._need_retrieve_broadcast():
            with retrieve_stats.profile_broadcast():
                with torch_dev.stream(self.broadcast_stream):
                    event, keepalive = self._broadcast_or_receive_memory_objs(
                        reordered_chunks,
                        ret_mask,
                        sync=sync,
                        to_gpu=True,
                        **kwargs,
                    )
                if not hasattr(self.gpu_connector, "load_stream"):
                    self.broadcast_stream.synchronize()
            # Engine must not pin GPU staging after retrieve returns.
            # Async keepalive holds only the last chunk still on load_stream.
            self._leader_gpu_substitute_objs = []
            if not sync and (reordered_chunks or keepalive):
                pending = PendingLoad(
                    req_id=str(req_id) if req_id is not None else "",
                    chunks=list(reordered_chunks),
                    event=event if event is not None else ReadyStoreEvent(),
                    retrieve_stats=retrieve_stats,
                    ret_mask=ret_mask,
                    kwargs_keepalive={
                        "memory_objs_for_togpu": keepalive,
                        **kwargs,
                    },
                )
                self._last_pending_load = pending
                return ret_mask
            self._release_retrieve_chunks(
                reordered_chunks, req_id, retrieve_stats, ret_mask
            )
            return ret_mask

        # NOTE(Jiayi): memory_obj doesn't have to be a pinned
        # cpu tensor for the sake of performance.
        # For example, disk->gpu is faster than disk->cpu->gpu.
        # RDMA is another example.
        memory_objs_for_togpu: List[Any] = []
        starts_list: List[int] = []
        ends_list: List[int] = []
        if len(reordered_chunks) > 0:
            with retrieve_stats.profile_to_gpu():
                _, memory_objs, starts, ends = zip(*reordered_chunks, strict=False)
                memory_objs_for_togpu = list(memory_objs)
                starts_list = list(starts)
                ends_list = list(ends)
                event = None
                if sync:
                    self.gpu_connector.batched_to_gpu(
                        memory_objs_for_togpu,
                        starts_list,
                        ends_list,
                        **kwargs,
                    )
                else:
                    batched_async = getattr(
                        self.gpu_connector, "batched_to_gpu_async", None
                    )
                    if callable(batched_async):
                        event = batched_async(
                            memory_objs_for_togpu,
                            starts_list,
                            ends_list,
                            **kwargs,
                        )
                    else:
                        self.gpu_connector.batched_to_gpu(
                            memory_objs_for_togpu,
                            starts_list,
                            ends_list,
                            **kwargs,
                        )

                if not sync:
                    pending = PendingLoad(
                        req_id=str(req_id) if req_id is not None else "",
                        chunks=list(reordered_chunks),
                        event=event if event is not None else ReadyStoreEvent(),
                        retrieve_stats=retrieve_stats,
                        ret_mask=ret_mask,
                        kwargs_keepalive={
                            "memory_objs_for_togpu": memory_objs_for_togpu,
                            **kwargs,
                        },
                        gds_leases=list(gds_leases),
                    )
                    self._last_pending_load = pending
                    return ret_mask

        if gds_leases:
            fake = PendingLoad(
                req_id=str(req_id) if req_id is not None else "",
                chunks=list(reordered_chunks),
                retrieve_stats=retrieve_stats,
                ret_mask=ret_mask,
                gds_leases=list(gds_leases),
            )
            self.complete_load(fake)
        else:
            self._release_retrieve_chunks(reordered_chunks, req_id, retrieve_stats, ret_mask)
        return ret_mask

    def _release_retrieve_chunks(
        self,
        reordered_chunks: List[Any],
        req_id: Any,
        retrieve_stats: Any,
        ret_mask: torch.Tensor,
    ) -> None:
        # TODO(Jiayi): Remove the following for loop with batched operations
        # TODO(Jiayi): Need to refactor the `remove_after_retrieve` logic.
        for key, memory_obj, _, _ in reordered_chunks:
            if self.remove_after_retrieve and not self._is_passive():
                assert self.storage_manager is not None
                self.storage_manager.remove(key, self.retrieve_locations)
                # Sync PDBackend.remove() does NOT call ref_count_down() internally
                # (unlike async PD and other backends), so we must call it manually.
                # See pd_backend.py line 605 TODO comment.
                if self._is_sync_pd_backend():
                    memory_obj.ref_count_down()
            else:
                if memory_obj.is_pinned:
                    memory_obj.unpin()
                memory_obj.ref_count_down()

        sm = getattr(self, "storage_manager", None)
        coordinator = getattr(sm, "preload_coordinator", None) if sm is not None else None
        if coordinator is not None and reordered_chunks:
            coordinator.on_h2d_complete(str(req_id) if req_id is not None else "")

        if retrieve_stats is not None:
            retrieved_tokens = torch.sum(ret_mask)
            self.stats_monitor.on_retrieve_finished(
                retrieve_stats,
                retrieved_tokens,
            )

    def _gds_align_bytes(self) -> int:
        native = self._gds_remote_native()
        if native is not None and hasattr(native, "align_bytes"):
            try:
                return max(1, int(native.align_bytes() or 4096))
            except Exception:
                return 4096
        return 4096

    def _gds_dma_len(self, logical_nbytes: int) -> int:
        return gds_dma_len(logical_nbytes, self._gds_align_bytes())

    def _alloc_gds_staging(
        self,
        kv_shapes: List[torch.Size],
        kv_dtypes: List[torch.dtype],
    ) -> Optional[MemoryObj]:
        """Test/prewarm helper. Store and retrieve hot paths must not call this."""
        connector = self.gpu_connector
        acquire = getattr(connector, "_acquire_gds_staging", None)
        if callable(acquire):
            lease = acquire(zero_padding=True)
            if lease is None:
                return None
            return lease.memory_obj
        return None

    def _gds_remote_native(self):
        sm = getattr(self, "storage_manager", None)
        backends = getattr(sm, "storage_backends", None) or {}
        for backend in backends.values():
            conn = getattr(backend, "connection", None)
            native = getattr(getattr(conn, "client", None), "surdata_native", None)
            if native is not None and hasattr(native, "object_get_to_gpu"):
                return native
        return None

    def _gds_adapter_backend_summary(self) -> str:
        sm = getattr(self, "storage_manager", None)
        backends = getattr(sm, "storage_backends", None) or {}
        if not backends:
            return "backends=<empty>"
        parts = []
        items = backends.items() if hasattr(backends, "items") else enumerate(backends)
        for name, backend in items:
            conn = getattr(backend, "connection", None)
            wrapped = None
            getter = getattr(conn, "getWrappedConnector", None)
            if callable(getter):
                try:
                    wrapped = getter()
                except Exception:
                    wrapped = "<error>"
            parts.append(
                "%s: backend=%s connection=%s wrapped=%s"
                % (
                    name,
                    type(backend).__name__,
                    type(conn).__name__,
                    type(wrapped).__name__,
                )
            )
        return "; ".join(parts)

    def _gds_remote_adapter(self):
        sm = getattr(self, "storage_manager", None)
        backends = getattr(sm, "storage_backends", None) or {}
        for backend in backends.values():
            conn = getattr(backend, "connection", None)
            adapter = unwrap_gds_remote_connector(conn)
            if adapter is not None:
                return adapter
        logger.error(
            "kv_io_mode=gds adapter missing submit_gds_object_put/get (%s)",
            self._gds_adapter_backend_summary(),
        )
        return None

    def _acquire_pd_header_host(self, nbytes: int) -> torch.Tensor:
        """Reuse a pinned host buffer for page-direct header H2D.

        Avoids ``pin_memory()`` on every chunk, which registers lock-page
        memory on the EngineCore thread.
        """
        need = max(1, int(nbytes))
        with self._pd_header_pin_lock:
            for buf in self._pd_header_pins:
                if int(buf.numel()) >= need and not bool(
                    getattr(buf, "_lmc_in_use", False)
                ):
                    buf._lmc_in_use = True  # type: ignore[attr-defined]
                    return buf.narrow(0, 0, need)
            pin = bool(torch.cuda.is_available())
            try:
                buf = torch.empty(need, dtype=torch.uint8, pin_memory=pin)
            except Exception:
                buf = torch.empty(need, dtype=torch.uint8)
            buf._lmc_in_use = True  # type: ignore[attr-defined]
            self._pd_header_pins.append(buf)
            if len(self._pd_header_pins) > 256:
                self._pd_header_pins = [
                    item
                    for item in self._pd_header_pins
                    if bool(getattr(item, "_lmc_in_use", False))
                ]
                if buf not in self._pd_header_pins:
                    self._pd_header_pins.append(buf)
            return buf.narrow(0, 0, need)

    def _release_pd_header_hosts(self, pending: PendingStore) -> None:
        """Mark header pin buffers free after the store no longer needs them."""
        keep = list(getattr(pending, "pd_keepalive", None) or [])
        if not keep:
            return
        ptrs = set()
        for item in keep:
            data_ptr = getattr(item, "data_ptr", None)
            if callable(data_ptr):
                try:
                    ptrs.add(int(data_ptr()))
                except Exception:
                    continue
        if not ptrs:
            return
        with self._pd_header_pin_lock:
            for buf in self._pd_header_pins:
                try:
                    base_ptr = int(buf.data_ptr())
                except Exception:
                    continue
                if base_ptr in ptrs:
                    buf._lmc_in_use = False  # type: ignore[attr-defined]

    def _fill_pd_header_host(self, packed: object) -> torch.Tensor:
        """Copy a packed header into a reused pinned host buffer.

        Args:
            packed: Header bytes (``bytes`` / ``bytearray`` / buffer).

        Returns:
            CPU ``uint8`` tensor of ``len(packed)`` suitable for H2D ``copy_``.
        """
        if isinstance(packed, (bytes, bytearray)):
            data = bytes(packed)
        else:
            data = bytes(packed)
        n = len(data)
        host = self._acquire_pd_header_host(max(n, 1))
        view = host if int(host.numel()) == n else host.narrow(0, 0, max(n, 0))
        if n == 0:
            return view
        src = (ctypes.c_char * n).from_buffer_copy(data)
        ctypes.memmove(int(view.data_ptr()), src, n)
        return view

    def _gds_store_stream(self, connector: Any) -> Tuple[Any, int]:
        """Return ``(store_stream, stream_ptr)`` after waiting on compute.

        Args:
            connector: GPU connector that may own ``store_stream``.

        Returns:
            Stream object and integer CUDA stream pointer. ``(None, 0)`` when
            CUDA or the store stream is unavailable.
        """
        stream = getattr(connector, "store_stream", None)
        if stream is None or not torch.cuda.is_available():
            return None, 0
        try:
            stream.wait_stream(torch.cuda.current_stream())
            return stream, int(stream.cuda_stream)
        except Exception:
            logger.debug("GDS store_stream wait skipped", exc_info=True)
            return None, 0

    def _note_gds_store_split(
        self,
        pending: PendingStore,
        cpu_prep_s: float,
        cuda_enqueue_s: float,
        native_submit_s: float,
    ) -> None:
        """Record and log EngineCore store split (CPU / CUDA enqueue / native)."""
        cpu_ms = max(0.0, float(cpu_prep_s) * 1000.0)
        cuda_ms = max(0.0, float(cuda_enqueue_s) * 1000.0)
        native_ms = max(0.0, float(native_submit_s) * 1000.0)
        pending.t_cpu_prep_ms = cpu_ms
        pending.t_cuda_enqueue_ms = cuda_ms
        pending.t_native_submit_ms = native_ms
        metrics = getattr(self, "_gds_metrics", None)
        if metrics is not None:
            metrics.note_store_split(
                int(cpu_ms * 1000.0),
                int(cuda_ms * 1000.0),
                int(native_ms * 1000.0),
            )
        logger.info(
            "GDS_STORE_SPLIT req=%s cpu_prep_ms=%.3f cuda_enqueue_ms=%.3f "
            "native_submit_ms=%.3f thread=%s",
            getattr(pending, "req_id", ""),
            cpu_ms,
            cuda_ms,
            native_ms,
            threading.current_thread().name,
        )

    def _enqueue_pd_header_copies(
        self,
        connector: Any,
        copy_jobs: List[Dict[str, Any]],
        copy_fn: Any,
    ) -> Any:
        """Enqueue header/residual copies on the store stream.

        Args:
            connector: GPU connector owning the staging arena.
            copy_jobs: Per-chunk host header, offsets, and residual layout.
            copy_fn: ``residual_copy_bytes`` (CUDA or CPU fallback).

        Returns:
            CUDA event recorded after the copies, or None on CPU.
        """
        if not copy_jobs:
            return None
        arena = connector.page_direct_staging_tensor()
        if arena is None:
            raise RuntimeError("page-direct staging arena missing during copy")
        store_stream, stream_ptr = self._gds_store_stream(connector)
        ctx = (
            torch.cuda.stream(store_stream)
            if store_stream is not None
            else nullcontext()
        )
        with ctx:
            for job in copy_jobs:
                packed_n = int(job["packed_n"])
                off = int(job["hdr_off"])
                host_hdr = job["host_hdr"]
                arena[off : off + packed_n].copy_(host_hdr, non_blocking=True)
                res_lease = job.get("res_lease")
                layout = job.get("layout")
                adm = job.get("adm")
                if res_lease is None or layout is None or not layout.residuals:
                    continue
                arena[res_lease[1] : res_lease[1] + res_lease[2]].zero_()
                for res in layout.residuals:
                    page = adm.page_views[res.page_index]
                    copy_fn(
                        res_lease[0] + int(res.staging_off),
                        int(page.gpu_addr) + int(res.page_off),
                        int(res.length),
                        stream_ptr,
                    )
        if not torch.cuda.is_available():
            return None
        event = torch.cuda.Event()
        if store_stream is not None:
            event.record(store_stream)
        else:
            event.record()
        return event

    def _release_gds_leases(self, pending) -> None:
        connector = self.gpu_connector
        release = getattr(connector, "release_gds_staging", None)
        pd_rel = getattr(connector, "release_page_direct_staging", None)
        leases = list(getattr(pending, "gds_leases", None) or [])
        pending.gds_leases = []
        for lease in leases:
            try:
                if (
                    isinstance(lease, tuple)
                    and len(lease) >= 3
                    and callable(pd_rel)
                ):
                    pd_rel(int(lease[1]), int(lease[2]))
                elif callable(release):
                    release(lease)
            except Exception:
                logger.exception(
                    "[req_id=%s] GDS lease release failed",
                    getattr(pending, "req_id", ""),
                )
        pd_rel = getattr(connector, "release_page_direct_staging", None)
        pd_leases = list(getattr(pending, "pd_leases", None) or [])
        pending.pd_leases = []
        if callable(pd_rel):
            for off, ln in pd_leases:
                try:
                    pd_rel(int(off), int(ln))
                except Exception:
                    logger.exception(
                        "[req_id=%s] page-direct staging release failed",
                        getattr(pending, "req_id", ""),
                    )
        self._release_pd_header_hosts(pending)
        pending.pd_keepalive = []
        self._release_direct_permit(pending)

    def _harvest_gds_store(self, pending: PendingStore) -> str:
        """Collect GDS put future and release bounce leases.

        Returns 'none' if there is no GDS subset, 'pending' if the future
        is still running, 'done' on success, 'failed' on put error.
        Host subset is left intact so a GDS failure cannot drop D2H keys.
        """
        fut = pending.put_future
        if fut is None and not pending.gds_leases and not getattr(
            pending, "pd_leases", None
        ):
            return "none"
        if fut is not None:
            done = getattr(fut, "done", None)
            if callable(done) and not done():
                return "pending"
            pending.t_native_done = time.perf_counter()
            stamp = getattr(fut, "stamp_pending", None)
            if callable(stamp):
                stamp(pending)
            exc = None
            try:
                getter = getattr(fut, "exception", None)
                if callable(getter):
                    exc = getter(timeout=0)
            except Exception as wait_exc:
                exc = wait_exc
            if exc is not None:
                pending.gds_put_error = exc
                pending.gds_state = GDS_PUT_FAILED
                metrics = getattr(self, "_gds_metrics", None)
                if metrics is not None:
                    metrics.store_future_error += 1
                logger.error(
                    "[req_id=%s] GDS put failed (host subset kept): %s "
                    "store_direct_chunks=%s store_d2h_chunks=%s "
                    "store_dropped_chunks=%s store_future_error=%s",
                    pending.req_id,
                    exc,
                    0 if metrics is None else metrics.store_direct_chunks,
                    0 if metrics is None else metrics.store_d2h_chunks,
                    0 if metrics is None else metrics.store_dropped_chunks,
                    0 if metrics is None else metrics.store_future_error,
                )
                self._release_gds_leases(pending)
                pending.put_future = None
                return "failed"
        self._release_gds_leases(pending)
        pending.put_future = None
        return "done"

    def _nbytes_for_tokens(self, num_tokens: int) -> int:
        shapes = self.metadata.get_shapes(num_tokens)
        dtypes = self.metadata.get_dtypes()
        if not isinstance(dtypes, list):
            dtypes = [dtypes] * len(shapes)
        total = 0
        for shape, dtype in zip(shapes, dtypes, strict=False):
            total += int(shape.numel() * dtype.itemsize)
        return total

    def _admit_store_d2h(self, pending: PendingStore, nbytes: int) -> bool:
        cap = int(getattr(self.config, "kv_gds_dram_fallback_bytes", 0) or 0)
        with self._gds_d2h_lock:
            if cap > 0 and self._gds_d2h_bytes + int(nbytes) > cap:
                return False
            if pending.d2h_admitted <= 0:
                self._gds_d2h_inflight += 1
                pending.d2h_admitted = 1
            self._gds_d2h_bytes += int(nbytes)
            pending.d2h_admitted_bytes += int(nbytes)
        return True

    def _release_store_d2h_admit(self, pending: PendingStore) -> None:
        nbytes = int(getattr(pending, "d2h_admitted_bytes", 0) or 0)
        admitted = int(getattr(pending, "d2h_admitted", 0) or 0)
        if nbytes <= 0 and admitted <= 0:
            return
        with self._gds_d2h_lock:
            self._gds_d2h_bytes = max(0, self._gds_d2h_bytes - nbytes)
            if admitted > 0:
                self._gds_d2h_inflight = max(0, self._gds_d2h_inflight - admitted)
        pending.d2h_admitted_bytes = 0
        pending.d2h_admitted = 0
        with self._gds_d2h_lock:
            self._gds_d2h_lock.notify_all()

    def _gds_wait_timeout_s(self) -> float:
        ms = int(getattr(self.config, "kv_gds_staging_wait_timeout_ms", 0) or 0)
        return max(0.0, float(ms) / 1000.0)

    def _gds_max_waiters(self) -> int:
        return int(getattr(self.config, "kv_gds_staging_max_waiters", 32) or 32)

    def _try_host_allocate(
        self, pending: PendingStore, kv_shapes, kv_dtypes, nbytes: int
    ) -> Any:
        """Try D2H admission + host MemoryObj once. Never wait on EngineCore."""
        metrics = getattr(self, "_gds_metrics", None)
        if not self._admit_store_d2h(pending, nbytes):
            if metrics is not None:
                metrics.d2h_budget_exhausted += 1
            return None
        memory_obj = None
        if self.storage_manager is not None:
            memory_obj = self.storage_manager.allocate(
                kv_shapes,
                kv_dtypes,
                busy_loop=False,
                fmt=self.fmt,
            )
        if memory_obj is not None:
            return memory_obj
        with self._gds_d2h_lock:
            self._gds_d2h_bytes = max(0, self._gds_d2h_bytes - nbytes)
            pending.d2h_admitted_bytes = max(
                0, int(pending.d2h_admitted_bytes) - nbytes
            )
            self._gds_d2h_lock.notify_all()
        if metrics is not None:
            metrics.d2h_alloc_fail += 1
        return None

    def _wait_host_allocate(
        self, pending: PendingStore, kv_shapes, kv_dtypes, nbytes: int
    ) -> Any:
        """Backward-compatible name for store D2H allocate. Never sleeps."""
        return self._try_host_allocate(pending, kv_shapes, kv_dtypes, nbytes)

    def _gds_immediate_drop(self) -> bool:
        pol = str(getattr(self.config, "kv_gds_store_overload_policy", "") or "")
        return pol == GDS_STORE_OVERLOAD_IMMEDIATE_DROP

    def _drop_gds_store(self, pending: PendingStore, reason: str) -> PendingStore:
        metrics = getattr(self, "_gds_metrics", None)
        n = int(getattr(pending, "requested_chunks", 0) or len(pending.keys) or 0)
        if metrics is not None:
            metrics.store_dropped_chunks += max(1, n)
        pending.store_reject_reason = reason
        pending.error = RuntimeError("GDS store dropped: %s" % reason)
        self._release_gds_leases(pending)
        self._release_direct_permit(pending)
        pending.keys = []
        pending.memory_objs = []
        pending.gds_keys = []
        pending.accepted_chunks = 0
        pending.dropped_chunks = n
        pending.put_future = None
        pending.event = ReadyStoreEvent()
        logger.info(
            "[req_id=%s] GDS store immediate_drop reason=%s chunks=%s",
            pending.req_id,
            reason,
            n,
        )
        return pending

    def _try_acquire_direct_permit(self, pending: PendingStore) -> bool:
        """Try-acquire a page-direct GDS store permit. Never waits."""
        cap = int(getattr(self.config, "kv_gds_direct_store_max_inflight", 0) or 0)
        if cap <= 0:
            return True
        with self._gds_direct_lock:
            if self._gds_direct_inflight >= cap:
                return False
            self._gds_direct_inflight += 1
            pending.direct_permit = 1
            return True

    def _release_direct_permit(self, pending: PendingStore) -> None:
        if int(getattr(pending, "direct_permit", 0) or 0) <= 0:
            return
        pending.direct_permit = 0
        with self._gds_direct_lock:
            self._gds_direct_inflight = max(0, self._gds_direct_inflight - 1)

    def _maybe_init_page_direct(self, **kwargs) -> None:
        connector = self.gpu_connector
        if connector is None:
            return
        init_ptr = getattr(connector, "initialize_kvcaches_ptr", None)
        if callable(init_ptr) and kwargs:
            init_ptr(**kwargs)
        if callable(getattr(connector, "page_direct_enabled", None)) and connector.page_direct_enabled():
            return
        adapter = self._gds_remote_adapter()
        if adapter is None:
            return
        geo_fn = getattr(adapter, "kvb_gds_geometry", None)
        if not callable(geo_fn):
            logger.warning("page-direct disabled: kvb_gds_geometry missing")
            return
        try:
            geo = geo_fn()
        except Exception:
            logger.exception("page-direct geometry query failed")
            metrics = getattr(self, "_gds_metrics", None)
            if metrics is not None:
                metrics.gds_cache_register_fail += 1
            return
        if int(getattr(geo, "gds_ok", 0) or 0) == 0:
            logger.warning(
                "page-direct disabled: GDS preflight gds_ok=0 writev_ex=%s readv_ex=%s",
                getattr(geo, "writev_ex_ok", 0),
                getattr(geo, "readv_ex_ok", 0),
            )
            return
        configure = getattr(connector, "configure_page_direct_registration", None)
        if callable(configure):
            configure(adapter.gds_buf_register, adapter.gds_buf_deregister)
        arena = int(getattr(self.config, "kv_gds_staging_arena_bytes", 0) or 0)
        if arena < 1:
            arena = 64 * 1024 * 1024
        try:
            connector.register_page_direct_kv_ranges(
                sub_stripe=int(geo.sub_stripe_bytes or 1),
                sector=int(geo.sector_bytes or 4096),
                registration_limit=int(geo.gds_registration_limit or 32),
                staging_arena_bytes=arena,
            )
        except Exception:
            logger.exception("page-direct KV register failed")
            metrics = getattr(self, "_gds_metrics", None)
            if metrics is not None:
                metrics.gds_cache_register_fail += 1
            return
        if not connector.page_direct_enabled():
            metrics = getattr(self, "_gds_metrics", None)
            if metrics is not None:
                metrics.gds_cache_register_fail += 1
            return
        probe = connector.page_direct_probe()
        views_n = max(1, int(getattr(connector, "num_layers", 1) or 1))
        page_bytes = int(getattr(connector, "_pd_page_bytes", 0) or 0)
        bs = int(getattr(connector, "block_size", 16) or 16)
        fmt = str(getattr(connector, "_pd_format_name", "") or "")
        kv_layout = str(getattr(connector, "_pd_kv_layout", "NHD") or "NHD")
        dtype_itemsize = 2
        kvc = getattr(connector, "kvcaches", None)
        if kvc:
            dtype_itemsize = int(kvc[0].element_size())
        fid = physical_page_format_id(
            format_name=fmt,
            kv_layout=kv_layout,
            dtype_itemsize=dtype_itemsize,
            vllm_block_size=bs,
            page_bytes=page_bytes,
            layer_count=views_n,
            kv_factor=1 if getattr(self.metadata, "use_mla", False) else 2,
            enum_order="layer_then_block",
        )
        chunk = int(self.metadata.chunk_size)
        max_pages = max(1, views_n * max(1, chunk // max(1, bs)))
        shapes = self.metadata.get_shapes(chunk)
        dtypes = self.metadata.get_dtypes()
        canon = 0
        if shapes:
            dt = dtypes[0] if isinstance(dtypes, list) else dtypes
            canon = int(shapes[0].numel() * dt.itemsize)
        self._pd_profile = compute_profile(
            max_pages=max_pages,
            page_bytes=page_bytes,
            sub_stripe=int(geo.sub_stripe_bytes or 1),
            sector=int(geo.sector_bytes or 4096),
            align=int(geo.align_bytes or 4096),
            physical_page_format_id=fid,
            canonical_payload_bytes=canon,
        )
        adapter._kvb_v2_profile = self._pd_profile
        zr = bool(probe.zero_residual) if probe else False
        logger.info(
            "page-direct profile extent=%s header=%s zero_residual=%s "
            "page_bytes=%s align=%s required_block_multiple=%s "
            "current_block_size=%s format_id=%s",
            self._pd_profile.record_extent_bytes,
            self._pd_profile.header_region_bytes,
            zr,
            page_bytes,
            int(geo.align_bytes or 4096),
            int(probe.required_block_multiple) if probe else 0,
            bs,
            fid,
        )
        if probe is not None and not zr:
            logger.warning(
                "page-direct zero_residual=0 (geometry, not hard-gate false "
                "positive) page_bytes=%s align=%s required_block_multiple=%s "
                "current_block_size=%s warn=%s",
                page_bytes,
                int(probe.align),
                int(probe.required_block_multiple),
                bs,
                probe.warn,
            )

    def _maybe_prewarm_gds_pool(self, **kwargs) -> None:
        if str(getattr(self.config, "kv_io_mode", "dram")) != "gds":
            return
        connector = self.gpu_connector
        supports = getattr(connector, "supports_gds_kv_pages", None)
        if connector is None or not (callable(supports) and supports()):
            raise RuntimeError(
                "kv_io_mode=gds requires VLLMPagedMemGPUConnectorV2 gather/scatter"
            )
        self._maybe_init_page_direct(**kwargs)
        return
    def issue_store_gds(
        self,
        tokens: Optional[Union[torch.Tensor, list[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[PendingStore]:
        """Prepare keys without L1, gather into GPU bounce, submit GDS put."""
        pending = self.prepare_store(
            tokens=tokens,
            hashes=hashes,
            offsets=offsets,
            mask=mask,
            **kwargs,
        )
        if pending is None:
            return None
        pending.kwargs_keepalive["_issue_store_kwargs"] = {
            "tokens": tokens,
            "hashes": hashes,
            "offsets": offsets,
            "mask": mask,
            **kwargs,
        }
        return self._issue_store_gds(pending, **kwargs)

    def _store_fallback_d2h(self, pending: PendingStore) -> Optional[PendingStore]:
        self._release_gds_leases(pending)
        args = dict((pending.kwargs_keepalive or {}).get("_issue_store_kwargs") or {})
        args["allocate_l1"] = True
        logger.warning(
            "[req_id=%s] GDS store fallback to D2H+host put", pending.req_id
        )
        result = self.issue_store_d2h(**args)
        if result is not None:
            result.gds_fallback_d2h = True
            result.gds_state = ""
            result.gds_leases = []
        return result

    def _issue_store_gds(self, pending: PendingStore, **kwargs) -> Optional[PendingStore]:
        if kwargs:
            pending.kwargs_keepalive.update(kwargs)
        starts = list(pending.kwargs_keepalive.get("starts") or [])
        ends = list(pending.kwargs_keepalive.get("ends") or [])
        gather_kwargs = {
            k: v
            for k, v in pending.kwargs_keepalive.items()
            if k not in ("starts", "ends", "_issue_store_kwargs")
        }
        connector = self.gpu_connector
        n = len(pending.keys)
        if n < 1:
            return None
        adapter = self._gds_remote_adapter()
        try:
            self._maybe_prewarm_gds_pool(**gather_kwargs)
        except Exception as exc:
            pending.error = exc
            logger.exception(
                "[req_id=%s] GDS page-direct init failed", pending.req_id
            )
            return pending
        metrics = getattr(self, "_gds_metrics", None)
        pending.t_gds_queue = time.perf_counter()
        slot_mapping = gather_kwargs.get("slot_mapping")
        pd_on = bool(
            connector is not None
            and callable(getattr(connector, "page_direct_enabled", None))
            and connector.page_direct_enabled()
            and self._pd_profile is not None
            and adapter is not None
            and callable(getattr(adapter, "kvb_store_from_gpu_iov", None))
            and slot_mapping is not None
        )
        host_idx: List[int] = []
        gpu_records: List[dict] = []
        gpu_idx: List[int] = []
        demote_reason = ""
        if pd_on and not self._try_acquire_direct_permit(pending):
            if metrics is not None:
                metrics.direct_permit_rejected += 1
            if self._gds_immediate_drop():
                return self._drop_gds_store(pending, "direct_gds_saturated")
            demote_reason = "direct_gds_saturated"
            pd_on = False
        copy_jobs: List[Dict[str, Any]] = []
        copy_fn = None
        sdk_profile = None
        t_split0 = time.perf_counter()
        if pd_on:
            _require_cpu_slot_mapping(slot_mapping)
            init_ptr = getattr(connector, "initialize_kvcaches_ptr", None)
            if callable(init_ptr):
                init_ptr(**gather_kwargs)
            try:
                copy_fn = getattr(
                    __import__("lmcache.c_ops", fromlist=["residual_copy_bytes"]),
                    "residual_copy_bytes",
                    None,
                )
            except Exception:
                copy_fn = None
            if copy_fn is None:
                from lmcache.v1.platform.torch_ops import residual_copy_bytes as copy_fn
            profile = self._pd_profile
            align = int(profile.align)
            probe = connector.page_direct_probe()
            zero_res = bool(probe and probe.zero_residual)
            max_w = self._gds_max_waiters()
            arena_fn = getattr(connector, "page_direct_staging_tensor", None)
            for i in range(n):
                adm = connector.get_page_direct_views(
                    slot_mapping, int(starts[i]), int(ends[i])
                )
                if not adm.complete_pages:
                    logger.warning(
                        "[req_id=%s] page-direct chunk %s not complete (%s); emergency D2H",
                        pending.req_id,
                        i,
                        adm.reason,
                    )
                    if metrics is not None:
                        metrics.page_direct_unavailable += 1
                    demote_reason = "page_direct_unavailable"
                    break
                hdr_need = int(profile.header_region_bytes)
                res_need = 0 if zero_res else int(profile.residual_capacity)
                hdr_lease = connector.acquire_page_direct_staging(
                    hdr_need, align=align, timeout_s=0.0, max_waiters=max_w
                )
                if hdr_lease is None:
                    if metrics is not None:
                        metrics.staging_lease_starved += 1
                    demote_reason = "staging_lease_starved"
                    break
                res_lease = None
                if res_need:
                    res_lease = connector.acquire_page_direct_staging(
                        res_need, align=align, timeout_s=0.0, max_waiters=max_w
                    )
                    if res_lease is None:
                        connector.release_page_direct_staging(hdr_lease[1], hdr_lease[2])
                        if metrics is not None:
                            metrics.staging_lease_starved += 1
                        demote_reason = "staging_lease_starved"
                        break
                rec, layout = build_raw_page_record(
                    adm.page_views,
                    profile,
                    header_ptr=hdr_lease[0],
                    residual_ptr=0 if res_lease is None else res_lease[0],
                    residual_len=0 if res_lease is None else res_lease[2],
                    kv_block_index=i,
                )
                packed = rec.pop("packed_header")
                rec.pop("layout", None)
                if not callable(arena_fn) or arena_fn() is None:
                    connector.release_page_direct_staging(hdr_lease[1], hdr_lease[2])
                    if res_lease is not None:
                        connector.release_page_direct_staging(
                            res_lease[1], res_lease[2]
                        )
                    if metrics is not None:
                        metrics.staging_lease_starved += 1
                    demote_reason = "staging_lease_starved"
                    break
                host_hdr = self._fill_pd_header_host(packed)
                pending.pd_keepalive.append(host_hdr)
                copy_jobs.append(
                    {
                        "host_hdr": host_hdr,
                        "packed_n": len(packed),
                        "hdr_off": int(hdr_lease[1]),
                        "res_lease": res_lease,
                        "layout": layout,
                        "adm": adm,
                    }
                )
                pending.pd_leases.append((hdr_lease[1], hdr_lease[2]))
                if res_lease is not None:
                    pending.pd_leases.append((res_lease[1], res_lease[2]))
                gpu_records.append(rec)
                gpu_idx.append(i)
        if not pd_on or demote_reason:
            if self._gds_immediate_drop() and demote_reason:
                return self._drop_gds_store(pending, demote_reason)
            if gpu_records or pending.pd_leases:
                logger.warning(
                    "[req_id=%s] demoting all %s chunks to D2H reason=%s",
                    pending.req_id,
                    n,
                    demote_reason or "page_direct_off",
                )
                gpu_records = []
                gpu_idx = []
            self._release_gds_leases(pending)
            host_idx = list(range(n))
            pending.store_reject_reason = demote_reason

        host_keys: List[CacheEngineKey] = []
        host_starts: List[int] = []
        host_ends: List[int] = []
        host_objs: List[MemoryObj] = []
        host_failed = False
        for i in host_idx:
            num_tokens = int(ends[i]) - int(starts[i])
            kv_shapes = self.metadata.get_shapes(num_tokens)
            kv_dtypes = self.metadata.get_dtypes()
            nbytes = self._nbytes_for_tokens(num_tokens)
            memory_obj = self._try_host_allocate(pending, kv_shapes, kv_dtypes, nbytes)
            if memory_obj is None:
                logger.warning(
                    "[req_id=%s] D2H allocate failed for chunk %s/%s",
                    pending.req_id,
                    i,
                    n,
                )
                host_failed = True
                break
            host_keys.append(pending.keys[i])
            host_starts.append(starts[i])
            host_ends.append(ends[i])
            host_objs.append(memory_obj)
        if host_failed:
            for obj in host_objs:
                try:
                    obj.ref_count_down()
                except Exception:
                    logger.exception(
                        "[req_id=%s] D2H rollback ref_count_down failed",
                        pending.req_id,
                    )
            self._release_store_d2h_admit(pending)
            pending.store_reject_reason = pending.store_reject_reason or "d2h_budget_exhausted"
            pending.error = pending.error or RuntimeError(
                "GDS store D2H allocate failed (no partial write)"
            )
            pending.keys = []
            pending.memory_objs = []
            pending.gds_keys = []
            pending.dropped_chunks = 0
            pending.requested_chunks = n
            pending.accepted_chunks = 0
            self._note_gds_store_split(
                pending, time.perf_counter() - t_split0, 0.0, 0.0
            )
            return pending

        pending.dropped_chunks = 0
        pending.requested_chunks = n
        pending.accepted_chunks = len(gpu_records) + len(host_objs)
        pending.gds_keys = [pending.keys[i] for i in gpu_idx]
        pending.gds_objs = []
        pending.gds_leases = []
        pending.keys = host_keys
        pending.memory_objs = host_objs
        pending.kwargs_keepalive["starts"] = [starts[i] for i in gpu_idx] + host_starts
        pending.kwargs_keepalive["ends"] = [ends[i] for i in gpu_idx] + host_ends
        pending.kwargs_keepalive["gds_starts"] = [starts[i] for i in gpu_idx]
        pending.kwargs_keepalive["gds_ends"] = [ends[i] for i in gpu_idx]
        pending.kwargs_keepalive["host_starts"] = host_starts
        pending.kwargs_keepalive["host_ends"] = host_ends
        tot_kv = 0
        tot_tok = 0
        for start, end in zip(host_starts, host_ends, strict=False):
            tot_tok += int(end) - int(start)
        for obj in host_objs:
            tot_kv += int(obj.get_size())
        for i in gpu_idx:
            tot_tok += int(ends[i]) - int(starts[i])
        pending.tot_kv_size = tot_kv
        pending.tot_token_num = tot_tok
        if metrics is not None:
            metrics.store_direct_chunks += len(gpu_records)
            metrics.store_d2h_chunks += len(host_objs)
            if host_objs:
                metrics.page_direct_emergency_d2h += len(host_objs)
        if not gpu_records and not host_objs:
            pending.store_reject_reason = pending.store_reject_reason or "d2h_budget_exhausted"
            pending.error = pending.error or RuntimeError(
                "GDS store produced no chunks (backpressure)"
            )
            self._note_gds_store_split(
                pending, time.perf_counter() - t_split0, 0.0, 0.0
            )
            return pending
        t_cpu_end = time.perf_counter()
        cuda_event = None
        if gpu_records and copy_jobs:
            try:
                cuda_event = self._enqueue_pd_header_copies(
                    connector, copy_jobs, copy_fn
                )
            except Exception as exc:
                pending.gds_put_error = exc
                pending.gds_state = GDS_PUT_FAILED
                if metrics is not None:
                    metrics.kvb_gpu_future_error += 1
                logger.exception(
                    "[req_id=%s] page-direct header copy enqueue failed",
                    pending.req_id,
                )
                self._release_gds_leases(pending)
                gpu_records = []
                copy_jobs = []
                if not host_objs:
                    pending.error = exc
                    pending.store_reject_reason = "native_put_failed"
                    self._note_gds_store_split(
                        pending, t_cpu_end - t_split0, 0.0, 0.0
                    )
                    return pending
        t_cuda_end = time.perf_counter()
        if gpu_records and adapter is not None:
            try:
                pending.put_future = adapter.kvb_store_from_gpu_iov(
                    pending.token_ids or gather_kwargs.get("tokens"),
                    gpu_records,
                    sdk_profile
                    if sdk_profile is not None
                    else profile_as_sdk_dict(
                        self._pd_profile, SURDATA_SDK_KV_LAYOUT_SPLIT_REGION
                    ),
                    cuda_event=cuda_event,
                )
                pending.gds_state = GDS_PUT_SUBMITTED
                pending.t_native_first_submit = time.perf_counter()
            except Exception as exc:
                pending.gds_put_error = exc
                pending.gds_state = GDS_PUT_FAILED
                if metrics is not None:
                    metrics.kvb_gpu_future_error += 1
                logger.exception(
                    "[req_id=%s] kvb_store_from_gpu_iov failed", pending.req_id
                )
                self._release_gds_leases(pending)
                pending.put_future = None
                pending.store_reject_reason = "native_put_failed"
                if not host_objs:
                    pending.error = exc
                    self._note_gds_store_split(
                        pending,
                        t_cpu_end - t_split0,
                        t_cuda_end - t_cpu_end,
                        time.perf_counter() - t_cuda_end,
                    )
                    return pending
        t_native_end = time.perf_counter()
        self._note_gds_store_split(
            pending,
            t_cpu_end - t_split0,
            t_cuda_end - t_cpu_end,
            t_native_end - t_cuda_end,
        )
        store_stats = pending.store_stats
        assert connector is not None
        host_event = None
        if host_objs:
            logger.warning(
                "[req_id=%s] emergency D2H KVB v2 for %s chunks",
                pending.req_id,
                len(host_objs),
            )
            with store_stats.profile_from_gpu():
                host_event = connector.batched_from_gpu_async(
                    host_objs, host_starts, host_ends, **gather_kwargs
                )
            pending.gds_fallback_d2h = True
            pending.event = host_event if host_event is not None else ReadyStoreEvent()
        else:
            pending.event = ReadyStoreEvent()
        if pending.event is None:
            pending.event = ReadyStoreEvent()
        logger.info(
            "[req_id=%s] GDS store split direct=%s d2h=%s dropped=%s "
            "store_direct_chunks=%s store_d2h_chunks=%s store_dropped_chunks=%s "
            "store_slot_starved=%s store_future_error=%s "
            "staging_lease_starved=%s d2h_budget_exhausted=%s "
            "page_direct_emergency_d2h=%s kvb_gpu_future_error=%s "
            "direct_permit_rejected=%s reason=%s",
            pending.req_id,
            len(gpu_records),
            len(host_objs),
            0,
            0 if metrics is None else metrics.store_direct_chunks,
            0 if metrics is None else metrics.store_d2h_chunks,
            0 if metrics is None else metrics.store_dropped_chunks,
            0 if metrics is None else metrics.store_slot_starved,
            0 if metrics is None else metrics.store_future_error,
            0 if metrics is None else metrics.staging_lease_starved,
            0 if metrics is None else metrics.d2h_budget_exhausted,
            0 if metrics is None else metrics.page_direct_emergency_d2h,
            0 if metrics is None else metrics.kvb_gpu_future_error,
            0 if metrics is None else metrics.direct_permit_rejected,
            pending.store_reject_reason,
        )
        return pending


    def _submit_page_direct_reads(
        self,
        pending: PendingLoad,
        tokens,
        mask,
        **kwargs,
    ) -> None:
        """Submit GDS reads for a looked-up prefix. Never waits on DMA."""
        connector = self.gpu_connector
        profile = self._pd_profile
        adapter = pending.pd_adapter
        hit = pending.pd_hit
        prefix = int(pending.pd_prefix or 0)
        spans = list(pending.pd_spans or [])
        slot_mapping = kwargs.get("slot_mapping")
        if (
            connector is None
            or profile is None
            or adapter is None
            or hit is None
            or prefix <= 0
            or slot_mapping is None
        ):
            raise RuntimeError("page-direct retrieve missing lookup state")
        _require_cpu_slot_mapping(slot_mapping)
        submit_fn = getattr(adapter, "kvb_read_to_gpu_submit", None)
        if not callable(submit_fn):
            raise RuntimeError("kvb_read_to_gpu_submit unavailable")
        max_w = self._gds_max_waiters()
        align = int(profile.align)
        extent = int(profile.record_extent_bytes)
        sdk_profile = profile_as_sdk_dict(profile, SURDATA_SDK_KV_LAYOUT_SPLIT_REGION)
        indices: List[int] = []
        ptrs: List[int] = []
        views: List[Any] = []
        leases: List[Any] = []
        try:
            for i in range(prefix):
                start, end = spans[i]
                adm = connector.get_page_direct_views(
                    slot_mapping, int(start), int(end)
                )
                if not adm.complete_pages:
                    raise RuntimeError(adm.reason or "incomplete page views")
                lease = connector.acquire_page_direct_staging(
                    extent, align=align, timeout_s=0.0, max_waiters=max_w
                )
                if lease is None:
                    raise RuntimeError("page-direct retrieve staging unavailable")
                leases.append(lease)
                views.append(adm.page_views)
                indices.append(i)
                ptrs.append(int(lease[0]))
            pending.gds_leases = leases
            pending.pd_page_views = views
            pending.gds_read_fut = submit_fn(
                hit, indices, ptrs, sdk_profile, 0
            )
            pending.gds_state = GDS_GATHER_SUBMITTED
            pending.t_gds_submit = time.perf_counter()
            pending.xfer_state = XFER_GATHERING
        except Exception:
            for lease in leases:
                try:
                    connector.release_page_direct_staging(lease[1], lease[2])
                except Exception:
                    logger.exception("release staging after submit fail")
            pending.gds_leases = []
            raise

    def _try_complete_page_direct(self, pending: PendingLoad) -> str:
        """Poll GDS reads, copy headers, scatter. Never .cpu()/wait/synchronize."""
        adapter = pending.pd_adapter
        fut = pending.gds_read_fut
        if fut is None:
            pending.error = RuntimeError("page-direct read future missing")
            pending.xfer_state = XFER_CANCELLING
            return "error"
        poll_fn = getattr(adapter, "kvb_read_to_gpu_poll", None)
        if callable(poll_fn):
            done, rc = poll_fn(fut)
        else:
            done = bool(fut.done())
            rc = int(fut.rc() or 0) if done else 0
        if not done:
            pending.xfer_state = XFER_GATHERING
            return "pending_gather"
        if rc != 0:
            err = ""
            em = getattr(fut, "error_message", None)
            if callable(em):
                err = str(em() or "")
            pending.error = RuntimeError(
                "GDS page-direct read failed rc=%s %s" % (rc, err)
            )
            pending.xfer_state = XFER_CANCELLING
            self._release_gds_leases(pending)
            return "error"
        connector = self.gpu_connector
        profile = self._pd_profile
        if connector is None or profile is None:
            pending.error = RuntimeError("page-direct complete missing connector")
            pending.xfer_state = XFER_CANCELLING
            self._release_gds_leases(pending)
            return "error"
        arena = connector.page_direct_staging_tensor()
        if arena is None:
            pending.error = RuntimeError("page-direct staging arena missing")
            pending.xfer_state = XFER_CANCELLING
            self._release_gds_leases(pending)
            return "error"
        hdr_need = int(profile.header_region_bytes)
        if pending.pd_header_event is None:
            hosts: List[Any] = []
            for lease in pending.gds_leases:
                host = torch.empty(hdr_need, dtype=torch.uint8, pin_memory=True)
                host.copy_(
                    arena[lease[1] : lease[1] + hdr_need],
                    non_blocking=True,
                )
                hosts.append(host)
            pending.pd_header_hosts = hosts
            event = None
            if torch.cuda.is_available():
                event = torch.cuda.Event()
                event.record()
            pending.pd_header_event = event
            pending.gds_state = GDS_DMA_DONE
            pending.t_dma_done = time.perf_counter()
            pending.xfer_state = XFER_GATHERING
            if event is not None:
                return "pending_gather"
        event = pending.pd_header_event
        if event is not None:
            query = getattr(event, "query", None)
            if callable(query) and not bool(query()):
                pending.xfer_state = XFER_GATHERING
                return "pending_gather"
        copy_fn = getattr(
            __import__("lmcache.c_ops", fromlist=["residual_copy_bytes"]),
            "residual_copy_bytes",
            None,
        )
        if copy_fn is None:
            from lmcache.v1.platform.torch_ops import residual_copy_bytes as copy_fn
        from lmcache.v1.kvb_v2 import PAYLOAD_ENCODING_RAW_PAGES, unpack_header

        tokens = (pending.kwargs_keepalive or {}).get("_gather_tokens")
        ntok = len(tokens) if tokens is not None else 0
        ret_mask = pending.ret_mask
        if ret_mask is None:
            ret_mask = torch.zeros(ntok, dtype=torch.bool)
            pending.ret_mask = ret_mask
        spans = list(pending.pd_spans or [])
        views = list(pending.pd_page_views or [])
        leases = list(pending.gds_leases or [])
        hosts = list(pending.pd_header_hosts or [])
        prefix = min(len(spans), len(views), len(leases), len(hosts))
        try:
            for i in range(prefix):
                hdr = bytes(hosts[i].numpy())
                parsed = unpack_header(hdr, profile.physical_page_format_id)
                if parsed.payload_encoding != PAYLOAD_ENCODING_RAW_PAGES:
                    raise RuntimeError("canonical encoding on GPU retrieve")
                page_views = views[i]
                if parsed.page_count != len(page_views):
                    raise RuntimeError("KVB v2 page_count does not match dest views")
                lease = leases[i]
                for di, desc in enumerate(parsed.descriptors[: parsed.page_count]):
                    page = page_views[di]
                    head = int(desc.page_payload_off)
                    body = int(desc.body_len)
                    residual = int(desc.residual_len)
                    if body:
                        copy_fn(
                            int(page.gpu_addr) + head,
                            lease[0] + int(desc.body_record_off),
                            body,
                            0,
                        )
                    if residual:
                        if head:
                            copy_fn(
                                int(page.gpu_addr),
                                lease[0] + int(desc.residual_record_off),
                                head,
                                0,
                            )
                        tail = residual - head
                        if tail > 0:
                            copy_fn(
                                int(page.gpu_addr) + head + body,
                                lease[0] + int(desc.residual_record_off) + head,
                                tail,
                                0,
                            )
                start, end = spans[i]
                ret_mask[int(start) : int(end)] = True
        except Exception as exc:
            pending.error = exc
            pending.xfer_state = XFER_CANCELLING
            self._release_gds_leases(pending)
            return "error"
        pending.chunks = []
        pending.h2d_issued = True
        pending.xfer_state = XFER_GPU_SCATTER_IN_FLIGHT
        pending.gds_state = GDS_HBM_ASSEMBLED
        pending.event = None
        pending.t_h2d_issue = time.perf_counter()
        return "issued"

    def _retrieve_page_direct(
        self,
        adapter,
        hit,
        prefix: int,
        spans: List[tuple[int, int]],
        ret_mask,
        slot_mapping,
    ) -> None:
        """Compatibility wrapper: submit then poll until scatter completes."""
        pending = PendingLoad(req_id="", chunks=[])
        pending.pd_adapter = adapter
        pending.pd_hit = hit
        pending.pd_prefix = int(prefix)
        pending.pd_spans = list(spans[:prefix])
        pending.kwargs_keepalive = {"slot_mapping": slot_mapping}
        self._submit_page_direct_reads(
            pending, None, None, slot_mapping=slot_mapping
        )
        while True:
            status = self._try_complete_page_direct(pending)
            if status == "pending_gather":
                continue
            if status == "error":
                raise pending.error or RuntimeError("page-direct retrieve failed")
            break
        if pending.ret_mask is not None and ret_mask is not None:
            n = min(int(ret_mask.numel()), int(pending.ret_mask.numel()))
            ret_mask[:n] = pending.ret_mask[:n]

    def _gds_process_tokens_internal(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ):
        """MN prefix lookup, then GPU page-direct retrieve when slot_mapping is set.

        Lookup-time gather (no ``slot_mapping``) returns empty chunks; the
        lookup ACK already used MN ``contains``. GDS never host-decodes
        RAW_PAGES via ``kvb_get_prefix``.
        """
        chunks: List[ProcessedChunk] = []
        leases: List[Any] = []
        request_configs = kwargs.get("request_configs")
        keys: List[CacheEngineKey] = []
        spans: List[tuple[int, int]] = []
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            keys.append(key)
            spans.append((start, end))
        if not keys:
            logger.info(
                "GDS gather has no keys slot_mapping=%s",
                kwargs.get("slot_mapping") is not None,
            )
            return chunks, 0, leases
        prefix = 0
        hit = None
        adapter = self._gds_remote_adapter()
        hit_fn = getattr(adapter, "lookup_prefix_hit", None)
        if not callable(hit_fn):
            backends = getattr(self.storage_manager, "storage_backends", None) or {}
            for backend in backends.values():
                hit_fn = getattr(backend, "lookup_prefix_hit", None)
                if callable(hit_fn):
                    adapter = backend
                    break
                pfn = getattr(backend, "prefix_contains_tokens", None)
                if callable(pfn) and adapter is None:
                    adapter = backend
        if callable(hit_fn):
            prefix, hit = hit_fn(tokens, len(keys))
            prefix = int(prefix or 0)
        elif adapter is not None:
            pfn = getattr(adapter, "prefix_contains_tokens", None)
            if callable(pfn):
                prefix = int(pfn(tokens, len(keys)) or 0)
        if prefix <= 0:
            logger.info(
                "GDS prefix miss keys=%s prefix=%s slot_mapping=%s",
                len(keys),
                prefix,
                kwargs.get("slot_mapping") is not None,
            )
            return chunks, 0, leases
        connector = getattr(self, "gpu_connector", None)
        slot_mapping = kwargs.get("slot_mapping")
        page_direct = (
            adapter is not None
            and hit is not None
            and connector is not None
            and callable(getattr(connector, "page_direct_enabled", None))
            and connector.page_direct_enabled()
            and self._pd_profile is not None
            and (
                callable(getattr(adapter, "kvb_read_to_gpu_submit", None))
                or callable(getattr(adapter, "kvb_read_to_gpu", None))
            )
        )
        if page_direct and slot_mapping is not None:
            # Lookup only; EngineCore polls DMA via PendingLoad / try_issue_h2d.
            pending_meta = kwargs.get("_pending_load")
            if pending_meta is not None:
                pending_meta.pd_hit = hit
                pending_meta.pd_prefix = prefix
                pending_meta.pd_spans = list(spans[:prefix])
                pending_meta.pd_adapter = adapter
            return chunks, prefix, leases
        logger.info(
            "GDS skip host prefix GET prefix=%s slot_mapping=%s page_direct=%s "
            "(lookup ACK already sent)",
            prefix,
            slot_mapping is not None,
            page_direct,
        )
        return chunks, 0, leases

    def submit_retrieve_gather(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        generation: int = 0,
        block_ids: tuple[int, ...] = (),
        **kwargs,
    ) -> PendingLoad:
        """Start CPU/DRAM gather without H2D.

        GDS page-direct submits native reads and returns immediately; scatter
        completes on ``try_issue_h2d`` / ``_poll_pending_loads``.
        """
        kwargs.pop("sync", None)
        req_id = str(self._get_req_id(kwargs) or "")
        ret_mask = torch.zeros(len(tokens), dtype=torch.bool)
        keepalive = dict(kwargs)
        keepalive["_gather_tokens"] = tokens
        keepalive["_gather_mask"] = mask

        def _work():
            return self._gather_pinned_chunks(tokens, mask, ret_mask, **kwargs)

        pending = PendingLoad(
            req_id=req_id,
            chunks=[],
            generation=int(generation),
            block_ids=tuple(block_ids),
            kwargs_keepalive=keepalive,
            xfer_state=XFER_GATHERING,
        )
        gds_page_direct = (
            str(getattr(self.config, "kv_io_mode", "dram")) == "gds"
            and kwargs.get("slot_mapping") is not None
        )
        if gds_page_direct:
            pending.ret_mask = ret_mask
            kwargs["_pending_load"] = pending
            try:
                self._gather_pinned_chunks(tokens, mask, ret_mask, **kwargs)
                if int(pending.pd_prefix or 0) > 0:
                    self._submit_page_direct_reads(
                        pending, tokens, mask, **kwargs
                    )
                else:
                    pending.error = RuntimeError(
                        "GDS gather produced no GPU pages "
                        "(page-direct miss or missing slot_mapping)"
                    )
                    pending.xfer_state = XFER_CANCELLING
            except Exception as exc:
                pending.error = exc
                pending.xfer_state = XFER_CANCELLING
                self._release_gds_leases(pending)
            pending.gather_future = None
            return pending
        loop = getattr(self.storage_manager, "loop", None)
        if loop is not None and getattr(loop, "is_running", lambda: False)():

            async def _coro():
                return await loop.run_in_executor(None, _work)

            pending.gather_future = asyncio.run_coroutine_threadsafe(_coro(), loop)
        else:
            if not hasattr(self, "_gather_pool"):
                self._gather_pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="lmcache-gather"
                )
            pending.gather_future = self._gather_pool.submit(_work)
        return pending

    def _resubmit_gather(self, pending: PendingLoad) -> None:
        """Re-run host gather when objects were not visible yet."""
        tokens = (pending.kwargs_keepalive or {}).get("_gather_tokens")
        mask = (pending.kwargs_keepalive or {}).get("_gather_mask")
        if tokens is None:
            return
        kwargs = {
            k: v
            for k, v in (pending.kwargs_keepalive or {}).items()
            if not str(k).startswith("_gather_")
        }
        ret_mask = torch.zeros(len(tokens), dtype=torch.bool)

        def _work():
            return self._gather_pinned_chunks(tokens, mask, ret_mask, **kwargs)

        pending.xfer_state = XFER_GATHERING
        pending.chunks = []
        loop = getattr(self.storage_manager, "loop", None)
        if loop is not None and getattr(loop, "is_running", lambda: False)():

            async def _coro():
                return await loop.run_in_executor(None, _work)

            pending.gather_future = asyncio.run_coroutine_threadsafe(_coro(), loop)
        else:
            if not hasattr(self, "_gather_pool"):
                self._gather_pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="lmcache-gather"
                )
            pending.gather_future = self._gather_pool.submit(_work)

    def _gather_pinned_chunks(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ) -> tuple[list, torch.Tensor, List[Any]]:
        """Prefetch gather from already-pinned L1 objects. Never waits on LOADING."""
        leases: List[Any] = []
        if str(getattr(self.config, "kv_io_mode", "dram")) == "gds":
            chunks, _sz, leases = self._gds_process_tokens_internal(
                tokens, mask, ret_mask, **kwargs
            )
        else:
            # Do not call _async_process_tokens_internal: that path polls
            # EventType.LOADING up to ~30s and writes _last_cancelled_load.
            chunks, _sz = self._process_tokens_internal(
                tokens, mask, ret_mask, **kwargs
            )
        return list(chunks), ret_mask, leases

    def try_issue_h2d(self, pending: PendingLoad) -> str:
        """Issue H2D once gather is ready.

        Returns ``pending_gather``, ``issued``, ``empty``, or ``error``.
        GDS page-direct scatter leaves ``chunks`` empty and sets ``ret_mask``;
        that is treated as already on GPU (``issued``), not as a host retry.
        A GDS gather with no ``ret_mask`` bits fails closed (``error``).
        """
        if pending.error is not None:
            pending.xfer_state = XFER_CANCELLING
            return "error"
        if pending.gds_read_fut is not None or int(pending.pd_prefix or 0) > 0:
            if pending.h2d_issued:
                return "issued"
            return self._try_complete_page_direct(pending)
        if pending.completed or pending.xfer_state in (
            XFER_COMPLETED,
            XFER_RELEASED,
        ):
            return "empty"
        if pending.h2d_issued or pending.xfer_state in (
            XFER_H2D_IN_FLIGHT,
            XFER_GDS_IN_FLIGHT,
            XFER_GPU_SCATTER_IN_FLIGHT,
        ):
            return "issued"
        fut = pending.gather_future
        if fut is not None:
            if not fut.done():
                pending.xfer_state = XFER_GATHERING
                return "pending_gather"
            try:
                result = fut.result(timeout=0)
            except Exception as exc:
                pending.error = exc
                pending.xfer_state = XFER_CANCELLING
                return "error"
            if isinstance(result, tuple) and len(result) == 3:
                chunks, ret_mask, leases = result
            else:
                chunks, ret_mask = result
                leases = []
            pending.chunks = chunks
            pending.ret_mask = ret_mask
            pending.gds_leases = list(leases or [])
            pending.gather_future = None
            if (
                str(getattr(self.config, "kv_io_mode", "dram")) == "gds"
                and pending.gds_leases
            ):
                pending.gds_state = GDS_DMA_DONE
                pending.t_dma_done = time.perf_counter()
                pending.xfer_state = XFER_GDS_IN_FLIGHT
            else:
                pending.xfer_state = XFER_L1_READY
        if not pending.chunks:
            kv_io = str(getattr(self.config, "kv_io_mode", "dram"))
            ret_mask = pending.ret_mask
            if (
                kv_io == "gds"
                and ret_mask is not None
                and bool(ret_mask.any())
            ):
                # Page-direct already scattered into HBM; no host chunks to H2D.
                pending.h2d_issued = True
                pending.xfer_state = XFER_GPU_SCATTER_IN_FLIGHT
                pending.gds_state = GDS_HBM_ASSEMBLED
                pending.event = None
                pending.t_h2d_issue = time.perf_counter()
                return "issued"
            if kv_io == "gds":
                pending.error = RuntimeError(
                    "GDS gather produced no GPU pages "
                    "(page-direct miss or missing slot_mapping)"
                )
                pending.xfer_state = XFER_CANCELLING
                return "error"
            retries = int(getattr(pending, "gather_retries", 0) or 0)
            tokens = (pending.kwargs_keepalive or {}).get("_gather_tokens")
            if retries < 16 and tokens is not None:
                pending.gather_retries = retries + 1
                self._resubmit_gather(pending)
                return "pending_gather"
            return "empty"
        if self.gpu_connector is None:
            pending.error = RuntimeError("gpu_connector required for H2D")
            return "error"
        from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface

        impl = type(self.gpu_connector).batched_to_gpu_async
        if impl is GPUConnectorInterface.batched_to_gpu_async:
            pending.error = RuntimeError("connector lacks real batched_to_gpu_async")
            return "error"
        _, memory_objs, starts, ends = zip(*pending.chunks, strict=False)
        kwargs = {
            k: v
            for k, v in pending.kwargs_keepalive.items()
            if not str(k).startswith("_gather_")
        }
        pending.event = self.gpu_connector.batched_to_gpu_async(
            list(memory_objs), list(starts), list(ends), **kwargs
        )
        pending.h2d_issued = True
        pending.t_h2d_issue = time.perf_counter()
        if str(getattr(self.config, "kv_io_mode", "dram")) == "gds":
            pending.xfer_state = XFER_GPU_SCATTER_IN_FLIGHT
            pending.gds_state = GDS_HBM_ASSEMBLED
            if pending.t_dma_done > 0:
                metrics = getattr(self, "_gds_metrics", None)
                if metrics is not None:
                    metrics.note_dma_hbm(
                        int((pending.t_h2d_issue - pending.t_dma_done) * 1e6)
                    )
        else:
            pending.xfer_state = XFER_H2D_IN_FLIGHT
        pending.kwargs_keepalive["memory_objs_for_togpu"] = list(memory_objs)
        return "issued"

    def _gds_retrieve_still_busy(self, pending: PendingLoad) -> bool:
        """True while a submitted GDS page-direct read or header copy is in flight."""
        fut = getattr(pending, "gds_read_fut", None)
        if fut is not None:
            poll = getattr(
                getattr(pending, "pd_adapter", None), "kvb_read_to_gpu_poll", None
            )
            if callable(poll):
                done, _rc = poll(fut)
                if not done:
                    return True
            else:
                done_fn = getattr(fut, "done", None)
                if callable(done_fn) and not done_fn():
                    return True
        hdr = getattr(pending, "pd_header_event", None)
        if hdr is not None and not self._event_is_ready(hdr):
            return True
        return False

    def _gds_lease_memory_ids(self, pending: PendingLoad) -> set[int]:
        ids: set[int] = set()
        for lease in pending.gds_leases or []:
            mem = getattr(lease, "memory_obj", None)
            if mem is not None:
                ids.add(id(mem))
        return ids

    def complete_load(self, pending: PendingLoad) -> None:
        """Unpin MemoryObjs after the H2D event is ready."""
        if pending.completed or pending.xfer_state in (
            XFER_COMPLETED,
            XFER_RELEASED,
        ):
            return
        pending.completed = True
        pending.xfer_state = XFER_COMPLETED
        if pending.gds_leases:
            pending.gds_state = GDS_COMPLETE
        if self.save_only_first_rank and self.metadata.is_first_rank():
            self._leader_gpu_substitute_objs = []
        mask = pending.ret_mask if pending.ret_mask is not None else torch.zeros(0)
        gds_obj_ids = self._gds_lease_memory_ids(pending)
        dram_chunks = [
            chunk
            for chunk in pending.chunks
            if id(chunk[1]) not in gds_obj_ids
        ]
        if pending.gds_leases:
            self._release_gds_leases(pending)
        if dram_chunks:
            self._release_retrieve_chunks(
                dram_chunks, pending.req_id, pending.retrieve_stats, mask
            )
        elif pending.retrieve_stats is not None:
            retrieved_tokens = torch.sum(mask)
            self.stats_monitor.on_retrieve_finished(
                pending.retrieve_stats,
                retrieved_tokens,
            )
        pending.chunks = []
        pending.kwargs_keepalive = {}
        pending.xfer_state = XFER_RELEASED

    def abandon_load(self, pending: PendingLoad, wait_event: bool = True) -> None:
        """Release MemoryObjs only after the H2D event is ready; never synchronize."""
        if pending.completed or pending.xfer_state in (
            XFER_COMPLETED,
            XFER_RELEASED,
        ):
            return
        pending.xfer_state = XFER_CANCELLING
        fut = pending.gather_future
        if fut is not None and not fut.done():
            self.abandoned_loads.append(pending)
            return
        if fut is not None and fut.done():
            try:
                result = fut.result(timeout=0)
                if isinstance(result, tuple) and len(result) == 3:
                    chunks, ret_mask, leases = result
                    pending.gds_leases = list(leases or [])
                else:
                    chunks, ret_mask = result
                pending.chunks = chunks
                pending.ret_mask = ret_mask
            except Exception:
                pass
            pending.gather_future = None
        if self._gds_retrieve_still_busy(pending):
            self.abandoned_loads.append(pending)
            return
        if not self._event_is_ready(pending.event):
            self.abandoned_loads.append(pending)
            return
        self.complete_load(pending)

    def poll_abandoned_loads(self) -> List[PendingLoad]:
        still: List[PendingLoad] = []
        done: List[PendingLoad] = []
        for pending in self.abandoned_loads:
            if pending.completed:
                done.append(pending)
                continue
            fut = pending.gather_future
            if fut is not None and not fut.done():
                still.append(pending)
                continue
            if self._gds_retrieve_still_busy(pending):
                still.append(pending)
                continue
            if self._event_is_ready(pending.event):
                self.complete_load(pending)
                done.append(pending)
            else:
                still.append(pending)
        self.abandoned_loads = still
        return done

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[Optional[torch.Tensor], None, None]:
        """
        Retrieve the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields Optional[torch.Tensor]. The tensor will
            be the boolean mask indicating which tokens are retrieved and will
            only be returned in the last iteration. In the first iteration,
            the generator retrieve the memory objects of the first layer from
            the storage backends. In the next iterations, it moves the KV cache
            of layer i from the memory objects (on CPU) to GPU and retrieves
            the memory objects of layer i+1 from the storage backends. In the
            last iteration, it moves the memory objects of the last layer to
            the GPU.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping retrieve_layer operation")
            yield torch.zeros(len(tokens), dtype=torch.bool)
            return

        assert self.storage_manager is not None
        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve_layer operation"
        )

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        starts = []
        ends = []
        keys = []

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        location = None
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)

            keys_multi_layer = key.split_layers(self.num_layers)

            # NOTE: Only check the first layer
            if current_location := self.storage_manager.contains(
                keys_multi_layer[0], self.retrieve_locations
            ):
                if location is None:
                    location = current_location
                else:
                    # TODO(Jiayi): Support multi-location retrieval in the future
                    assert location == current_location, (
                        "All retrieved keys should be from the same location "
                        "when use layerwise retrieval."
                        "Please support multi-location retrieval in the future."
                    )
            else:
                break

            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)

            ret_mask[start:end] = True

        if keys:
            # Transpose the keys into layer major format
            keys_layer_major = [list(row) for row in zip(*keys, strict=False)]

            get_generator = self.storage_manager.layerwise_batched_get(
                keys_layer_major,
                location=location,
            )

            assert_layerwise_gpu_connector(self.gpu_connector)

            mem_obj_consumer = self.gpu_connector.batched_to_gpu(starts, ends, **kwargs)
            next(mem_obj_consumer)

            to_count_down = []
            for layer_id in range(self.num_layers):
                task = next(get_generator)

                assert task is not None

                if layer_id == 0:
                    # NOTE(Yuwei): For sglang integration we need to provide retrieved
                    # tokens number in the first layer loading since there is no lookup
                    yield torch.sum(ret_mask)
                else:
                    yield None

                mem_objs_layer = task.result()
                mem_obj_consumer.send(mem_objs_layer)
                to_count_down.extend(mem_objs_layer)

            for mem_obj in to_count_down:
                mem_obj.ref_count_down()
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield None

        yield None

        # synchronize the last layer
        next(mem_obj_consumer)

        # Unpin any disk-loaded staging objects now that the device-side sync
        # has been enqueued (mem_obj_consumer advanced past its sync point).
        # Without this, pin_count stays at 1 forever and the CPU staging pool
        # fills up, causing the next retrieve to deadlock inside allocate().
        for mem_obj in to_count_down:
            if mem_obj.is_pinned:
                mem_obj.unpin()

        retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
        if not self._is_passive():
            logger.info(
                "[req_id=%s] Retrieved %d out of %d out of total %d tokens",
                req_id,
                retrieved_tokens,
                num_required_tokens,
                len(tokens),
            )

        yield ret_mask

    @_lmcache_nvtx_annotate
    def lookup(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        lookup_id: Optional[str] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.

        :param Optional[Union[torch.Tensor, List[int]]] tokens: the input tokens,
        with shape [seq_len]

        :param Optional[List[int]] hashes: the input hashes, with length [num_chunks]
        :param Optional[List[int]] offsets: the offsets of each chunk,
        with length [num_chunks]

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of
        ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param Optional[str] lookup_id: The lookup ID to
            associate with the lookup. When pin is true, this argument is
            required to be not None.

        :param bool pin: If True, pin the KV cache in the storage.

        :param Optional[dict] request_configs: the configs of the request.

        :return: An int indicating how many prefix tokens exist inside LMCache.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping lookup operation")
            return 0

        assert self.storage_manager is not None

        if tokens is not None:
            lookup_stats = self.stats_monitor.on_lookup_request(len(tokens))
        else:
            assert offsets is not None
            assert hashes is not None
            lookup_stats = self.stats_monitor.on_lookup_request(sum(offsets))

        if search_range is None:
            search_range = self.retrieve_locations

        res = 0
        try:
            chunk_info_iterator = self.token_database.process_tokens(
                tokens=tokens,
                hashes=hashes,
                offsets=offsets,
                request_configs=request_configs,
            )

            # TODO: support batched_contains when layerwise is enabled
            if self.use_layerwise:
                for start, end, key in chunk_info_iterator:
                    assert isinstance(key, CacheEngineKey)

                    # TODO(Jiayi): Optimize by checking only the existence of the key
                    # of one layer
                    key_all_layers = key.split_layers(self.num_layers)

                    hit_chunks, block_mapping = self.storage_manager.batched_contains(
                        key_all_layers,  # type: ignore
                        search_range,
                        pin,
                        token_ids=tokens,
                    )
                    # Only all layers are hit and hit in one location,
                    # we consider this key as a hit
                    if hit_chunks == self.num_layers and len(block_mapping) == 1:
                        if pin:
                            assert lookup_id is not None, (
                                "lookup_id is required when pin is True"
                            )
                            location = next(iter(block_mapping.keys()))
                            self.lookup_pins[lookup_id][location].extend(key_all_layers)
                        res = end
                        continue
                    return res
            else:
                chunk_info_list = []
                keys = []
                for chunk_info in chunk_info_iterator:
                    assert isinstance(chunk_info[2], CacheEngineKey)
                    start, end, _ = chunk_info
                    chunk_info_list.append(chunk_info)
                    keys.append(chunk_info[2])
                hit_chunks, block_mapping = self.storage_manager.batched_contains(
                    keys, search_range, pin, token_ids=tokens
                )
                if pin and block_mapping:
                    assert lookup_id is not None, (
                        "lookup_id is required when pin is True"
                    )
                    self.lookup_pins[lookup_id] = block_mapping
                for idx, (start, end, key) in enumerate(chunk_info_list):
                    if idx < hit_chunks:
                        res = end
                        continue
                    return res

            # all tokens where found, return the maximal end
            return res
        finally:
            self.stats_monitor.on_lookup_finished(lookup_stats, res)
            # vllm lookup sets pin to True
            if pin:
                # touch_cache is tightly coupled with batched_contains
                self.storage_manager.touch_cache()

    @_lmcache_nvtx_annotate
    def move(
        self,
        tokens: Union[torch.Tensor, List[int]],
        old_position: str,
        new_position: tuple[str, str],
        event_id: str,
        do_copy: bool = True,
    ) -> int:
        """
        Perform cross-node move of the KV cache.
        """
        assert self.storage_manager is not None

        num_tokens = self.lookup(
            tokens,
            search_range=[old_position],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("Move is not performed as there are no tokens to move.")
            return 0

        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[old_position]

        memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=old_position,
        )
        assert None not in memory_objs, "Failed to get memory objects to move"
        logger.debug(
            f"Trying to send {len(memory_objs)} memory objects to {new_position}"
        )

        # TODO: reduce loops
        token_dim = memory_objs[0].meta.fmt.token_dim()  # type: ignore
        offsets = [m.meta.shape[token_dim] for m in memory_objs]  # type: ignore

        transfer_spec = {
            "target_peer_init_url": new_position[0],
            "offsets": offsets,
        }

        logger.info(self.storage_manager.storage_backends)
        p2p_backend = self.storage_manager.storage_backends["P2PBackend"]

        future = asyncio.run_coroutine_threadsafe(
            p2p_backend.async_batched_submit_put_task(
                keys,
                memory_objs,  # type: ignore
                transfer_spec=transfer_spec,
            ),
            self.storage_manager.loop,
        )

        future.result()

        if not do_copy:
            self.storage_manager.batched_remove(keys, locations=[old_position])

        logger.debug(
            "Moving %d token from %s to %s", num_tokens, old_position, new_position
        )
        return num_tokens

    # TODO(Jiayi): Add layerwise support.
    @_lmcache_nvtx_annotate
    def async_lookup_and_prefetch(
        self,
        lookup_id: str,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
    ) -> None:
        """
        An async version of lookup + prefetch.

        There are three categories of backends:
        (1) sync lookup + sync retrieval (e.g., cpu)
        (2) sync lookup + async retrieval (e.g., disk)
        (3) async lookup + async retrieval (e.g., p2p)
        """
        if self.storage_manager is None:
            raise RuntimeError("storage_manager unavailable (init failed)")

        keys: list[CacheEngineKey] = []
        cum_chunk_lengths = [0]

        if search_range is None:
            search_range = self.retrieve_locations

        # When layerwise is enabled, store_layer writes per-layer keys
        # (LayerCacheEngineKey, chunk_hash + layer_id). Async lookup must
        # split each chunk key into num_layers per-layer keys so the
        # storage backend hot_cache lookups match the same key type.
        keys_per_chunk = self.num_layers if self.use_layerwise else 1

        # TODO(Jiayi): make token database able to return list.
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            hashes=hashes,
            offsets=offsets,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            if self.use_layerwise:
                keys.extend(key.split_layers(self.num_layers))
            else:
                keys.append(key)
            cum_chunk_lengths.append(end)

        fut = asyncio.run_coroutine_threadsafe(
            self.storage_manager.async_lookup_and_prefetch(
                lookup_id,
                keys,
                cum_chunk_lengths,
                search_range,
                pin,
                keys_per_chunk=keys_per_chunk,
                token_ids=tokens,
            ),
            self.storage_manager.loop,
        )

        def _on_prefetch_done(done_fut, req=lookup_id):
            cb = getattr(
                self.storage_manager, "_on_lookup_prefetch_future_done", None
            )
            if callable(cb):
                cb(req, done_fut)

        fut.add_done_callback(_on_prefetch_done)

    def cleanup_memory_objs(self, lookup_id: str) -> None:
        """
        Cleanup memory objects allocated during prefetch for an aborted lookup.

        DONE: claim and release on this thread (lookup_unpin hot path).
        ONGOING: record abort; prefetch_all_done_callback releases later.
        NOT_FOUND: no-op (zero-hit or already reclaimed).
        """
        try:
            if not self.event_manager.mark_lookup_aborted(
                EventType.LOADING, lookup_id
            ):
                return
            future = self.event_manager.claim_done_event(
                EventType.LOADING, lookup_id, only_if_aborted=True
            )
            if future is None:
                return
            release_prefetch_result(future.result())
        except Exception as e:
            logger.error(
                "Error during cleanup_memory_objs for lookup_id=%s: %s",
                lookup_id,
                e,
            )

    # TODO(Jiayi): Need to handle the case where `tokens=None`.
    # In this case, we compress all tokens.
    # TODO(Jiayi): support other compression methods.
    @_lmcache_nvtx_annotate
    def compress(
        self,
        tokens: Union[torch.Tensor, List[int]],
        method: str,
        location: str,
        event_id: str,
    ) -> int:
        assert self.storage_manager is not None
        if method not in ["cachegen"]:
            logger.warning("Unsupported compression method: %s.", method)
            return 0

        # First Party
        from lmcache.v1.storage_backend.naive_serde import CreateSerde

        serializer, _ = CreateSerde(method, self.metadata, self.config)

        num_tokens = self.lookup(
            tokens,
            search_range=[location],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("Move is not performed as there are no tokens to move.")
            return 0

        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[location]

        memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=location,
        )
        assert None not in memory_objs, (
            "LMCacheEngine.compress: Failed to get memory objects to compress"
        )

        compressed_memory_objs = []
        for memory_obj in memory_objs:
            assert memory_obj is not None
            compressed_memory_obj = serializer.serialize(memory_obj)
            if memory_obj.is_pinned:
                memory_obj.unpin()
            compressed_memory_objs.append(compressed_memory_obj)

        self.storage_manager.batched_remove(keys, locations=[location])

        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=compressed_memory_objs,
            location=location,
        )

        return num_tokens

    @_lmcache_nvtx_annotate
    def decompress(
        self,
        tokens: Union[torch.Tensor, List[int]],
        method: str,
        location: str,
        event_id: str,
    ) -> int:
        assert self.storage_manager is not None
        if method not in ["cachegen"]:
            logger.warning("Unsupported decompression method: %s.", method)
            return 0

        # First Party
        from lmcache.v1.storage_backend.naive_serde import CreateSerde

        _, deserializer = CreateSerde(method, self.metadata, self.config)

        num_tokens = self.lookup(
            tokens,
            search_range=[location],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("there are no tokens to decompress.")
            return 0

        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[location]

        compressed_memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=location,
        )

        assert None not in compressed_memory_objs, (
            "LMCacheEngine.compress: Failed to get compressed "
            "memory objects to decompress"
        )

        memory_objs = []
        for compressed_memory_obj in compressed_memory_objs:
            assert compressed_memory_obj is not None
            memory_obj = deserializer.deserialize(compressed_memory_obj)
            if compressed_memory_obj.is_pinned:
                compressed_memory_obj.unpin()
            memory_objs.append(memory_obj)

        self.storage_manager.batched_remove(keys, locations=[location])

        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=memory_objs,
            location=location,
        )

        return num_tokens

    @_lmcache_nvtx_annotate
    def lookup_unpin(self, lookup_id: str) -> None:
        if lookup_id in self.lookup_pins:
            assert self.storage_manager is not None
            for location, keys in self.lookup_pins.pop(lookup_id).items():
                self.storage_manager.batched_unpin(keys, [location])

        elif (
            self.async_loading is not None
            and self.event_manager.get_event_status(EventType.LOADING, lookup_id)
            != EventStatus.NOT_FOUND
        ):
            self.cleanup_memory_objs(lookup_id)

    @_lmcache_nvtx_annotate
    def clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
        request_configs: Optional[dict] = None,
    ) -> int:
        # TODO: need to clear by request_configs
        if self.save_only_first_rank:
            if self.metadata.is_first_rank():
                num_removed = self._clear(tokens, locations, request_configs)
                return num_removed
            else:
                return 0
        return self._clear(tokens, locations, request_configs)

    @_lmcache_nvtx_annotate
    def get_kv_events(self) -> Iterable[CacheStoreEvent]:
        if self.kv_events_enabled and (events := self.kv_events):
            self.kv_events = []
            return events
        return []

    def _clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
        request_configs: Optional[dict] = None,
    ) -> int:
        assert self.storage_manager is not None
        assert isinstance(self.storage_manager, StorageManager)
        # Clear all caches if tokens is None
        if tokens is None or len(tokens) == 0:
            num_cleared = self.storage_manager.clear(locations)
            return num_cleared

        num_removed = 0
        # Only remove the caches for the given tokens
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, request_configs=request_configs
        ):
            assert isinstance(key, CacheEngineKey)
            removed = self.storage_manager.remove(key, locations)
            num_removed += removed
        return num_removed

    @_lmcache_nvtx_annotate
    def health(
        self,
    ) -> int:
        """
        Check the health of the cache engine.
        return: 0 if healthy, otherwise the error code
        """
        assert self.storage_manager is not None
        return 0 if self.storage_manager.memcheck() else -1

    def close(self) -> None:
        """Close the cache engine and free all the resources"""
        logger.info("Closing LMCacheEngine...")

        if self.hidden_state_store is not None:
            try:
                logger.info("Closing hidden_state_store...")
                self.hidden_state_store.close()
            except Exception as e:
                logger.error(f"Error closing hidden_state_store: {e}")

        if self.lmcache_worker is not None:
            try:
                logger.info("Closing lmcache_worker...")
                self.lmcache_worker.close()
                logger.info("lmcache_worker closed successfully")
            except Exception as e:
                logger.error("Error closing lmcache_worker: %s", e)

        try:
            logger.info("Closing storage_manager...")
            if self.storage_manager is not None:
                self.storage_manager.close()
            logger.info("storage_manager closed successfully")
        except Exception as e:
            logger.error("Error closing storage_manager: %s", e)

        logger.info("LMCacheEngine closed.")

    def _async_process_tokens_internal(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ) -> ProcessTokensInternalResult:
        """
        This function is used to get the memory objects from the event manager.

        Args:
            tokens: Input tokens to process
            mask: Mask indicating valid token positions
            ret_mask: Output mask updated with cache hit positions
            **kwargs: Additional keyword arguments
        """
        assert "req_id" in kwargs, "req_id is required for async loading"
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        tot_kv_size = 0
        chunks: List[ProcessedChunk] = []
        req_id = kwargs["req_id"]
        extra = getattr(self.config, "extra_config", None) or {}
        timeout_ms = extra.get(
            "kv.dram_preload_timeout_ms",
            getattr(self.config, "dram_preload_timeout_ms", 30000),
        )
        lookup_ms = getattr(self.config, "lookup_timeout_ms", 0) or 0
        try:
            timeout_s = max(1.0, float(timeout_ms) / 1000.0, float(lookup_ms) / 1000.0)
        except (TypeError, ValueError):
            timeout_s = 30.0
        deadline = time.monotonic() + timeout_s
        future = None
        while time.monotonic() < deadline:
            status = self.event_manager.get_event_status(EventType.LOADING, req_id)
            if status == EventStatus.DONE:
                future = self.event_manager.try_get_done_future(
                    EventType.LOADING, req_id
                )
                if future is not None:
                    break
            time.sleep(0.01)
        if future is None:
            logger.error(
                "dram_op request=%s phase=loading_watchdog timeout_s=%.1f status=%s",
                req_id,
                timeout_s,
                self.event_manager.get_event_status(EventType.LOADING, req_id),
            )
            self._l1_future_not_ready = int(getattr(self, "_l1_future_not_ready", 0)) + 1
            self._last_cancelled_load = str(req_id)
            return [], 0
        # Future is already complete: result() must not block.
        try:
            keyed_memory_objs = future.result()
            memory_obj_map: dict[CacheEngineKey, MemoryObj] = {}
        except Exception as e:
            logger.error("Error popping event for request %s: %s", req_id, e)
            self._l1_future_not_ready = int(getattr(self, "_l1_future_not_ready", 0)) + 1
            self._last_cancelled_load = str(req_id)
            return [], 0

        for backend_results in keyed_memory_objs:
            for key, memory_obj in backend_results:
                memory_obj_map[key] = memory_obj

        # TODO(Jiayi): hashing inside `process_tokens` can be skipped.
        used_keys: set[CacheEngineKey] = set()
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            memory_obj = memory_obj_map.get(key)
            if memory_obj is None:
                # returned chunks are expected to be contiguous.
                # break at the first missing chunk.
                break
            chunks.append((key, memory_obj, start, end))
            tot_kv_size += memory_obj.get_size()
            ret_mask[start:end] = True
            used_keys.add(key)

        # NOTE: free the memory objects that are not hit.
        for key, mem_obj in memory_obj_map.items():
            if key not in used_keys:
                mem_obj.ref_count_down()

        return chunks, tot_kv_size

    def _process_tokens_internal(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ) -> ProcessTokensInternalResult:
        """Process tokens and populate the reordered lists.

        This function is used to process tokens and populate the reordered lists.

        Args:
            tokens: Input tokens to process
            mask: Mask indicating valid token positions
            ret_mask: Output mask updated with cache hit positions
            **kwargs: Additional keyword arguments
        """
        assert self.storage_manager is not None

        tot_kv_size = 0
        reordered_chunks: List[ProcessedChunk] = []
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        chunk_infos = []
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            chunk_infos.append((key, start, end))

        # block_mapping: location -> [(CacheEngineKey, start, end)]
        if (
            "req_id" in kwargs
            and kwargs["req_id"] in self.lookup_pins
            and len(self.lookup_pins[kwargs["req_id"]]) == 1
        ):
            location = next(iter(self.lookup_pins[kwargs["req_id"]].keys()))
            block_mapping = {location: chunk_infos}
        else:
            block_mapping = self.storage_manager.get_block_mapping(
                chunk_infos, token_ids=tokens
            )

        last_failed_block_start = None
        for location, blocks in block_mapping.items():
            keys = [key for key, _, _ in blocks]
            memory_objs = self.storage_manager.batched_get(
                keys=keys,
                location=location,
            )

            used_keys: set[CacheEngineKey] = set()
            for (key, start, end), memory_obj in zip(blocks, memory_objs, strict=False):
                if memory_obj is None:
                    logger.warning(
                        "The cache block is in the storage, but it can't be retrieved"
                    )
                    if (
                        last_failed_block_start is None
                        # The minimum value should be taken here to ensure that
                        # the prefix keys are all consecutive successful.
                        or last_failed_block_start > start
                    ):
                        last_failed_block_start = start
                    break
                reordered_chunks.append((key, memory_obj, start, end))
                tot_kv_size += memory_obj.get_size()
                ret_mask[start:end] = True
                used_keys.add(key)

            for (key, _, _), memory_obj in zip(blocks, memory_objs, strict=False):
                if memory_obj is not None and key not in used_keys:
                    logger.debug(
                        "ref_count_down for %s of %s as the previous key failed",
                        key,
                        location,
                    )
                    memory_obj.ref_count_down()

        if last_failed_block_start is not None:
            ret_mask[last_failed_block_start:] = False

            kept_chunks: List[ProcessedChunk] = []
            for key, memory_obj, start, end in reordered_chunks:
                if end <= last_failed_block_start:
                    kept_chunks.append((key, memory_obj, start, end))
                else:
                    tot_kv_size -= memory_obj.get_size()
                    # This chunk will not be used. If the engine is configured
                    # to remove-after-retrieve, the caller would normally call
                    # remove (which frees the block), but since we are dropping
                    # these chunks here, we must free them ourselves to avoid
                    # leaking PD buffer pool memory.
                    if self.remove_after_retrieve:
                        assert self.storage_manager is not None
                        self.storage_manager.remove(key, self.retrieve_locations)
                        # Sync PDBackend.remove() does NOT call ref_count_down()
                        # internally (unlike async PD and other backends), so we
                        # must call it manually. See pd_backend.py line 605.
                        if self._is_sync_pd_backend():
                            memory_obj.ref_count_down()
                    else:
                        memory_obj.ref_count_down()
            reordered_chunks = kept_chunks
        return reordered_chunks, tot_kv_size

    def _need_retrieve_broadcast(self) -> bool:
        """MLA TP retrieve must NCCL-broadcast when there is a peer rank.

        ``world_size`` is this Engine's tensor-parallel size, not the node
        GPU count. DP replicas with TP=1 skip the substitute ``.to(cuda)``.
        """
        ws = int(getattr(self.metadata, "world_size", 1) or 1)
        return bool(self.save_only_first_rank) and ws > 1

    def _to_gpu_one_chunk(
        self,
        memory_obj: MemoryObj,
        start: int,
        end: int,
        *,
        sync: bool,
        **kwargs,
    ) -> Any:
        assert self.gpu_connector is not None
        if sync:
            self.gpu_connector.batched_to_gpu(
                [memory_obj], [start], [end], **kwargs
            )
            return None
        batched_async = getattr(self.gpu_connector, "batched_to_gpu_async", None)
        if callable(batched_async):
            return batched_async([memory_obj], [start], [end], **kwargs)
        self.gpu_connector.batched_to_gpu([memory_obj], [start], [end], **kwargs)
        return ReadyStoreEvent()

    def _broadcast_or_receive_memory_objs(
        self,
        reordered_chunks,
        ret_mask,
        *,
        sync: bool = True,
        to_gpu: bool = False,
        **kwargs,
    ):
        """Broadcast or receive MLA KV, optionally copying each chunk to paged KV.

        Leader broadcasts; other ranks reconstruct. With ``to_gpu``, each chunk
        is copied into paged KV before the next is staged so TP peak HBM is
        about one chunk. Intermediate copies wait; only the last chunk may stay
        async (keepalive). CUDA headroom must not skip this broadcast.

        Returns:
            (event, keepalive_gpu_objs) when to_gpu else (None, []).
        """
        last_event = None
        keepalive: List[MemoryObj] = []
        if self.metadata.is_first_rank():
            chunk_count = len(reordered_chunks)
            self.broadcast_object_fn(chunk_count, self.metadata.first_rank)

            self._leader_gpu_substitute_objs = []

            for idx, (_key, memory_obj, start, end) in enumerate(reordered_chunks):
                metadata_dict = memory_obj.metadata.to_dict()
                combined_metadata = (start, end, metadata_dict)
                self.broadcast_object_fn(combined_metadata, self.metadata.first_rank)

                raw_tensor = memory_obj.raw_tensor
                assert raw_tensor is not None
                tensor_to_broadcast = raw_tensor.to(
                    f"{torch_device_type}:{self.metadata.worker_id}"
                )
                self.broadcast_fn(tensor_to_broadcast, self.metadata.first_rank)

                gpu_mo = TensorMemoryObj(
                    raw_data=tensor_to_broadcast,
                    metadata=memory_obj.metadata,
                    parent_allocator=None,
                )
                is_last = idx == chunk_count - 1
                if to_gpu:
                    chunk_sync = sync or not is_last
                    last_event = self._to_gpu_one_chunk(
                        gpu_mo, start, end, sync=chunk_sync, **kwargs
                    )
                    self._leader_gpu_substitute_objs = []
                    if not chunk_sync:
                        keepalive.append(gpu_mo)
                else:
                    self._leader_gpu_substitute_objs.append(gpu_mo)
        else:
            chunk_count = self.broadcast_object_fn(None, self.metadata.first_rank)
            if chunk_count is None:
                logger.warning(
                    "rank=%d received None chunk_count", self.metadata.worker_id
                )
                return last_event, keepalive

            for idx in range(chunk_count):
                combined_metadata = self.broadcast_object_fn(
                    None, self.metadata.first_rank
                )
                if combined_metadata is None:
                    logger.warning(
                        "rank=%d received None combined_metadata",
                        self.metadata.worker_id,
                    )
                    break
                start, end, metadata_dict = combined_metadata
                ret_mask[start:end] = True

                metadata = MemoryObjMetadata.from_dict(metadata_dict)
                local_rank = self.metadata.worker_id % torch_dev.device_count()
                raw_tensor = torch.empty(
                    torch.Size([metadata.get_size()]),
                    dtype=torch.uint8,
                    device=f"{torch_device_type}:{local_rank}",
                )
                self.broadcast_fn(raw_tensor, self.metadata.first_rank)

                memory_obj = TensorMemoryObj(
                    raw_data=raw_tensor, metadata=metadata, parent_allocator=None
                )
                reordered_chunks.append((None, memory_obj, start, end))
                is_last = idx == chunk_count - 1
                if to_gpu:
                    chunk_sync = sync or not is_last
                    last_event = self._to_gpu_one_chunk(
                        memory_obj, start, end, sync=chunk_sync, **kwargs
                    )
                    if not chunk_sync:
                        keepalive.append(memory_obj)

        return last_event, keepalive

    def _is_passive(self):
        """
        A 'passive' CacheEngine means that the node itself will not store/retrieve
        the data directly, but from the "active" worker (i.e., rank 0 in MLA)
        """
        return self.save_only_first_rank and not self.metadata.is_first_rank()

    def _is_sync_pd_backend(self) -> bool:
        """Check if the PD backend is the sync variant.

        :return: True when PD is enabled and ``pd_backend_mode`` is ``"sync"``.
        :rtype: bool
        """
        return self.config.enable_pd and self.config.pd_backend_mode == "sync"

    def _get_slot_mapping_list(
        self,
        slot_mapping: Optional[Union[torch.Tensor, List[int]]],
    ) -> Optional[List[int]]:
        """
        Convert slot_mapping to list if it's a tensor, otherwise return as is.

        :param slot_mapping: The slot_mapping to convert,
            can be a torch.Tensor or List[int], or None
        :type slot_mapping: Optional[Union[torch.Tensor, List[int]]]
        :return: The slot_mapping as a List[int], or None if input is None
        :rtype: Optional[List[int]]
        """
        if slot_mapping is None:
            return None
        if isinstance(slot_mapping, torch.Tensor):
            return slot_mapping.tolist()
        # At this point, slot_mapping must be List[int]
        return slot_mapping

    def _log_kvcache_for_check(
        self,
        operation: str,
        kwargs: dict,
        token_count: int,
        require_req_id: bool = False,
    ) -> None:
        """
        Helper method to log KVCache Check information.

        This method centralizes the KVCache Check logging logic that was
        duplicated in multiple methods.

        Args:
            operation: The operation being performed (e.g., "Store", "retrieve")
            kwargs: The keyword arguments containing slot_mapping and req_id
            token_count: The number of tokens involved in the operation
            require_req_id: Whether req_id must be present (default: False)
        """
        if not self.kvcache_check_log_enabled:
            return

        slot_mapping = kwargs.get("slot_mapping")
        if slot_mapping is None:
            return

        if require_req_id:
            req_id = kwargs.get("req_id")
            if req_id is None:
                return
        else:
            req_id = kwargs.get("req_id", "unspecified")

        # Convert slot_mapping to list if it's a tensor
        slot_mapping_list = self._get_slot_mapping_list(slot_mapping)
        # slot_mapping_list should not be None when slot_mapping is not None
        assert slot_mapping_list is not None

        logger.info(
            "[KVCache Check] %s request %s, tokens=%d, slot_mapping: %s",
            operation,
            req_id,
            token_count,
            compress_slot_mapping(slot_mapping_list),
        )


class LMCacheEngineBuilder:
    _instances: Dict[str, LMCacheEngine] = {}
    _cfgs: Dict[str, LMCacheEngineConfig] = {}
    _metadatas: Dict[str, LMCacheMetadata] = {}
    _stat_loggers: Dict[str, LMCacheStatsLogger] = {}

    # TODO(Jiayi): Please remove this helper function in the future.
    # Currently, it's only used for testing.
    @staticmethod
    def _Create_memory_allocator(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        numa_mapping: Optional[NUMAMapping] = None,
    ) -> MemoryAllocatorInterface:
        # NOTE: should remove this function after fixing the unit tests:
        # raise RuntimeError("_Create_memory_allocator is deprecated!")
        extra_config = config.extra_config
        enable_nixl_storage = extra_config is not None and extra_config.get(
            "enable_nixl_storage"
        )

        if enable_nixl_storage:
            # TODO(Jiayi): weird to import from transfer utils.
            # First Party
            from lmcache.v1.transfer_channel.transfer_utils import (
                get_correct_device,
            )

            corrected_device = get_correct_device(
                config.nixl_buffer_device,
                metadata.worker_id,
            )

            buffer = torch.empty(
                config.nixl_buffer_size,
                dtype=torch.uint8,
                device=corrected_device,
            )

            if corrected_device == "cpu":
                if not current_device_spec.pin_memory(
                    buffer.data_ptr(), config.nixl_buffer_size
                ):
                    raise RuntimeError("Failed to pin NIXL CPU buffer for DMA access")
            else:
                logger.info("Setting device to %s", corrected_device)
                torch_dev.set_device(corrected_device)

            return PagedTensorMemoryAllocator(
                buffer,
                [torch.Size(metadata.kv_shape)],
                [metadata.kv_dtype],
                MemoryFormat.KV_2LTD,
            )

        if config.gds_path is not None:
            assert config.gds_buffer_size is not None
            return CuFileMemoryAllocator(config.gds_buffer_size * 1024**2)

        max_local_cpu_size = config.max_local_cpu_size
        # save_only_first_rank only works when use mla
        save_only_first_rank = (
            config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )
        if save_only_first_rank and metadata.is_first_rank():
            # Only the first rank will save the cache,
            # so we need to set it lager than other ranks
            first_rank_max_local_cpu_size = (
                config.extra_config.get(
                    "first_rank_max_local_cpu_size", max_local_cpu_size
                )
                if config.extra_config
                else max_local_cpu_size
            )
            return MixedMemoryAllocator(
                int(first_rank_max_local_cpu_size * 1024**3),
                numa_mapping=numa_mapping,
            )
        return MixedMemoryAllocator(
            int(max_local_cpu_size * 1024**3),
            numa_mapping=numa_mapping,
        )

    @staticmethod
    def _Create_token_database(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ) -> TokenDatabase:
        if config.enable_blending:
            return SegmentTokenDatabase(config, metadata)
        return ChunkedTokenDatabase(config, metadata)

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        gpu_connector: Optional[GPUConnectorInterface],
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ) -> LMCacheEngine:
        """
        Builds a new LMCacheEngine instance if it doesn't already exist for the
        given ID.

        raises: ValueError if the instance already exists with a different
            configuration.
        """
        logger.info("Creating LMCacheEngine instance %s", instance_id)
        if instance_id not in cls._instances:
            numa_mapping = NUMADetector.get_numa_mapping(config)
            logger.info("NUMA mapping for instance %s: %s", instance_id, numa_mapping)
            token_database = cls._Create_token_database(config, metadata)
            stat_logger = LMCacheStatsLogger(
                metadata,
                log_interval=10,
                config=config,
            )

            engine = LMCacheEngine(
                config,
                metadata,
                token_database,
                gpu_connector,
                broadcast_fn,
                broadcast_object_fn,
            )

            cls._instances[instance_id] = engine
            cls._cfgs[instance_id] = config
            cls._metadatas[instance_id] = metadata
            cls._stat_loggers[instance_id] = stat_logger
            return engine
        else:
            if (
                cls._cfgs[instance_id] != config
                or cls._metadatas[instance_id] != metadata
            ):
                raise ValueError(
                    f"Instance {instance_id} already exists with a different "
                    f"configuration or metadata."
                )
            return cls._instances[instance_id]

    @classmethod
    def get(cls, instance_id: str) -> Optional[LMCacheEngine]:
        """Returns the LMCacheEngine instance associated with the instance ID,
        or None if not found."""
        return cls._instances.get(instance_id)

    @classmethod
    def destroy(cls, instance_id: str) -> None:
        """Close and delete the LMCacheEngine instance by the instance ID"""
        # TODO: unit test for this
        logger.info("Destroying LMCacheEngine instance: %s", instance_id)

        if instance_id in cls._instances:
            stat_logger = cls._stat_loggers[instance_id]
            try:
                logger.info("Shutting down stats logger...")
                stat_logger.shutdown()
                logger.info("Stats logger shut down successfully")
            except Exception as e:
                logger.error("Error shutting down stats logger: %s", e)

            engine = cls._instances[instance_id]
            try:
                logger.info("Closing cache engine...")
                engine.close()
                logger.info("Cache engine closed successfully")
            except Exception as e:
                logger.error("Error closing cache engine: %s", e)

            try:
                logger.info("Cleaning up instance dictionaries...")
                cls._instances.pop(instance_id, None)
                cls._cfgs.pop(instance_id, None)
                cls._metadatas.pop(instance_id, None)
                cls._stat_loggers.pop(instance_id, None)
                logger.info("Instance dictionaries cleaned up")
            except Exception as e:
                logger.error("Error cleaning up instances: %s", e)

            try:
                logger.info("Destroying stats monitor...")
                LMCStatsMonitor.DestroyInstance()
                logger.info("Stats monitor destroyed successfully")
            except Exception as e:
                logger.error("Error destroying stats monitor: %s", e)

            logger.info("LMCacheEngine instance %s destroyed", instance_id)
        else:
            logger.warning("Instance %s not found for destruction", instance_id)
