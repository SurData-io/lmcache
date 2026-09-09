# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Generator, Optional, Union
import math
import os
import sys
import time
import concurrent.futures

# Third Party
from vllm.config import (
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus
from vllm.version import __version__ as VLLM_VERSION
import torch

try:
    from vllm.v1.outputs import KVConnectorXferRecord
except ImportError:  # pragma: no cover

    @dataclass(frozen=True)
    class KVConnectorXferRecord:  # type: ignore[no-redef]
        request_id: str
        generation: int = 0
        block_ids: tuple[int, ...] = ()
        outcome: str = "success"

# First Party
# Use LMCache's own math utilities instead of vllm's
# (avoids dependency on vllm internal changes like https://github.com/vllm-project/vllm/pull/27188)
from lmcache import utils
from lmcache.banner import print_banner_once
from lmcache.integration.vllm.utils import (
    ENGINE_NAME,
    apply_mm_hashes_to_token_ids,
    extract_mm_features,
    lmcache_get_or_create_config,
)
from lmcache.integration.vllm.vllm_service_factory import VllmServiceFactory
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.utils import CacheStoreEvent, _lmcache_nvtx_annotate, cdiv
from lmcache.v1.cache_engine import LMCacheEngine, PendingLoad, PendingStore, ReadyStoreEvent
from lmcache.v1.compute.blend import LMCBlenderBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.config_base import validate_and_set_config_value
from lmcache.v1.gds_store_policy import gds_defer_partial_prefill_store
from lmcache.v1.manager import LMCacheManager

if TYPE_CHECKING:
    # Third Party
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request

    # First Party
    from lmcache.v1.lookup_client.abstract_client import LookupClientInterface

logger = init_logger(__name__)


@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in LMCache
    lmcache_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool
    # Prefetch/HBM restore generation. Prefetch emit must not delete this
    # spec: a later vLLM preemption still needs the same restore plan.
    prefetch_issued: bool = False
    generation: int = 0
    # Generation copied onto the last wire LoadSpec (before stored gen increments).
    issued_generation: int = 0
    max_h2d_tokens: int = 0
    allocated_span: int = 0


@dataclass
class SaveSpec:
    # Skip already saved tokens
    skip_leading_tokens: int
    # Whether the scheduler allow us to save the tokens
    can_save: bool


@dataclass
class DisaggSpec:
    req_id: str
    receiver_id: str
    receiver_host: str
    receiver_init_port: int
    receiver_alloc_port: int
    is_last_prefill: bool = False
    num_transferred_tokens: int = 0
    total_chunks: int = 0
    receiver_query_port: Optional[list[int]] = None


tmp_disagg_tracker: dict[str, DisaggSpec] = {}


def extract_request_configs(sampling_params: SamplingParams) -> Optional[dict]:
    request_configs = None
    if sampling_params and sampling_params.extra_args is not None:
        if kv_transfer_params := sampling_params.extra_args.get("kv_transfer_params"):
            for k, v in kv_transfer_params.items():
                if k.startswith("lmcache."):
                    if request_configs is None:
                        request_configs = {}
                    request_configs[k] = v
    return request_configs


@dataclass
class RequestTracker:
    # Request id
    req_id: str

    # Total prompt token length
    prompt_len: int

    # The token ids that has been scheduled so far
    token_ids: list[int]

    # The block ids that has been allocated so far
    # NOTE: allocated blocks could be more than the number of tokens
    allocated_block_ids: list[int]

    # The number of tokens that has been saved
    num_saved_tokens: int = 0

    # Disagg spec for the request
    disagg_spec: Optional[DisaggSpec] = None

    # Multimodal hashes and positions
    mm_hashes: Optional[list[str]] = None
    mm_positions: Optional[list["PlaceholderRange"]] = None

    # The configs of the request, includes tags and other configs
    request_configs: Optional[dict] = None

    # Whether the request is in decode phase
    is_decode_phase = False

    # Whether the request cache should be saved
    skip_save: bool = False

    # The number of tokens that are cached in LMCache for this request
    num_lmcache_cached_tokens: int = 0

    # idle | waiting_for_load | ready — MP-style retrieve state.
    load_state: str = "idle"

    @_lmcache_nvtx_annotate
    @staticmethod
    def from_new_request(
        lmcache_config: LMCacheEngineConfig,
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
        lmcache_cached_tokens: int,
        skip_save: bool,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            lmcache_config (LMCacheEngineConfig): the LMCache engine config.
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.
            lmcache_cached_tokens (int): the number of tokens that are
                cached in LMCache.
            request_priority (int): the priority of the request
            skip_save (bool): whether the request cache should be saved
        """
        # vLLM 0.9.0 update: request.block_ids changed from list[int] to
        # tuple[list[int]]
        # Need to check the type of request.block_ids

        unfolded_block_ids = []

        if not isinstance(new_request.block_ids[0], list):
            unfolded_block_ids = new_request.block_ids.copy()
        else:
            # According to the vLLM code
            # (https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/
            # sched/scheduler.py#L943),
            # only one KVCacheGroup is supported in connector for now.

            # TODO: Please support multiple KVCacheGroup in connector.
            # NOTE: Also, `update` method in RequestTracker should be
            # updated accordingly.
            unfolded_block_ids = new_request.block_ids[0].copy()

        # NOTE: Initialized in `update_state_after_alloc`
        disagg_spec = tmp_disagg_tracker.pop(new_request.req_id, None)

        request_configs = extract_request_configs(new_request.sampling_params)

        mm_hashes, mm_positions = extract_mm_features(new_request, modify=True)

        return RequestTracker(
            req_id=new_request.req_id,
            prompt_len=len(new_request.prompt_token_ids),
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].copy(),
            allocated_block_ids=unfolded_block_ids,
            num_saved_tokens=lmcache_cached_tokens,
            disagg_spec=disagg_spec,
            mm_hashes=mm_hashes,
            mm_positions=mm_positions,
            skip_save=skip_save,
            request_configs=request_configs,
            num_lmcache_cached_tokens=lmcache_cached_tokens,
        )

    def update(
        self,
        new_token_ids: list[int],
        new_block_ids: Union[Optional[tuple[list[int], ...]], list[int]],
        preempted: bool = False,
        lmcache_cached_tokens: int = 0,
        vllm_cached_tokens: int = 0,
        all_token_ids: Optional[list[int]] = None,
    ) -> None:
        """Update the request tracker when a running request is
        scheduled again

        vllm_cached_tokens: the number of tokens that are cached in vLLM
        is only used for preempted requests
        all_token_ids: the full token list from the vLLM request, used to
        restore token_ids for preempted requests to ensure chunk keys match
        """

        if new_block_ids is None:
            # https://github.com/vllm-project/vllm/commit/
            # b029de9902aa3ac58806c8c17776c7074175b6db#
            # diff-cafd89ce8a698a56acb24ada62831cbc7a980782f78a52d1742ba238031f296cL94
            new_block_ids = []
        elif len(new_block_ids) == 0:
            new_block_ids = []
        elif isinstance(new_block_ids, tuple):
            new_block_ids = new_block_ids[0]
        elif isinstance(new_block_ids, list):
            # If input is a list, flatten it to handle potential nesting.
            # This also correctly processes already-flat lists.
            new_block_ids = [
                i
                for elem in new_block_ids
                for i in (elem if isinstance(elem, list) else [elem])
            ]
        else:
            raise ValueError(f"Unsupported new_block_ids type {type(new_block_ids)}")

        if preempted:
            assert all_token_ids is not None, (
                f"Preempted request {self.req_id} has no all_token_ids"
            )
            # the block ids will change after preemption
            self.allocated_block_ids = new_block_ids
            # reset the number of saved tokens
            self.num_saved_tokens = lmcache_cached_tokens
            num_computed_tokens = max(lmcache_cached_tokens, vllm_cached_tokens)

            # FIX: For preempted requests, restore token_ids from the full
            # token list to ensure chunk keys match what was used during
            # lookup. The lookup uses request.all_token_ids, so we need the
            # same tokens for retrieve.
            num_tokens_needed = max(
                num_computed_tokens + len(new_token_ids),
                lmcache_cached_tokens,
            )
            self.token_ids = all_token_ids[:num_tokens_needed]
        else:
            self.allocated_block_ids.extend(new_block_ids)
            self.token_ids.extend(new_token_ids)

        # When a request is scheduled again, and the number of new tokens
        # is 1 (excluding chunked prefill), the request is in decode phase.
        # TODO: Need to further exclude the case of chunked prefill with 1 token.
        if len(new_token_ids) == 1:
            self.is_decode_phase = True

    def append_allocated_blocks(self, blocks: Any) -> None:
        """Append only newly allocated block ids (async-load second alloc)."""
        if blocks is None:
            return
        get_ids = getattr(blocks, "get_block_ids", None)
        if not callable(get_ids):
            return
        block_ids = get_ids() or ()
        if not block_ids:
            return
        group0 = block_ids[0] if isinstance(block_ids[0], (list, tuple)) else block_ids
        existing = len(self.allocated_block_ids)
        extra = list(group0[existing:])
        if extra:
            self.allocated_block_ids.extend(extra)


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Request tokens
    token_ids: list[int]  # torch.Tensor
    # Slot mapping
    slot_mapping: torch.Tensor

    # Whether is last prefill or not
    is_last_prefill: bool = False

    # Skip save or not
    save_spec: Optional[SaveSpec] = None
    # load_spec
    load_spec: Optional[LoadSpec] = None
    # disagg spec
    disagg_spec: Optional[DisaggSpec] = None
    # the configs of the request
    request_configs: Optional[dict] = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        lmcache_chunk_size: int = 256,
        load_spec: Optional[LoadSpec] = None,
        discard_partial_chunks: bool = True,
        save_decode_cache: bool = False,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM.
            lmcache_chunk_size (int): the chunk size for LMCache.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            discard_partial_chunks (bool): whether to discard partial chunks.
            save_decode_cache (bool): whether to save the cache in decode phase.

        Returns:
            the request metadata if we need to perform load/save
            operations, None otherwise.
        """
        input_token_ids = tracker.token_ids
        input_token_len = len(input_token_ids)

        is_last_prefill = False
        if input_token_len >= tracker.prompt_len:
            is_last_prefill = True

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        # 3. if save_decode_cache is False and it is in decode phase

        skip_leading_tokens = tracker.num_saved_tokens
        chunk_boundary = (
            cdiv(tracker.num_saved_tokens + 1, lmcache_chunk_size) * lmcache_chunk_size
        )

        # NOTE(vladnosiv): for disagg, you cannot skip saving, as saving is a transfer
        # Check if request_configs has lmcache.skip_save set to True
        request_skip = (tracker.request_configs or {}).get("lmcache.skip_save", False)

        skip_save = tracker.disagg_spec is None and (
            tracker.skip_save
            or (tracker.num_saved_tokens > 0 and input_token_len < chunk_boundary)
            or (tracker.is_decode_phase and not save_decode_cache)
            or request_skip
        )

        if skip_save and load_spec is None:
            return None

        # Calculate number of tokens to save based on discard_partial_chunks
        # setting

        # NOTE(vladnosiv): for the input_token_len chunk prefill,
        # we are required to discard partial chunks,
        # as new tokens will be added in the next iteration.
        if not is_last_prefill or discard_partial_chunks:
            num_tokens_to_save = (
                input_token_len // lmcache_chunk_size * lmcache_chunk_size
            )
        else:
            num_tokens_to_save = input_token_len

        save_spec = SaveSpec(skip_leading_tokens, not skip_save)

        # Calculate the token ids and slot mappings for load and save
        load_tokens = 0
        if load_spec is not None and load_spec.can_load:
            load_tokens = int(load_spec.lmcache_cached_tokens)
        token_ids = input_token_ids[: max(num_tokens_to_save, load_tokens)]

        # If the request has multimodal hashes, apply them to the token ids
        if tracker.mm_hashes:
            # TODO: Optimize this
            token_ids = torch.tensor(token_ids)
            assert tracker.mm_positions is not None, (
                "tracker got mm_hashes but no mm_positions"
            )
            apply_mm_hashes_to_token_ids(
                token_ids, tracker.mm_hashes, tracker.mm_positions
            )
            token_ids = token_ids.tolist()

        num_blocks = len(tracker.allocated_block_ids)

        if len(token_ids) > num_blocks * block_size:
            logger.error(
                "The number of tokens is more than the number of blocks"
                " for request %s. "
                "Something might be wrong in scheduling logic!",
                tracker.req_id,
            )
            logger.error(
                "Num tokens: %d, num blocks: %d, block size: %d",
                len(token_ids),
                num_blocks,
                block_size,
            )

        block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)
        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids.reshape((num_blocks, 1)) * block_size
        )

        slot_mapping = slot_mapping.flatten()[: len(token_ids)]
        assert slot_mapping.dtype == torch.long  # TODO: this could be removed

        # For load operation: log if the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens (%d cached in vLLM) for request %s",
                load_spec.lmcache_cached_tokens,
                load_spec.vllm_cached_tokens,
                tracker.req_id,
            )

        # For disagg requests, compute total_chunks for sender admission control.
        if tracker.disagg_spec is not None and tracker.disagg_spec.total_chunks == 0:
            # Only compute once (on first batch)
            total_chunks_for_req = math.ceil(tracker.prompt_len / lmcache_chunk_size)
            tracker.disagg_spec.total_chunks = total_chunks_for_req

        # Note: We keep load_spec even when can_load=False to pass metrics to worker
        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            is_last_prefill=is_last_prefill,
            save_spec=save_spec,
            load_spec=load_spec,
            disagg_spec=tracker.disagg_spec,
            request_configs=tracker.request_configs,
        )


@dataclass
class LMCacheConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)
    need_flush_before_forward: bool = False
    aborted_req_ids: set[str] = field(default_factory=set)

    @_lmcache_nvtx_annotate
    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)


def _has_preemption_reqs(scheduler_output: SchedulerOutput) -> bool:
    """Return whether the scheduler output contains preemption-related requests.

    A preemption is detected if:
    - ``scheduled_cached_reqs.resumed_req_ids``: requests resumed this step
    - ``scheduler_output.preempted_req_ids``: requests preempted this step
    """
    cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
    resumed_ids = getattr(cached_reqs, "resumed_req_ids", None)
    if resumed_ids:
        return True
    preempted_ids = getattr(scheduler_output, "preempted_req_ids", None)
    if preempted_ids:
        return True
    return False


def infer_num_gpu_blocks(kv_caches: dict[str, torch.Tensor]) -> int:
    """Infer vLLM 0.20.1 paged-KV block count from registered tensors.

    Verified layouts for vLLM 0.20.1:
    - MLA 3D ``[num_blocks, block_size, head_size]``
    - FlashAttention 5D ``[num_blocks, 2, block_size, num_heads, head_size]``
    - Stacked K/V 4D ``[2, num_blocks, block_size, ...]`` (dim0 is K/V)

    Args:
        kv_caches: Layer name to KV tensor mapping from ``register_kv_caches``.

    Returns:
        Shared block count, or 0 when tensors disagree or are empty.
    """
    counts: list[int] = []
    for tensor in kv_caches.values():
        if tensor is None:
            continue
        shape = tuple(int(d) for d in getattr(tensor, "shape", ()) or ())
        if not shape:
            return 0
        if len(shape) >= 4 and shape[0] == 2 and shape[1] > 2:
            counts.append(shape[1])
        else:
            counts.append(shape[0])
    if not counts:
        return 0
    first = counts[0]
    if first <= 0 or any(c != first for c in counts):
        return 0
    return first


class LMCacheConnectorV1Impl:
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        # Banner from the scheduler role only, so tensor-parallel
        # deployments print it once rather than once per worker.
        if role == KVConnectorRole.SCHEDULER:
            print_banner_once(sys.stderr)

        self._parent = parent
        self._vllm_config = vllm_config
        self._role = role
        self.device = vllm_config.device_config.device
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.worker_count = vllm_config.parallel_config.tensor_parallel_size

        # Load and configure LMCache config
        config = lmcache_get_or_create_config()
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration is should be passed for vLLM v1."
        )
        self._apply_extra_config(config, vllm_config)
        self.config = config
        ktc = vllm_config.kv_transfer_config
        timeout_ms = int(getattr(ktc, "kv_hbm_assembly_timeout_ms", 0) or 0)
        # 0 = watchdog disabled; normal GDS/prefetch completes on DMA events.
        if bool(getattr(config, "hbm_prefetch_enable", False)) and timeout_ms != 0 and not (
            50 <= timeout_ms <= 2000
        ):
            raise ValueError(
                "hbm_prefetch_enable requires kv_hbm_assembly_timeout_ms=0 "
                f"(watchdog off) or in [50, 2000], got {timeout_ms}"
            )

        service_factory = VllmServiceFactory(config, vllm_config, role.name.lower())
        self._manager = LMCacheManager(config, service_factory, connector=self)

        # Start services managed by LMCacheManager
        self._manager.start_services()

        # Initialize connector-specific state
        self._init_connector_state(role, vllm_config, config)

        # Setup metrics for monitoring data structures
        self._setup_metrics()

        logger.info(
            "LMCache initialized for role %s with version %s, "
            "vllm version %s, lmcache cache_engine metadata: %s",
            role,
            utils.get_version(),
            VLLM_VERSION,
            getattr(self.lmcache_engine, "metadata", None),
        )

    def _apply_extra_config(
        self, config: LMCacheEngineConfig, vllm_config: "VllmConfig"
    ) -> None:
        """Apply extra config from vLLM to LMCache config."""
        kv_connector_extra_config = (
            vllm_config.kv_transfer_config.kv_connector_extra_config
        )
        if kv_connector_extra_config:
            for key, value in kv_connector_extra_config.items():
                if key.startswith("lmcache."):
                    config_key = key[8:]  # Remove "lmcache." prefix
                    if validate_and_set_config_value(config, config_key, value):
                        logger.info(
                            "Updated config %s from vLLM extra config",
                            config_key,
                        )
        from lmcache.v1.config import enforce_gds_invariants

        enforce_gds_invariants(config)
        validate = getattr(config, "validate", None)
        if callable(validate):
            validate()

    def _init_connector_state(
        self,
        role: KVConnectorRole,
        vllm_config: "VllmConfig",
        config: LMCacheEngineConfig,
    ) -> None:
        """Initialize connector-specific state variables."""
        self.async_loading = config.enable_async_loading
        self.layerwise_retrievers: list[
            Generator[Optional[torch.Tensor], None, None]
        ] = []
        self._layerwise_save_storers: dict[
            str, Generator[Optional[torch.Tensor], None, None]
        ] = {}
        self._stats_monitor = LMCStatsMonitor.GetOrCreate()

        # Role-specific initialization
        if role == KVConnectorRole.SCHEDULER:
            self._unfinished_requests: dict[str, "Request"] = {}
        else:
            self.use_layerwise = config.use_layerwise
            self.enable_blending = config.enable_blending

            if self.enable_blending:
                assert self.lmcache_engine is not None
                assert self.lmcache_engine.gpu_connector is not None, (
                    "GPU connector must be available for blending"
                )
                self.blender = LMCBlenderBuilder.get_or_create(
                    ENGINE_NAME,
                    self.lmcache_engine,
                    self.lmcache_engine.gpu_connector,
                    config,
                )

        # Legacy compatibility check
        self._check_legacy_register_kv_caches()

        self.kv_caches: dict[str, torch.Tensor] = {}
        self._block_size = vllm_config.cache_config.block_size
        self.load_specs: dict[str, LoadSpec] = {}
        self.kv_cache_manager: Optional["KVCacheManager"] = None
        self._request_trackers: dict[str, RequestTracker] = {}

        self._discard_partial_chunks = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "discard_partial_chunks", False
            )
            or not config.save_unfull_chunk
        )

        self._lmcache_chunk_size = config.chunk_size

        self.skip_last_n_tokens = vllm_config.kv_transfer_config.get_from_extra_config(
            "skip_last_n_tokens", 0
        )

        self.num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.current_layer = 0

        self.force_skip_save = bool(os.environ.get("LMCACHE_FORCE_SKIP_SAVE", False))
        self._requests_priority: dict[str, int] = {}

        # Chunked KV loading: cap the number of external tokens
        # reported per scheduling step to prevent GPU block pool
        # exhaustion at high concurrency with long contexts.
        # Configurable via kv_connector_extra_config:
        #   "lmcache.max_tokens_per_load": <int>
        # Default 0 (no cap — original behavior, all matched tokens
        # reported at once). Set to a positive value (e.g. 131072)
        # to enable chunked loading.
        self._max_tokens_per_load: int = int(
            getattr(config, "max_tokens_per_load", 0)
            or vllm_config.kv_transfer_config.get_from_extra_config(
                "lmcache.max_tokens_per_load", 0
            )
            or 0
        )
        if str(getattr(config, "kv_io_mode", "dram") or "dram").lower() == "gds":
            self._max_tokens_per_load = 0
        self._invalid_block_ids: set[int] = set()

        # Worker-side async D2H store (not shared with the scheduler process).
        self._pending_saves: dict[str, list[PendingStore]] = {}
        self._ready_saves: set[str] = set()
        # finished_req_ids is one-shot; remember finished reqs whose D2H is
        # still in flight so get_finished can report finished_sending later.
        self._finished_reqs_waiting_for_store: set[str] = set()
        self._pending_bytes: int = 0
        self._delayed_free_blocks: int = 0
        self._async_store_drops: int = 0
        self._async_d2h_inflight: int = 0
        self._async_d2h_event_ready: int = 0
        self._async_put_submit: int = 0
        self._async_put_failure: int = 0
        self._async_preempt_flush: int = 0
        self._async_t_issue_us: int = 0
        self._async_t_query_us: int = 0
        self._async_t_complete_us: int = 0
        self._async_t_sync_us: int = 0
        self._slot_pin_pool: list = []
        self._slot_pin_busy: set = set()
        self._slot_pin_inflight: list = []
        self._pending_loads: dict[tuple[str, int], PendingLoad] = {}
        self._xfer_recving: list = []
        self._xfer_cancelled: list = []
        self._prefetch_issue_count = 0
        self._prefetch_event_ready_count = 0
        self._prefetch_skip_watermark = 0
        self._prefetch_skip_inflight = 0
        self._prefetch_skip_cuda_free = 0
        self._prefetch_cancelled = 0
        self._prefetch_inflight = 0
        self._prefetch_pending_gather = 0
        self._prefetch_capacity_wait = 0
        self._prefetch_fallback = 0
        self._async_put_complete: int = 0
        self._async_put_backlog: int = 0
        self._host_pinned_bytes: int = 0
        self._hist_d2h_issue_to_ready_us: list[int] = []
        self._hist_ready_to_release_us: list[int] = []
        self._hist_h2d_issue_to_ready_us: list[int] = []
        self._own_stale_event = 0
        self._own_duplicate_generation = 0
        self._own_block_mismatch = 0
        self._own_early_free = 0
        self._h2d_fifo_depth = 0
        self._put_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="lmcache-host-put"
        )
        self._cancelled_recving: set[str] = set()
        self._hbm_cancelled_req_ids: set[str] = set()
        self._pending_worker_abort_ids: set[str] = set()
        self._l1_future_not_ready = 0
        self._gds_lookup_defer_started: dict[str, float] = {}
        self._gds_lookup_defer_counts: dict[str, int] = {}
        self._gds_lookup_fallback_ids: set[str] = set()
        self._gds_lookup_fallback_count = 0
        self._gds_lookup_late_hit_count = 0
        self._last_cuda_free_bytes = None
        self._last_compute_load_sync_ms = 0.0
        self._load_spec_emit_prefetch = 0
        self._load_spec_emit_new = 0
        self._load_spec_emit_resumed = 0
        self._load_spec_missing = 0
        self._load_spec_rebuilt = 0
        self._hbm_preempt_resume = 0
        self._load_spec_finish = 0
        self._load_spec_abort = 0
        self._async_max_pending_bytes = int(
            getattr(config, "async_store_max_pending_bytes", 0) or 0
        )
        self._async_max_pending_requests = int(
            getattr(config, "async_store_max_pending_requests", 0) or 0
        )
        self._async_max_delayed_free_blocks = int(
            getattr(config, "async_store_max_delayed_free_blocks", 0) or 0
        )
        self._async_max_delayed_free_ratio = float(
            getattr(config, "async_store_max_delayed_free_ratio", 0.25) or 0.0
        )
        self._gpu_kv_block_total = 0
        self._gpu_kv_block_total_error_logged = False
        logger.info(
            "async_store_d2h_enabled=%s layerwise=%s max_pending_bytes=%s "
            "max_pending_requests=%s max_delayed_free_blocks=%s "
            "max_delayed_free_ratio=%s",
            self._async_store_enabled(),
            bool(getattr(self, "use_layerwise", False)),
            self._async_max_pending_bytes,
            self._async_max_pending_requests,
            self._async_max_delayed_free_blocks,
            self._async_max_delayed_free_ratio,
        )

    def _check_legacy_register_kv_caches(self) -> None:
        """Check for legacy connector without register_kv_caches implementation."""
        if self.lmcache_engine is None:
            return

        child_class = self._parent.__class__
        parent_class = KVConnectorBase_V1
        child_method = getattr(child_class, "register_kv_caches", None)
        parent_method = getattr(parent_class, "register_kv_caches", None)

        if child_method is None or parent_method is None:
            implements = False
        else:
            implements = child_method is not parent_method

        if not implements:
            logger.warning(
                "Please use the latest lmcache connector, otherwise some "
                "features may not work, such as DSA"
            )
            self._manager.post_init()

    # ==================== Property Accessors ====================

    @property
    def lmcache_engine(self) -> Optional[LMCacheEngine]:
        """Get the LMCache engine instance from manager."""
        return self._manager.lmcache_engine

    @property
    def lmcache_engine_metadata(self):
        """Get the LMCache engine metadata from manager."""
        return self._manager.lmcache_engine_metadata

    @property
    def lookup_client(self) -> Optional["LookupClientInterface"]:
        """Get the lookup client from manager."""
        return self._manager.lookup_client

    @property
    def lookup_server(self):
        """Get the lookup server from manager."""
        return self._manager.lookup_server

    def _setup_metrics(self) -> None:
        """Setup metrics for monitoring data structures in the connector."""
        metadata = self._manager.lmcache_engine_metadata
        if metadata is None:
            logger.warning(
                "LMCache metadata is not initialized, "
                "connector metrics will not be collected"
            )
            return
        prometheus_logger = PrometheusLogger.GetOrCreate(
            metadata,
            config=self.config,
        )

        # Set up metrics for scheduler-specific and general data structures
        metrics_map = {
            "_unfinished_requests": "scheduler_unfinished_requests_count",
            "load_specs": "connector_load_specs_count",
            "_request_trackers": "connector_request_trackers_count",
            "kv_caches": "connector_kv_caches_count",
            "layerwise_retrievers": "connector_layerwise_retrievers_count",
            "_invalid_block_ids": "connector_invalid_block_ids_count",
            "_requests_priority": "connector_requests_priority_count",
        }

        for attr_name, metric_name in metrics_map.items():
            if hasattr(self, attr_name):
                metric = getattr(prometheus_logger, metric_name)
                # Use a default argument in the lambda to capture
                # the current value of `attr_name`
                # to avoid issues with late binding in closures.
                metric.set_function(lambda name=attr_name: len(getattr(self, name)))

    def get_inference_info(self) -> dict:
        """Get inference information including vLLM config and related details.

        Returns:
            dict: Dictionary containing inference information
        """
        # Get vLLM config information
        vllm_config = self._vllm_config

        # Use vLLM config's string representation and add specific configs
        inference_info = {
            "vllm_version": VLLM_VERSION,
            "lmcache_version": utils.get_version(),
            "vllm_config": str(vllm_config),
            "model_config": {
                "model": getattr(vllm_config.model_config, "model", None),
                "dtype": str(getattr(vllm_config.model_config, "dtype", None)),
                "max_model_len": getattr(
                    vllm_config.model_config, "max_model_len", None
                ),
                "vocab_size": getattr(vllm_config.model_config, "vocab_size", None),
                "num_layers": getattr(
                    vllm_config.model_config, "get_num_layers", lambda _: None
                )(vllm_config.parallel_config),
                "num_attention_heads": getattr(
                    vllm_config.model_config, "get_num_attention_heads", lambda _: None
                )(vllm_config.parallel_config),
                "num_kv_heads": getattr(
                    vllm_config.model_config, "get_num_kv_heads", lambda _: None
                )(vllm_config.parallel_config),
                "head_size": getattr(
                    vllm_config.model_config, "get_head_size", lambda: None
                )(),
            },
            "cache_config": {
                "block_size": getattr(vllm_config.cache_config, "block_size", None),
                "cache_dtype": str(
                    getattr(vllm_config.cache_config, "cache_dtype", None)
                ),
                "gpu_memory_utilization": getattr(
                    vllm_config.cache_config, "gpu_memory_utilization", None
                ),
                "swap_space": getattr(vllm_config.cache_config, "swap_space", None),
                "enable_prefix_caching": getattr(
                    vllm_config.cache_config, "enable_prefix_caching", None
                ),
            },
        }

        return inference_info

    def get_inference_version(self) -> str:
        """Get vLLM version information.

        Returns:
            str: vLLM version string
        """
        return VLLM_VERSION

    # TODO(chunxiaozheng): in the latest lmcache_connector, we use `register_kv_caches`
    #  to init self.kv_caches, we keep it in order to be compatible with old versions
    #  and will be removed in the future.
    @_lmcache_nvtx_annotate
    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext"):
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                logger.debug("The layer %s does not have kv_cache, skip it", layer_name)
                continue

            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[
                    forward_context.virtual_engine
                ]

    ####################
    # Worker side APIs
    ####################
    @_lmcache_nvtx_annotate
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        logger.info("Registering KV caches")
        # TODO(chunxiaozheng): `_init_kv_caches_from_forward_context` is
        #  not called, we should consider removing it.
        assert len(self.kv_caches) == 0 and len(kv_caches) > 0
        self.kv_caches = kv_caches
        self._gpu_kv_block_total = infer_num_gpu_blocks(kv_caches)
        if (
            float(getattr(self, "_async_max_delayed_free_ratio", 0.0) or 0.0) > 0.0
            and self._gpu_kv_block_total <= 0
        ):
            logger.error(
                "async_store_max_delayed_free_ratio is enabled but GPU KV "
                "block count could not be inferred from registered tensors"
            )
        logger.info(
            "registered kv caches layers=%s gpu_kv_block_total=%s",
            len(kv_caches),
            self._gpu_kv_block_total,
        )
        self._manager.post_init()

    @_lmcache_nvtx_annotate
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation
        """
        self.current_layer = 0

        if len(self.kv_caches) == 0:
            logger.warning(
                "Please update LMCacheConnector, "
                "use register_kv_caches to init kv_caches"
            )
            self._init_kv_caches_from_forward_context(forward_context)

        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.debug(
                "In connector.start_load_kv, attn_metadata is None "
                "(KV-only / no-forward step); still issuing loads"
            )

        # LMCache failed to initialize and is running in degraded mode; skip the
        # KV load so vLLM falls back to recompute instead of crashing EngineCore.
        if self.lmcache_engine is None:
            return

        wait_xfer = getattr(
            getattr(self.lmcache_engine, "gpu_connector", None),
            "wait_inflight_xfer_for_compute",
            None,
        )
        if self._async_store_enabled() and callable(wait_xfer):
            wait_xfer()

        self.layerwise_retrievers = []

        for idx, request in enumerate(metadata.requests):
            if request.load_spec is None or not request.load_spec.can_load:
                continue
            last_idx = idx

        for idx, request in enumerate(metadata.requests):
            # Update metrics for all requests that have a load_spec
            if request.load_spec is not None:
                self._stats_monitor.update_interval_vllm_hit_tokens(
                    request.load_spec.vllm_cached_tokens
                )
                self._stats_monitor.update_interval_prompt_tokens(
                    len(request.token_ids)
                )

            if request.load_spec is None or not request.load_spec.can_load:
                continue

            tokens = request.token_ids
            slot_mapping = request.slot_mapping
            assert len(tokens) == len(slot_mapping)

            token_mask = torch.ones(len(tokens), dtype=torch.bool)
            masked_token_count = (
                request.load_spec.vllm_cached_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            token_mask[:masked_token_count] = False

            lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens
            if self.use_layerwise:
                slot_gpu = self._copy_slot_mapping_to_device(slot_mapping)
                if idx == last_idx:
                    sync = True
                else:
                    sync = False
                # NOTE(Jiayi): Perform blending before layerwise prefix caching
                if self.enable_blending:
                    # TODO(Jiayi): Need to make prefix caching and blending compatible
                    self.blender.blend(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_gpu[:lmcache_cached_tokens],
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                    )
                else:
                    layerwise_retriever = self.lmcache_engine.retrieve_layer(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_gpu[:lmcache_cached_tokens],
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                        sync=sync,
                    )
                    # NOTE: retrieve for two layers at the first layer
                    next(layerwise_retriever)
                    next(layerwise_retriever)
                    self.layerwise_retrievers.append(layerwise_retriever)
            else:
                if self.kv_io_mode() == "gds":
                    slot_for_retrieve = self._slot_mapping_for_gds_page_direct(
                        slot_mapping
                    )
                else:
                    slot_for_retrieve = self._copy_slot_mapping_to_device(
                        slot_mapping
                    )
                retrieve_kwargs = dict(
                    kvcaches=kvcaches,
                    slot_mapping=slot_for_retrieve[:lmcache_cached_tokens],
                    vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                    request_configs=request.request_configs,
                    req_id=request.req_id,
                )
                generation = int(
                    getattr(request.load_spec, "issued_generation", 0)
                    or getattr(request.load_spec, "generation", 0)
                    or 0
                )
                block_ids = tuple(
                    int(x) for x in (getattr(request, "allocated_block_ids", None) or [])
                )
                if self._hbm_prefetch_enabled() or self.kv_io_mode() in ("dram", "gds"):
                    self._start_or_progress_async_load(
                        request,
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        retrieve_kwargs,
                        generation,
                        block_ids,
                    )
                    continue
                t_load = time.perf_counter()
                slot_gpu = self._copy_slot_mapping_to_device(slot_mapping)
                retrieve_kwargs["slot_mapping"] = slot_gpu[:lmcache_cached_tokens]
                ret_token_mask = self.lmcache_engine.retrieve(
                    tokens[:lmcache_cached_tokens],
                    token_mask[:lmcache_cached_tokens],
                    **retrieve_kwargs,
                )
                cancelled_id = getattr(
                    self.lmcache_engine, "_last_cancelled_load", None
                )
                if cancelled_id:
                    self.lmcache_engine._last_cancelled_load = None
                    self._cancelled_recving.add(str(cancelled_id))
                    self._prefetch_cancelled = (
                        int(getattr(self, "_prefetch_cancelled", 0)) + 1
                    )
                    self._l1_future_not_ready = (
                        int(getattr(self, "_l1_future_not_ready", 0)) + 1
                    )
                    unpin = getattr(self.lmcache_engine, "lookup_unpin", None)
                    if callable(unpin):
                        unpin(request.req_id)
                    logger.warning(
                        "cancelled_recving req=%s generation=%s reason=l1_future_not_ready",
                        request.req_id,
                        int(
                            getattr(
                                getattr(request, "load_spec", None), "generation", 0
                            )
                            or 0
                        ),
                    )
                    continue
                if retrieve_kwargs.get("sync", True):
                    self._last_compute_load_sync_ms = (
                        time.perf_counter() - t_load
                    ) * 1000.0
                else:
                    self._last_compute_load_sync_ms = 0.0
                pending = getattr(self.lmcache_engine, "_last_pending_load", None)
                if self._hbm_prefetch_enabled():
                    if pending is not None:
                        self.lmcache_engine._last_pending_load = None
                    else:
                        pending = PendingLoad(
                            req_id=request.req_id,
                            chunks=[],
                            event=ReadyStoreEvent(),
                        )
                    self._pending_loads[request.req_id] = pending
                    self._prefetch_issue_count += 1
                    self._prefetch_inflight = len(self._pending_loads)

                # Check the result
                num_retrieved_tokens = ret_token_mask.sum().item()
                num_expected_tokens = (
                    lmcache_cached_tokens - request.load_spec.vllm_cached_tokens
                )
                if num_retrieved_tokens < num_expected_tokens:
                    logger.error(
                        "Request %s"
                        "The number of retrieved tokens is less than the "
                        "expected number of tokens! This should not happen!",
                        request.req_id,
                    )
                    logger.error(
                        "Num retrieved tokens: %d, num expected tokens: %d",
                        num_retrieved_tokens,
                        num_expected_tokens,
                    )
                    """
                    Report failed block IDs in case of partial failure.
                    """
                    missing_blocks = self.record_failed_blocks(
                        request.req_id,
                        token_mask[:lmcache_cached_tokens],
                        ret_token_mask,
                        slot_mapping[:lmcache_cached_tokens],
                    )
                    self._invalid_block_ids.update(missing_blocks)

    def record_failed_blocks(
        self,
        request_id: str,
        expected_mask: torch.Tensor,
        ret_mask: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> set[int]:
        """Record block IDs associated with failed load attempts.

        Args:
            request_id: request id from vLLM.
            expected_mask: Boolean tensor indicating which tokens were expected to
                be loaded from LMCache. True means the token should be loaded,
                False means the token is already cached in vLLM and does not need
                to be loaded from LMCache.
            ret_mask: Boolean tensor indicating which tokens were actually
                successfully retrieved from LMCache. True means the token was
                successfully loaded. For example, if 256 tokens are expected to be
                loaded, but only 192 tokens are successfully loaded, then the
                ret_mask will be a tensor of 256 items like [T, T, ..., F, F, ...]
                where the first 192 elements are True and the last 64 elements
                are False.
            slot_mapping: Tensor indicating slot IDs for each token. The block
                ID is computed by dividing the slot ID by the block size.

        Example:
            expected_mask = [F, T, T, T] meaning the 1st is in vLLM cache
            ret_mask = [F, T, F, F] meaning failure from loading the 3rd
            missing_mask = expected_mask & ~ret_mask = [F, F, T, T]
            missing_indices = [2, 3]
            then missing_blocks is calculated from slot_mapping and missing_indices

        Returns:
            set[int]: Set of block IDs that failed to load.
        """

        if expected_mask.numel() == 0:
            return set()

        expected_mask_cpu = expected_mask.to(device="cpu", dtype=torch.bool)
        ret_mask_cpu = ret_mask.to(device="cpu", dtype=torch.bool)

        if ret_mask_cpu.shape[0] != expected_mask_cpu.shape[0]:
            logger.debug("expected_mask_cpu.shape[0] != ret_mask_cpu.shape[0]")
            return set()

        missing_mask = expected_mask_cpu & ~ret_mask_cpu
        if not torch.any(missing_mask):
            return set()

        missing_indices = torch.nonzero(missing_mask, as_tuple=False).view(-1)
        if missing_indices.numel() == 0:
            return set()

        slot_mapping_cpu = slot_mapping.to(device="cpu", dtype=torch.long)
        if slot_mapping_cpu.shape[0] > missing_mask.shape[0]:
            slot_mapping_cpu = slot_mapping_cpu[: missing_mask.shape[0]]

        missing_blocks_tensor = torch.unique(
            slot_mapping_cpu[missing_indices] // self._block_size
        )
        missing_blocks = {int(block.item()) for block in missing_blocks_tensor}

        if not missing_blocks:
            return set()

        logger.warning(
            "Request %s failed to load %d tokens across %d blocks",
            request_id,
            missing_indices.numel(),
            len(missing_blocks),
        )
        return missing_blocks

    @_lmcache_nvtx_annotate
    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        if self.layerwise_retrievers:
            logger.debug(f"Waiting for layer {self.current_layer} to be loaded")

        # Wait for the layer to be loaded
        for layerwise_retriever in self.layerwise_retrievers:
            ret_token_mask = next(layerwise_retriever)

            if self.current_layer == self.num_layers - 1:
                assert ret_token_mask is not None
                num_retrieved_tokens = ret_token_mask.sum().item()
                logger.info(f"Retrieved {num_retrieved_tokens} tokens")

        if self.layerwise_retrievers:
            self.current_layer += 1

        return

    @_lmcache_nvtx_annotate
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        # Degraded mode (LMCache init failed): nothing to save, fall back silently.
        if self.lmcache_engine is None:
            return

        if not self.use_layerwise:
            return

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return
        if self._parent._connector_metadata is None:
            logger.warning(
                "In connector.save_kv_layer, but the connector metadata is None"
            )
            return
        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0

        kvcaches = list(self.kv_caches.values())
        is_first = True

        for request in connector_metadata.requests:
            save_spec = request.save_spec
            if (
                save_spec is None or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                continue
            if gds_defer_partial_prefill_store(
                getattr(self.config, "kv_io_mode", "dram"),
                bool(request.is_last_prefill),
            ):
                continue

            layerwise_storer = self._layerwise_save_storers.get(request.req_id)
            if layerwise_storer is None:
                token_ids = request.token_ids
                assert isinstance(token_ids, list)

                slot_mapping = request.slot_mapping
                assert isinstance(slot_mapping, torch.Tensor)
                assert len(slot_mapping) == len(token_ids)

                slot_mapping = self._copy_slot_mapping_to_device(slot_mapping)

                if self.kv_role == "kv_producer":
                    skip_leading_tokens = 0
                else:
                    assert save_spec is not None
                    skip_leading_tokens = save_spec.skip_leading_tokens

                    if skip_leading_tokens == len(token_ids):
                        continue  # skip this request
                    # Align to lmcache chunk size
                    skip_leading_tokens = (
                        skip_leading_tokens
                        // self._lmcache_chunk_size
                        * self._lmcache_chunk_size
                    )

                store_mask = torch.ones(len(token_ids), dtype=torch.bool)
                store_mask[:skip_leading_tokens] = False

                logger.debug(
                    "Storing KV cache for %d out of %d tokens "
                    "(skip_leading_tokens=%d) for request %s",
                    len(token_ids) - skip_leading_tokens,
                    len(token_ids),
                    skip_leading_tokens,
                    request.req_id,
                )

                # TODO (Jiayi): need to make layerwise storing
                # compatible with disagg spec
                layerwise_storer = self.lmcache_engine.store_layer(
                    token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                    offset=skip_leading_tokens,
                    sync=is_first,
                    req_id=request.req_id,
                )
                self._layerwise_save_storers[request.req_id] = layerwise_storer
                if is_first:
                    is_first = False

            next(layerwise_storer)

    def _async_store_enabled(self) -> bool:
        """Return whether registered async D2H store is enabled.

        Scheduler and worker both must agree: ``self.use_layerwise`` is set on
        the worker after register_kv_caches, while the scheduler only has
        ``config.use_layerwise``. Mismatch would delay-free GPU pages on the
        scheduler while ``get_finished`` on the worker returned nothing.
        """
        if bool(getattr(self, "use_layerwise", False)):
            return False
        cfg = getattr(self, "config", None)
        if cfg is None or bool(getattr(cfg, "use_layerwise", False)):
            return False
        return bool(getattr(cfg, "async_store_d2h_enabled", False))

    def _hist_p50_us(self, name: str) -> int:
        vals = list(getattr(self, name, None) or [])
        if not vals:
            return 0
        ordered = sorted(int(x) for x in vals)
        return int(ordered[len(ordered) // 2])

    def _hist_append(self, name: str, us: int, cap: int = 256) -> None:
        lst = getattr(self, name, None)
        if lst is None:
            lst = []
            setattr(self, name, lst)
        lst.append(int(us))
        if len(lst) > cap:
            del lst[: len(lst) - cap]

    def async_store_metrics_snapshot(self) -> dict[str, int]:
        return {
            "d2h_inflight": int(getattr(self, "_async_d2h_inflight", 0)),
            "d2h_event_ready": int(getattr(self, "_async_d2h_event_ready", 0)),
            "put_submit": int(getattr(self, "_async_put_submit", 0)),
            "put_complete": int(getattr(self, "_async_put_complete", 0)),
            "put_backlog": int(getattr(self, "_async_put_backlog", 0)),
            "put_failure": int(getattr(self, "_async_put_failure", 0)),
            "host_put_wait_us": int(getattr(self, "_async_t_complete_us", 0)),
            "host_pinned_bytes": int(getattr(self, "_host_pinned_bytes", 0)),
            "store_drop": int(getattr(self, "_async_store_drops", 0)),
            "pending_bytes": int(getattr(self, "_pending_bytes", 0)),
            "delayed_free_blocks": int(getattr(self, "_delayed_free_blocks", 0)),
            "pending_requests": len(getattr(self, "_pending_saves", {}) or {}),
            "waiting_for_store": len(
                getattr(self, "_finished_reqs_waiting_for_store", set()) or set()
            ),
            "ready_saves": len(getattr(self, "_ready_saves", set()) or set()),
            "preempt_flush": int(getattr(self, "_async_preempt_flush", 0)),
            "t_issue_us": int(getattr(self, "_async_t_issue_us", 0)),
            "t_query_us": int(getattr(self, "_async_t_query_us", 0)),
            "t_complete_us": int(getattr(self, "_async_t_complete_us", 0)),
            "t_sync_us": int(getattr(self, "_async_t_sync_us", 0)),
            "d2h_issue_to_ready_p50_us": self._hist_p50_us(
                "_hist_d2h_issue_to_ready_us"
            ),
            "d2h_ready_to_release_p50_us": self._hist_p50_us(
                "_hist_ready_to_release_us"
            ),
            "h2d_issue_to_ready_p50_us": self._hist_p50_us(
                "_hist_h2d_issue_to_ready_us"
            ),
            "own_stale_event": int(getattr(self, "_own_stale_event", 0)),
            "own_duplicate_generation": int(
                getattr(self, "_own_duplicate_generation", 0)
            ),
            "own_block_mismatch": int(getattr(self, "_own_block_mismatch", 0)),
            "own_early_free": int(getattr(self, "_own_early_free", 0)),
            "h2d_fifo_depth": int(getattr(self, "_h2d_fifo_depth", 0)),
            "prefetch_issue_count": int(getattr(self, "_prefetch_issue_count", 0)),
            "prefetch_event_ready_count": int(
                getattr(self, "_prefetch_event_ready_count", 0)
            ),
            "prefetch_skip_watermark": int(
                getattr(self, "_prefetch_skip_watermark", 0)
            ),
            "prefetch_skip_inflight": int(getattr(self, "_prefetch_skip_inflight", 0)),
            "prefetch_skip_cuda_free": int(
                getattr(self, "_prefetch_skip_cuda_free", 0)
            ),
            "cuda_free_bytes": int(getattr(self, "_last_cuda_free_bytes", 0) or 0),
            "prefetch_cancelled": int(getattr(self, "_prefetch_cancelled", 0)),
            "prefetch_pending_gather": int(
                getattr(self, "_prefetch_pending_gather", 0)
            ),
            "prefetch_capacity_wait": int(
                getattr(self, "_prefetch_capacity_wait", 0)
            ),
            "prefetch_fallback": int(getattr(self, "_prefetch_fallback", 0)),
            "lookup_inflight": sum(
                1
                for v in (getattr(self.lookup_client, "reqs_status", {}) or {}).values()
                if v is None
            ),
            "waiting_for_load": sum(
                1
                for t in (getattr(self, "_request_trackers", {}) or {}).values()
                if getattr(t, "load_state", "") == "waiting_for_load"
            ),
            "l1_future_not_ready": int(
                getattr(getattr(self, "lmcache_engine", None), "_l1_future_not_ready", 0)
                or getattr(self, "_l1_future_not_ready", 0)
            ),
            "hbm_inflight": int(getattr(self, "_prefetch_inflight", 0)),
            "h2d_event_ready": int(
                getattr(self, "_prefetch_event_ready_count", 0)
            ),
            "hbm_load_cancelled": int(getattr(self, "_prefetch_cancelled", 0)),
            "compute_load_sync_ms": float(
                getattr(self, "_last_compute_load_sync_ms", 0.0)
            ),
            "load_spec_emit_prefetch": int(
                getattr(self, "_load_spec_emit_prefetch", 0)
            ),
            "load_spec_emit_new": int(getattr(self, "_load_spec_emit_new", 0)),
            "load_spec_emit_resumed": int(
                getattr(self, "_load_spec_emit_resumed", 0)
            ),
            "load_spec_missing": int(getattr(self, "_load_spec_missing", 0)),
            "load_spec_rebuilt": int(getattr(self, "_load_spec_rebuilt", 0)),
            "hbm_preempt_resume": int(getattr(self, "_hbm_preempt_resume", 0)),
            "load_spec_finish": int(getattr(self, "_load_spec_finish", 0)),
            "load_spec_abort": int(getattr(self, "_load_spec_abort", 0)),
        }

    def _slot_mapping_device_is_cuda(self) -> bool:
        device = self.device
        if isinstance(device, torch.device):
            return device.type == "cuda"
        return str(device).startswith("cuda")

    def _gc_slot_pin_inflight(self) -> None:
        still = []
        for event, buf in getattr(self, "_slot_pin_inflight", []) or []:
            query = getattr(event, "query", None)
            if callable(query) and not query():
                still.append((event, buf))
            else:
                idx = getattr(buf, "_lmc_pin_idx", None)
                if idx is not None:
                    self._slot_pin_busy.discard(int(idx))
        self._slot_pin_inflight = still

    def _acquire_slot_pin(self, n: int, dtype: torch.dtype) -> torch.Tensor:
        for i, buf in enumerate(self._slot_pin_pool):
            if i in self._slot_pin_busy:
                continue
            if buf.numel() >= n and buf.dtype == dtype:
                self._slot_pin_busy.add(i)
                buf._lmc_pin_idx = i  # type: ignore[attr-defined]
                return buf
        cap = max(n, 4096)
        buf = torch.empty(cap, dtype=dtype, pin_memory=True)
        idx = len(self._slot_pin_pool)
        self._slot_pin_pool.append(buf)
        self._slot_pin_busy.add(idx)
        buf._lmc_pin_idx = idx  # type: ignore[attr-defined]
        return buf

    def _copy_slot_mapping_to_device(
        self, slot_mapping: torch.Tensor
    ) -> torch.Tensor:
        """Copy slot_mapping to the connector device without pageable H2D sync.

        Pageable ``Tensor.to(cuda)`` can drain the compute stream. Stage through
        a pinned host buffer, then ``non_blocking`` H2D on the current stream so
        ``store_stream.wait_stream(current)`` observes the copy. Each in-flight
        H2D owns a pool pin until the copy event is ready — a shared pin would
        race CPU overwrite against DMA.
        """
        if slot_mapping is None:
            return slot_mapping
        if not self._slot_mapping_device_is_cuda() or not torch.cuda.is_available():
            dst = self.device
            if str(slot_mapping.device) == str(dst):
                return slot_mapping
            return slot_mapping.to(dst)
        if slot_mapping.is_cuda:
            return slot_mapping

        cpu = slot_mapping.detach().contiguous()
        if cpu.dtype != torch.int64:
            cpu = cpu.to(dtype=torch.int64)
        if cpu.device.type != "cpu":
            cpu = cpu.to("cpu")
        n = int(cpu.numel())
        if n == 0:
            return torch.empty_like(cpu, device=self.device)

        self._gc_slot_pin_inflight()
        pinned = self._acquire_slot_pin(n, cpu.dtype)
        pinned[:n].copy_(cpu.view(-1))
        gpu = torch.empty(cpu.shape, dtype=cpu.dtype, device=self.device)
        gpu.reshape(-1).copy_(pinned[:n], non_blocking=True)
        gpu._lmc_pin = pinned  # type: ignore[attr-defined]
        event = torch.cuda.Event()
        event.record()
        self._slot_pin_inflight.append((event, pinned))
        return gpu

    def _slot_mapping_for_gds_page_direct(
        self, slot_mapping: torch.Tensor
    ) -> torch.Tensor:
        """Return CPU slot_mapping for GDS page-direct I/O.

        Page-direct uses slot ids only to compute block_id and GPU page
        addresses. Keep the tensor on CPU so ``get_page_direct_views`` never
        does a per-chunk synchronous D2H.

        Args:
            slot_mapping: Token-to-slot tensor. CPU is the expected contract.

        Returns:
            The input tensor when it is already on CPU; otherwise a one-shot
            CPU copy of a CUDA tensor.

        Raises:
            ValueError: If ``slot_mapping`` is None.
        """
        if slot_mapping is None:
            raise ValueError("GDS page-direct requires slot_mapping")
        if slot_mapping.device.type == "cpu":
            return slot_mapping
        count = int(getattr(self, "_gds_slot_mapping_cuda_fallback", 0) or 0) + 1
        self._gds_slot_mapping_cuda_fallback = count
        logger.warning(
            "GDS page-direct received CUDA slot_mapping; converting once "
            "(count=%s). Adapter must pass CPU mapping.",
            count,
        )
        return slot_mapping.detach().to("cpu")

    def _gpu_block_count_upper_bound(self, slot_mapping: torch.Tensor) -> int:
        """Upper-bound GPU blocks without host sync or ``torch.unique``.

        Args:
            slot_mapping: Token-to-slot tensor; may be on GPU.

        Returns:
            ``ceil(numel / block_size)``, at least 1 when the mapping is non-empty.
            This is a conservative watermark estimate (never lower than distinct
            blocks for a contiguous token range).
        """
        if slot_mapping is None or slot_mapping.numel() == 0:
            return 0
        block_size = max(1, int(getattr(self, "_block_size", 1) or 1))
        n = int(slot_mapping.numel())
        return max(1, (n + block_size - 1) // block_size)

    def _gpu_block_count(self, slot_mapping: torch.Tensor) -> int:
        """Count distinct GPU blocks from a CPU slot_mapping (never unique() on GPU)."""
        if slot_mapping is None or slot_mapping.numel() == 0:
            return 0
        cpu = slot_mapping
        if cpu.device.type != "cpu":
            cpu = cpu.detach().to("cpu")
        valid = cpu[cpu >= 0]
        if valid.numel() == 0:
            return 0
        block_size = max(1, int(getattr(self, "_block_size", 1) or 1))
        return int(torch.unique(valid // block_size).numel())

    def _async_store_over_watermark(
        self, req_id: str, nbytes: int, nblocks: int
    ) -> bool:
        max_bytes = int(getattr(self, "_async_max_pending_bytes", 0) or 0)
        max_reqs = int(getattr(self, "_async_max_pending_requests", 0) or 0)
        max_blocks = int(getattr(self, "_async_max_delayed_free_blocks", 0) or 0)
        ratio = float(getattr(self, "_async_max_delayed_free_ratio", 0.0) or 0.0)
        pending_bytes = int(getattr(self, "_pending_bytes", 0))
        delayed_blocks = int(getattr(self, "_delayed_free_blocks", 0))
        pending_saves = getattr(self, "_pending_saves", None) or {}
        if max_bytes > 0 and pending_bytes + nbytes > max_bytes:
            self._async_store_drop_reason = "watermark_bytes"
            return True
        new_req = 0 if req_id in pending_saves else 1
        if max_reqs > 0 and len(pending_saves) + new_req > max_reqs:
            self._async_store_drop_reason = "watermark_requests"
            return True
        derived = 0
        if ratio > 0.0:
            total = int(getattr(self, "_gpu_kv_block_total", 0) or 0)
            if total <= 0:
                if not getattr(self, "_gpu_kv_block_total_error_logged", False):
                    logger.error(
                        "async store refused: delayed-free ratio %.3f enabled "
                        "but gpu_kv_block_total is unknown",
                        ratio,
                    )
                    self._gpu_kv_block_total_error_logged = True
                self._async_store_drop_reason = "watermark_ratio_unknown"
                return True
            derived = max(1, int(total * ratio))
        if derived > 0 and max_blocks > 0:
            max_blocks = min(derived, max_blocks)
        elif derived > 0:
            max_blocks = derived
        if max_blocks > 0 and delayed_blocks + nblocks > max_blocks:
            self._async_store_drop_reason = "watermark_blocks"
            return True
        self._async_store_drop_reason = ""
        return False

    def _account_pending_add(self, pending: PendingStore) -> None:
        self._pending_bytes += int(pending.tot_kv_size)
        self._delayed_free_blocks += int(pending.gpu_block_count)
        self._async_d2h_inflight += 1
        self._pending_saves.setdefault(pending.req_id, []).append(pending)

    def _account_pending_remove(self, pending: PendingStore) -> None:
        self._pending_bytes = max(0, self._pending_bytes - int(pending.tot_kv_size))
        self._delayed_free_blocks = max(
            0, self._delayed_free_blocks - int(pending.gpu_block_count)
        )
        self._async_d2h_inflight = max(0, self._async_d2h_inflight - 1)

    def _try_issue_async_save(
        self,
        request: ReqMeta,
        token_ids,
        store_mask,
        slot_mapping: torch.Tensor,
        skip_leading_tokens: int,
        kvcaches,
    ) -> bool:
        """Issue async D2H. Returns True if pending was registered."""
        engine = self.lmcache_engine
        assert engine is not None
        mask = store_mask
        if mask is not None and getattr(mask, "device", None) is not None:
            if mask.device.type != "cpu":
                mask = mask.detach().to("cpu")
        nblocks = self._gpu_block_count_upper_bound(slot_mapping)
        # Byte watermark uses allocated size after prepare; pre-check request/block
        # caps first, then re-check bytes after issue.
        if self._async_store_over_watermark(request.req_id, 0, nblocks):
            self._async_store_drops += 1
            logger.info(
                "async store drop req=%s reason=%s pending_bytes=%s "
                "pending_reqs=%s delayed_blocks=%s gpu_kv_block_total=%s",
                request.req_id,
                getattr(self, "_async_store_drop_reason", "watermark") or "watermark",
                self._pending_bytes,
                len(self._pending_saves),
                self._delayed_free_blocks,
                getattr(self, "_gpu_kv_block_total", 0),
            )
            return False

        if self.kv_io_mode() == "gds":
            slot_for_store = self._slot_mapping_for_gds_page_direct(slot_mapping)
        else:
            slot_for_store = self._copy_slot_mapping_to_device(slot_mapping)
        t0 = time.perf_counter()
        pending = engine.issue_store_async(
            token_ids,
            mask=store_mask if mask is None else mask,
            kvcaches=kvcaches,
            slot_mapping=slot_for_store,
            offset=skip_leading_tokens,
            transfer_spec=request.disagg_spec,
            request_configs=request.request_configs,
            req_id=request.req_id,
        )
        self._async_t_issue_us += int((time.perf_counter() - t0) * 1e6)
        if pending is None:
            return False
        pending.gpu_block_count = nblocks
        pending.t_issue = time.perf_counter()
        pending.t_kv_quarantine = pending.t_issue
        pending.generation = int(getattr(request, "generation", 0) or 0)
        if getattr(pending, "error", None) is not None:
            self._async_put_failure += 1
            engine.abandon_store(pending, wait_event=True)
            logger.warning(
                "async store dropped after D2H issue failure req=%s", request.req_id
            )
            return False
        if self._async_store_over_watermark(
            request.req_id, int(pending.tot_kv_size), 0
        ):
            self._async_store_drops += 1
            engine.abandon_store(pending, wait_event=True)
            logger.info(
                "async store drop req=%s reason=pending_bytes tot_kv_size=%s",
                request.req_id,
                pending.tot_kv_size,
            )
            return False
        self._account_pending_add(pending)
        logger.debug(
            "async D2H issued req=%s chunks=%s bytes=%s blocks=%s",
            request.req_id,
            len(pending.keys),
            pending.tot_kv_size,
            nblocks,
        )
        return True

    def _event_ready(self, pending: PendingStore) -> bool:
        """CUDA D2H event only. Host put must not gate GPU page release."""
        event = pending.event
        if event is None:
            return True
        query = getattr(event, "query", None)
        if callable(query):
            t0 = time.perf_counter()
            event_ready = bool(query())
            self._async_t_query_us += int((time.perf_counter() - t0) * 1e6)
            return event_ready
        return True

    def _release_gpu_for_pending(self, pending: PendingStore) -> None:
        if getattr(pending, "gpu_released", False):
            self._own_early_free = int(getattr(self, "_own_early_free", 0)) + 1
            return
        now = time.perf_counter()
        t_issue = float(getattr(pending, "t_issue", 0.0) or 0.0)
        if t_issue > 0:
            self._hist_append(
                "_hist_d2h_issue_to_ready_us", int((now - t_issue) * 1e6)
            )
        pending.t_event_ready = now
        pending.gpu_released = True
        self._async_d2h_event_ready += 1
        self._account_pending_remove(pending)
        self._host_pinned_bytes = int(getattr(self, "_host_pinned_bytes", 0)) + int(
            pending.tot_kv_size or 0
        )
        self._hist_append("_hist_ready_to_release_us", 0)

    def _run_put(self, pending: PendingStore) -> None:
        engine = getattr(self, "lmcache_engine", None)
        try:
            if pending.error is not None or getattr(pending, "drop_put", False):
                if engine is not None:
                    engine.abandon_store(pending, wait_event=False)
                self._async_put_failure += int(pending.error is not None)
                return
            if engine is None:
                return
            t0 = time.perf_counter()
            engine.complete_store(pending)
            wait_us = int((time.perf_counter() - t0) * 1e6)
            self._async_t_complete_us += wait_us
            metrics = getattr(engine, "_gds_metrics", None)
            if metrics is not None and hasattr(metrics, "host_put_wait_us"):
                metrics.host_put_wait_us.append(wait_us)
                if len(metrics.host_put_wait_us) > 4096:
                    metrics.host_put_wait_us = metrics.host_put_wait_us[-2048:]
            self._async_put_submit += 1
        except Exception:
            self._async_put_failure += 1
            logger.exception(
                "complete_store failed req=%s; dropping cache", pending.req_id
            )
            if engine is not None:
                engine.abandon_store(pending, wait_event=False)
        finally:
            self._async_put_complete = int(getattr(self, "_async_put_complete", 0)) + 1
            self._async_put_backlog = max(
                0, int(getattr(self, "_async_put_backlog", 0)) - 1
            )
            self._host_pinned_bytes = max(
                0,
                int(getattr(self, "_host_pinned_bytes", 0))
                - int(pending.tot_kv_size or 0),
            )

    def _enqueue_put(self, pending: PendingStore) -> None:
        self._async_put_backlog = int(getattr(self, "_async_put_backlog", 0)) + 1
        engine = getattr(self, "lmcache_engine", None)
        metrics = getattr(engine, "_gds_metrics", None) if engine is not None else None
        if metrics is not None:
            metrics.host_put_backlog = int(self._async_put_backlog)
        executor = getattr(self, "_put_executor", None)
        if executor is None:
            self._run_put(pending)
            return
        executor.submit(self._run_put, pending)

    def _complete_or_drop_pending(self, pending: PendingStore) -> bool:
        """Release GPU on D2H event; put continues on the host queue."""
        self._release_gpu_for_pending(pending)
        self._enqueue_put(pending)
        return True

    def _poll_pending_saves(self) -> None:
        self._async_t_query_us = 0
        engine = self.lmcache_engine
        poll_abandoned = getattr(engine, "poll_abandoned_stores", None) if engine else None
        if callable(poll_abandoned):
            poll_abandoned()
        if not self._pending_saves:
            return
        remaining: dict[str, list[PendingStore]] = {}
        for req_id, pendings in self._pending_saves.items():
            still: list[PendingStore] = []
            for pending in pendings:
                gds_state = getattr(pending, "gds_state", "") or ""
                if getattr(pending, "drop_put", False) or getattr(pending, "error", None) is not None:
                    if not self._event_ready(pending):
                        still.append(pending)
                        continue
                    self._release_gpu_for_pending(pending)
                    if engine is not None:
                        engine.abandon_store(pending, wait_event=False)
                    continue
                if getattr(pending, "gds_fallback_d2h", False) and not (
                    pending.gds_leases or pending.put_future
                ):
                    if not self._event_ready(pending):
                        still.append(pending)
                        continue
                    self._complete_or_drop_pending(pending)
                    continue
                gds_state = getattr(pending, "gds_state", "") or ""
                kv_io_mode = str(
                    getattr(getattr(engine, "config", None), "kv_io_mode", "dram")
                    or "dram"
                )
                is_gds = bool(
                    pending.gds_leases
                    or pending.put_future
                    or gds_state in ("GATHER_SUBMITTED", "PUT_SUBMITTED", "PUT_DONE")
                    or (
                        kv_io_mode == "gds"
                        and (pending.gds_keys or pending.gds_objs)
                    )
                )
                if is_gds:
                    if not self._event_ready(pending):
                        still.append(pending)
                        continue
                    fut = getattr(pending, "put_future", None)
                    done = getattr(fut, "done", None) if fut is not None else None
                    if callable(done) and not done():
                        still.append(pending)
                        continue
                    if pending.memory_objs:
                        self._complete_or_drop_pending(pending)
                        continue
                    if engine is not None:
                        engine.complete_store(pending)
                    if (not pending.submitted) and (
                        pending.put_future or pending.gds_leases
                    ):
                        still.append(pending)
                        continue
                    self._release_gpu_for_pending(pending)
                    continue
                if not self._event_ready(pending):
                    still.append(pending)
                    continue
                self._complete_or_drop_pending(pending)
            if still:
                remaining[req_id] = still
            else:
                self._ready_saves.add(req_id)
        self._pending_saves = remaining

    def _flush_pending_stores(self) -> None:
        """Keep D2H in quarantine until the CUDA event; never sync or early-free."""
        if not self._pending_saves:
            return
        self._async_preempt_flush += 1
        for pendings in self._pending_saves.values():
            for pending in pendings:
                pending.quarantined = True

    def _abandon_req_pendings(self, req_id: str) -> None:
        """Mark abort: skip put, hold GPU pages until D2H event ready."""
        pendings = (getattr(self, "_pending_saves", None) or {}).get(req_id) or []
        for pending in pendings:
            pending.drop_put = True
            pending.quarantined = True

    def _ingest_aborted_req_ids(self, metadata: KVConnectorMetadata) -> None:
        """Abandon D2H only for aborted requests. Empty abort must not drain."""
        aborted = getattr(metadata, "aborted_req_ids", None) or set()
        for req_id in aborted:
            self._abandon_req_pendings(req_id)
            self._abandon_req_loads(req_id)

    def _drain_pending_stores(self) -> None:
        """Shutdown-only: wait and drop every in-flight D2H. Do not object-put."""
        engine = getattr(self, "lmcache_engine", None)
        pending_saves = getattr(self, "_pending_saves", None) or {}
        for _req_id, pendings in list(pending_saves.items()):
            for pending in pendings:
                if engine is not None:
                    t0 = time.perf_counter()
                    engine.abandon_store(pending, wait_event=True)
                    self._async_t_sync_us += int((time.perf_counter() - t0) * 1e6)
                self._account_pending_remove(pending)
        self._pending_saves = {}
        self._ready_saves = set()
        waiting = getattr(self, "_finished_reqs_waiting_for_store", None)
        if waiting is not None:
            waiting.clear()
        self._pending_bytes = 0
        self._delayed_free_blocks = 0
        self._async_d2h_inflight = 0

    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """Issue D2H / record CUDA event. Do not synchronize or wait on put."""

        # Degraded mode (LMCache init failed): no engine to save to / unpin from.
        if self.lmcache_engine is None:
            return

        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)
        self._ingest_aborted_req_ids(connector_metadata)

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            # But still need to unpin the kv caches according to req_id
            # to balance the pin count from contains()
            assert self.lmcache_engine is not None, (
                "LMCacheEngine must be initialized to unpin requests."
            )
            pending_ids = self._pending_req_ids()
            for request in connector_metadata.requests:
                if request.req_id in pending_ids:
                    continue
                self.lmcache_engine.lookup_unpin(request.req_id)

            return

        if self.use_layerwise:
            pending_ids = self._pending_req_ids()
            for request in connector_metadata.requests:
                layerwise_storer = self._layerwise_save_storers.pop(
                    request.req_id, None
                )
                if layerwise_storer is not None:
                    next(layerwise_storer)
                if request.req_id in pending_ids:
                    continue
                # unpin the kv caches according to req_id
                self.lmcache_engine.lookup_unpin(request.req_id)
            return

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        assert self.lmcache_engine is not None

        # Probe decoder cache before store if bidirectional mode is enabled
        bidir_enabled = getattr(self.config, "pd_bidirectional", False)
        if self._async_store_enabled():
            self._async_t_issue_us = 0

        for request in connector_metadata.requests:
            # Keep L1 lookup pin until async H2D event is ready.
            if request.req_id in self._pending_req_ids():
                continue
            if request.req_id in (self._pending_saves or {}):
                continue
            # unpin the kv caches according to req_id
            self.lmcache_engine.lookup_unpin(request.req_id)

            save_spec = request.save_spec
            if (
                save_spec is None or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                continue
            if gds_defer_partial_prefill_store(
                getattr(self.config, "kv_io_mode", "dram"),
                bool(request.is_last_prefill),
            ):
                continue

            token_ids = request.token_ids

            slot_mapping = request.slot_mapping
            assert isinstance(slot_mapping, torch.Tensor)
            if len(slot_mapping) != len(token_ids):
                logger.warning(
                    "Skipping KV save for request %s: slot_mapping/token_ids "
                    "length mismatch (slot_mapping=%d, token_ids=%d). Likely "
                    "an upstream allocation/preemption desync; the engine "
                    "stays alive and only this request's save is dropped.",
                    request.req_id,
                    len(slot_mapping),
                    len(token_ids),
                )
                continue

            skip_leading_tokens = save_spec.skip_leading_tokens
            # shared storage disaggregation will not have a disagg_spec passed in
            if self.kv_role == "kv_producer" and request.disagg_spec:
                skip_leading_tokens = min(
                    skip_leading_tokens, request.disagg_spec.num_transferred_tokens
                )

            if skip_leading_tokens == len(token_ids):
                continue  # skip this request
            # Align to lmcache chunk size
            skip_leading_tokens = (
                skip_leading_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )

            store_mask = torch.ones(len(token_ids), dtype=torch.bool)
            store_mask[:skip_leading_tokens] = False

            logger.debug(
                "Storing KV cache for %d out of %d tokens "
                "(skip_leading_tokens=%d) for request %s",
                len(token_ids) - skip_leading_tokens,
                len(token_ids),
                skip_leading_tokens,
                request.req_id,
            )

            is_last_prefill = request.is_last_prefill
            if is_last_prefill:
                if request.disagg_spec:
                    request.disagg_spec.is_last_prefill = True
            else:
                if not self.enable_blending:
                    token_len = len(token_ids)
                    aligned_token_len = (
                        token_len // self._lmcache_chunk_size * self._lmcache_chunk_size
                    )
                    token_ids = token_ids[:aligned_token_len]
                    store_mask = store_mask[:aligned_token_len]
                    slot_mapping = slot_mapping[:aligned_token_len]

            if skip_leading_tokens >= len(token_ids):
                continue
            if store_mask.numel() == 0 or int(store_mask.sum().item()) == 0:
                continue

            # Probe decoder cache before store
            if bidir_enabled and request.disagg_spec is not None:
                try:
                    self._probe_decoder_cache(request, token_ids)
                except Exception as e:
                    logger.warning(
                        "Bidirectional NIXL cache probe failed for %s: %s",
                        request.req_id,
                        e,
                    )

            if self._async_store_enabled():
                if not self._try_issue_async_save(
                    request,
                    token_ids,
                    store_mask,
                    slot_mapping,
                    skip_leading_tokens,
                    kvcaches,
                ):
                    continue
            else:
                if self.kv_io_mode() == "gds":
                    slot_for_store = self._slot_mapping_for_gds_page_direct(
                        slot_mapping
                    )
                else:
                    slot_for_store = self._copy_slot_mapping_to_device(slot_mapping)
                self.lmcache_engine.store(
                    token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_for_store,
                    offset=skip_leading_tokens,
                    transfer_spec=request.disagg_spec,
                    request_configs=request.request_configs,
                    req_id=request.req_id,
                )

            # Probe decoder cache after store
            if (
                bidir_enabled
                and request.disagg_spec is not None
                and request.disagg_spec.receiver_query_port is not None
            ):
                try:
                    self._probe_decoder_cache(request, token_ids)
                except Exception as e:
                    logger.warning(
                        "Bidirectional NIXL cache probe failed for %s: %s",
                        request.req_id,
                        e,
                    )

            # Update skip_leading_tokens only on last rank to ensure
            # each PP stage stores its own KV cache
            if get_pp_group().is_last_rank:
                skip_to = len(token_ids)
                if self._async_store_enabled():
                    queued = self._pending_saves.get(request.req_id) or []
                    pending = queued[-1] if queued else None
                    if pending is None:
                        continue
                    ends = (pending.kwargs_keepalive or {}).get("ends") or []
                    if ends:
                        skip_to = int(ends[-1])
                    elif pending.tot_token_num:
                        skip_to = int(pending.tot_token_num)
                save_spec.skip_leading_tokens = skip_to
                tracker = (getattr(self, "_request_trackers", {}) or {}).get(
                    request.req_id
                )
                if tracker is not None:
                    tracker.num_saved_tokens = skip_to
                if request.disagg_spec:
                    request.disagg_spec.num_transferred_tokens = skip_to

        if self._async_store_enabled() and str(
            getattr(self.config, "kv_io_mode", "dram") or "dram"
        ).lower() == "gds":
            pass  # non-blocking: no WAIT_LEASE drain on the GPU worker thread

    def _drain_gds_wait_lease(self) -> None:
        """Retry WAIT_LEASE stores on this GPU worker thread until slots free.

        Scheduler may go idle before get_finished polls again; CUDA gather
        must not run on a background thread.
        """
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            waiting = any(
                getattr(pending, "gds_state", "") == "WAIT_LEASE"
                for plist in (self._pending_saves or {}).values()
                for pending in plist
            )
            if not waiting:
                return
            self._poll_pending_saves()
            time.sleep(0.001)
        logger.warning("GDS WAIT_LEASE drain timed out after 120s")

    def _probe_decoder_cache(self, request: ReqMeta, token_ids: list[int]) -> None:
        """Query the decoder's cache to check which blocks are already cached.

        This is the bidirectional NIXL cache probe: the prefiller queries the
        decoder via ZMQ to find out which KV blocks are already in the
        decoder's GPU memory. This validates the cache query channel works
        E2E through the real inference path.

        In the future, this information can be used to skip prefill
        computation for cached blocks.
        """
        sm = self.lmcache_engine.storage_manager  # type: ignore[union-attr]
        if sm is None or sm.allocator_backend is None:
            return
        pd_backend = sm.allocator_backend
        if not hasattr(pd_backend, "query_remote_cache"):
            return
        if not hasattr(pd_backend, "cache_query_sockets"):
            return

        # Get query port from LMCache config (pd_peer_query_port)
        query_ports = self.config.pd_peer_query_port
        if query_ports is None:
            return

        # Build cache keys using the token database's process_tokens
        td = self.lmcache_engine.token_database  # type: ignore[union-attr]
        if td is None:
            return

        chunk_keys = []
        for _start, _end, key in td.process_tokens(
            tokens=token_ids, mask=None, make_key=True
        ):
            chunk_keys.append(key)

        if not chunk_keys:
            return

        # Build receiver_id from disagg_spec
        disagg = request.disagg_spec
        init_port = disagg.receiver_init_port  # type: ignore[union-attr]
        if isinstance(init_port, list):
            init_port = init_port[pd_backend.tp_rank]  # type: ignore[union-attr]
        receiver_id = disagg.receiver_host + str(init_port)  # type: ignore[union-attr]

        # Ensure peer and cache query connections
        alloc_port = disagg.receiver_alloc_port  # type: ignore[union-attr]
        if isinstance(alloc_port, list):
            alloc_port = alloc_port[pd_backend.tp_rank]  # type: ignore[union-attr]
        query_port = query_ports[pd_backend.tp_rank]  # type: ignore[union-attr]

        pd_backend._ensure_peer_connection(  # type: ignore[union-attr]
            receiver_id=receiver_id,
            receiver_host=disagg.receiver_host,  # type: ignore[union-attr]
            receiver_init_port=init_port,
            receiver_alloc_port=alloc_port,
        )
        pd_backend._ensure_cache_query_connection(  # type: ignore[union-attr]
            receiver_id=receiver_id,
            receiver_host=disagg.receiver_host,  # type: ignore[union-attr]
            receiver_query_port=query_port,
        )

        # Query decoder cache
        cache_resp = pd_backend.query_remote_cache(receiver_id, chunk_keys)

        logger.info(
            "Bidirectional NIXL cache probe: req=%s, "
            "queried %d chunks, decoder has %d cached "
            "(%.0f%% hit rate)",
            request.req_id,
            len(chunk_keys),
            len(cache_resp.cached_keys),
            100.0 * len(cache_resp.cached_keys) / len(chunk_keys) if chunk_keys else 0,
        )

    def kv_io_mode(self) -> str:
        cfg = getattr(self, "config", None)
        extra = getattr(cfg, "extra_config", None) or {}
        raw = extra.get("kv.io_mode", getattr(cfg, "kv_io_mode", "dram"))
        mode = str(raw or "dram").strip().lower()
        if mode not in ("dram", "gds"):
            raise ValueError("kv.io_mode must be dram|gds, got %r" % raw)
        return mode

    def _hbm_prefetch_enabled(self) -> bool:
        if bool(getattr(self, "use_layerwise", False)):
            return False
        cfg = getattr(self, "config", None)
        if cfg is None:
            return False
        if bool(getattr(cfg, "use_layerwise", False)):
            return False
        return bool(getattr(cfg, "hbm_prefetch_enable", False))

    @property
    def hbm_prefetch_enabled(self) -> bool:
        return self._hbm_prefetch_enabled()

    def _refresh_cuda_free_bytes(self) -> Optional[int]:
        getter = getattr(self, "_cuda_mem_get_info", None)
        try:
            if callable(getter):
                free, _total = getter()
                self._last_cuda_free_bytes = int(free)
                return int(free)
            if torch.cuda.is_available():
                free, _total = torch.cuda.mem_get_info()
                self._last_cuda_free_bytes = int(free)
                return int(free)
        except Exception:
            pass
        cached = getattr(self, "_last_cuda_free_bytes", None)
        if cached is not None:
            return int(cached)
        return None

    def _pending_req_ids(self) -> set[str]:
        ids: set[str] = set()
        for key in getattr(self, "_pending_loads", {}) or {}:
            ids.add(key[0] if isinstance(key, tuple) else str(key))
        return ids

    def _pending_key(self, req_id: str, generation: int = 0) -> tuple[str, int]:
        return (str(req_id), int(generation or 0))

    def _load_req_id(self, key) -> str:
        if isinstance(key, tuple):
            return str(key[0])
        return str(key)

    def _load_generation(self, key, pending=None) -> int:
        if isinstance(key, tuple) and len(key) > 1:
            return int(key[1] or 0)
        return int(getattr(pending, "generation", 0) or 0)

    def _xfer_record(
        self,
        pending: PendingLoad,
        outcome: str,
    ) -> KVConnectorXferRecord:
        return KVConnectorXferRecord(
            request_id=str(pending.req_id),
            generation=int(getattr(pending, "generation", 0) or 0),
            block_ids=tuple(getattr(pending, "block_ids", ()) or ()),
            outcome=outcome,
        )

    def _start_or_progress_async_load(
        self,
        request,
        tokens,
        token_mask,
        retrieve_kwargs: dict,
        generation: int,
        block_ids: tuple[int, ...],
    ) -> None:
        engine = self.lmcache_engine
        if engine is None:
            return
        key = self._pending_key(request.req_id, generation)
        loads = getattr(self, "_pending_loads", None)
        if loads is None:
            self._pending_loads = {}
            loads = self._pending_loads
        for old_key in list(loads):
            if self._load_req_id(old_key) != str(request.req_id):
                continue
            old_gen = self._load_generation(old_key, loads[old_key])
            if old_gen != int(generation):
                self._own_stale_event = int(getattr(self, "_own_stale_event", 0)) + 1
                stale = loads.pop(old_key, None)
                if stale is not None and engine is not None:
                    engine.abandon_load(stale, wait_event=False)
        pending = loads.get(key)
        if pending is not None:
            self._own_duplicate_generation = int(
                getattr(self, "_own_duplicate_generation", 0)
            ) + 1
            exp = tuple(getattr(pending, "block_ids", ()) or ())
            if exp and block_ids and exp != tuple(block_ids):
                self._own_block_mismatch = int(
                    getattr(self, "_own_block_mismatch", 0)
                ) + 1
                loads.pop(key, None)
                if engine is not None:
                    engine.abandon_load(pending, wait_event=False)
                return
        if pending is None:
            submit = getattr(engine, "submit_retrieve_gather", None)
            if not callable(submit):
                if self.kv_io_mode() == "gds":
                    logger.error(
                        "GDS page-direct load requires submit_retrieve_gather; "
                        "refusing generic retrieve fallback with CPU "
                        "slot_mapping req=%s",
                        request.req_id,
                    )
                    self._gds_retrieve_submit_missing = int(
                        getattr(self, "_gds_retrieve_submit_missing", 0)
                    ) + 1
                    cancelled = getattr(self, "_cancelled_recving", None)
                    if cancelled is None:
                        self._cancelled_recving = set()
                        cancelled = self._cancelled_recving
                    cancelled.add(str(request.req_id))
                    self._prefetch_cancelled = int(
                        getattr(self, "_prefetch_cancelled", 0)
                    ) + 1
                    unpin = getattr(engine, "lookup_unpin", None)
                    if callable(unpin):
                        unpin(request.req_id)
                    return
                retrieve_kwargs["sync"] = False
                engine.retrieve(tokens, token_mask, **retrieve_kwargs)
                pending = getattr(engine, "_last_pending_load", None)
                if pending is None:
                    pending = PendingLoad(
                        req_id=request.req_id,
                        chunks=[],
                        event=ReadyStoreEvent(),
                        generation=generation,
                        block_ids=block_ids,
                        h2d_issued=True,
                    )
                else:
                    pending.generation = generation
                    pending.block_ids = block_ids
                    pending.h2d_issued = True
                engine._last_pending_load = None
                self._pending_loads[key] = pending
                self._prefetch_issue_count += 1
                self._prefetch_inflight = len(self._pending_loads)
                return
            pending = submit(
                tokens,
                token_mask,
                generation=generation,
                block_ids=block_ids,
                **retrieve_kwargs,
            )
            self._pending_loads[key] = pending
            self._prefetch_issue_count += 1
        issue = getattr(engine, "try_issue_h2d", None)
        if callable(issue):
            status = issue(pending)
            if status == "error":
                self._cancelled_recving.add(str(request.req_id))
                self._xfer_cancelled.append(self._xfer_record(pending, "error"))
                logger.warning(
                    "GDS/async load error req=%s err=%s",
                    request.req_id,
                    pending.error,
                )
                engine.abandon_load(pending, wait_event=False)
                self._pending_loads.pop(key, None)
                self._prefetch_cancelled += 1
            elif status in ("empty", "pending_gather"):
                if not getattr(pending, "_gather_notified", False):
                    pending._gather_notified = True
                    self._prefetch_pending_gather = int(
                        getattr(self, "_prefetch_pending_gather", 0)
                    ) + 1
                    self._xfer_recving.append(
                        self._xfer_record(
                            pending,
                            "pending_gather" if status == "pending_gather" else "gather_empty",
                        )
                    )
            elif status == "capacity_wait":
                self._prefetch_capacity_wait = int(
                    getattr(self, "_prefetch_capacity_wait", 0)
                ) + 1
                if not getattr(pending, "_cap_notified", False):
                    pending._cap_notified = True
                    self._xfer_recving.append(
                        self._xfer_record(pending, "capacity_wait")
                    )
            elif status == "issued" and not getattr(pending, "_h2d_notified", False):
                pending._h2d_notified = True
                self._xfer_recving.append(self._xfer_record(pending, "h2d_issued"))
        self._prefetch_inflight = len(self._pending_loads)

    def _event_ready_any(self, event) -> bool:
        if event is None:
            return True
        query = getattr(event, "query", None)
        if callable(query):
            return bool(query())
        return True

    def _poll_pending_loads(self) -> set[str]:
        finished: set[str] = set()
        remaining: dict[str, PendingLoad] = {}
        engine = getattr(self, "lmcache_engine", None)
        poll_abandoned = getattr(engine, "poll_abandoned_loads", None) if engine else None
        if callable(poll_abandoned):
            for abandoned in poll_abandoned():
                unpin = getattr(engine, "lookup_unpin", None)
                if callable(unpin):
                    unpin(abandoned.req_id)
                if getattr(abandoned, "_skip_cancel_count", False):
                    continue
                self._xfer_cancelled.append(self._xfer_record(abandoned, "cancel"))
                self._cancelled_recving.add(str(abandoned.req_id))
                self._prefetch_cancelled = int(
                    getattr(self, "_prefetch_cancelled", 0)
                ) + 1
        for key, pending in list(
            (getattr(self, "_pending_loads", {}) or {}).items()
        ):
            req_id = pending.req_id
            if pending.error is not None:
                if engine is not None:
                    engine.abandon_load(pending, wait_event=False)
                self._xfer_cancelled.append(self._xfer_record(pending, "error"))
                self._cancelled_recving.add(str(req_id))
                logger.warning(
                    "GDS/async load error req=%s err=%s",
                    req_id,
                    pending.error,
                )
                self._prefetch_cancelled = int(
                    getattr(self, "_prefetch_cancelled", 0)
                ) + 1
                finished.add(req_id)
                continue
            issued = bool(getattr(pending, "h2d_issued", False))
            state = str(getattr(pending, "xfer_state", "") or "")
            if not issued and state not in (
                "H2D_IN_FLIGHT",
                "GDS_IN_FLIGHT",
                "GPU_SCATTER_IN_FLIGHT",
                "COMPLETED",
                "RELEASED",
            ):
                issue = getattr(engine, "try_issue_h2d", None) if engine else None
                if callable(issue):
                    status = issue(pending)
                    if status == "error":
                        engine.abandon_load(pending, wait_event=False)
                        self._xfer_cancelled.append(self._xfer_record(pending, "error"))
                        self._cancelled_recving.add(str(req_id))
                        logger.warning(
                            "GDS/async load error req=%s err=%s",
                            req_id,
                            pending.error,
                        )
                        self._prefetch_cancelled = int(
                            getattr(self, "_prefetch_cancelled", 0)
                        ) + 1
                        finished.add(req_id)
                        continue
                    if status in ("empty", "pending_gather"):
                        remaining[key] = pending
                        if not getattr(pending, "_gather_notified", False):
                            pending._gather_notified = True
                            self._prefetch_pending_gather = int(
                                getattr(self, "_prefetch_pending_gather", 0)
                            ) + 1
                            self._xfer_recving.append(
                                self._xfer_record(
                                    pending,
                                    "pending_gather"
                                    if status == "pending_gather"
                                    else "gather_empty",
                                )
                            )
                        continue
                    if status == "capacity_wait":
                        remaining[key] = pending
                        self._h2d_fifo_depth = max(
                            int(getattr(self, "_h2d_fifo_depth", 0)),
                            len(remaining) + 1,
                        )
                        self._prefetch_capacity_wait = int(
                            getattr(self, "_prefetch_capacity_wait", 0)
                        ) + 1
                        if not getattr(pending, "_cap_notified", False):
                            pending._cap_notified = True
                            self._xfer_recving.append(
                                self._xfer_record(pending, "capacity_wait")
                            )
                        continue
                    if status == "issued" and not getattr(pending, "_h2d_notified", False):
                        pending._h2d_notified = True
                        self._xfer_recving.append(self._xfer_record(pending, "h2d_issued"))
                else:
                    remaining[key] = pending
                    continue
            if not getattr(pending, "h2d_issued", False):
                remaining[key] = pending
                continue
            if not self._event_ready_any(pending.event):
                remaining[key] = pending
                continue
            key_gen = self._load_generation(key, pending)
            if int(getattr(pending, "generation", 0) or 0) != int(key_gen):
                self._own_stale_event = int(getattr(self, "_own_stale_event", 0)) + 1
                if engine is not None:
                    engine.abandon_load(pending, wait_event=False)
                continue
            t_h2d = float(getattr(pending, "t_h2d_issue", 0.0) or 0.0)
            if t_h2d > 0:
                self._hist_append(
                    "_hist_h2d_issue_to_ready_us",
                    int((time.perf_counter() - t_h2d) * 1e6),
                )
            token_mask = (pending.kwargs_keepalive or {}).get("_gather_mask")
            slot_mapping = (pending.kwargs_keepalive or {}).get("slot_mapping")
            ret_mask = pending.ret_mask
            if token_mask is not None and ret_mask is not None and slot_mapping is not None:
                missing = self.record_failed_blocks(
                    req_id, token_mask, ret_mask, slot_mapping
                )
                if missing:
                    self._invalid_block_ids.update(missing)
                if missing and not bool(torch.any(ret_mask)):
                    self._cancelled_recving.add(str(req_id))
            if engine is not None:
                engine.complete_load(pending)
                unpin = getattr(engine, "lookup_unpin", None)
                if callable(unpin):
                    unpin(req_id)
            else:
                pending.completed = True
            finished.add(req_id)
            self._xfer_recving.append(self._xfer_record(pending, "success"))
            self._prefetch_event_ready_count = int(
                getattr(self, "_prefetch_event_ready_count", 0)
            ) + 1
        self._pending_loads = remaining
        self._prefetch_inflight = len(remaining)
        return finished

    def _abandon_req_loads(self, req_id: str, *, count_cancel: bool = True) -> None:
        pending_map = getattr(self, "_pending_loads", None) or {}
        keys = [
            k
            for k in list(pending_map)
            if self._load_req_id(k) == str(req_id)
        ]
        engine = getattr(self, "lmcache_engine", None)
        for key in keys:
            pending = pending_map.pop(key, None)
            if pending is None:
                continue
            if not count_cancel:
                setattr(pending, "_skip_cancel_count", True)
            if engine is not None:
                engine.abandon_load(pending, wait_event=False)
            if count_cancel:
                self._xfer_cancelled.append(self._xfer_record(pending, "cancel"))
                self._cancelled_recving.add(str(req_id))
                self._prefetch_cancelled = int(
                    getattr(self, "_prefetch_cancelled", 0)
                ) + 1
        self._prefetch_inflight = len(pending_map)

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        metadata = None
        try:
            metadata = self._parent._get_connector_metadata()
        except Exception:
            pass
        if metadata is not None:
            self._ingest_aborted_req_ids(metadata)

        recving = self._poll_pending_loads() or set()
        cancelled = self.take_cancelled_recving()
        if cancelled:
            recving |= cancelled
        recving = recving or None

        sending: Optional[set[str]] = None
        if self._async_store_enabled():
            self._poll_pending_saves()
            sending_set: set[str] = set()
            waiting = getattr(self, "_finished_reqs_waiting_for_store", None)
            if waiting is None:
                self._finished_reqs_waiting_for_store = set()
                waiting = self._finished_reqs_waiting_for_store
            for req_id in finished_req_ids:
                if req_id in self._pending_saves:
                    # D2H still in flight; keep GPU pages until the event is ready.
                    waiting.add(req_id)
                    continue
                sending_set.add(req_id)
                waiting.discard(req_id)
                self._ready_saves.discard(req_id)

            for req_id in list(waiting):
                if req_id in self._pending_saves:
                    continue
                sending_set.add(req_id)
                waiting.discard(req_id)
                self._ready_saves.discard(req_id)
            sending = sending_set or None

        logger.debug(
            "get_finished sending=%s recving=%s pending_saves=%s pending_loads=%s",
            sending,
            recving,
            list(getattr(self, "_pending_saves", {}) or {}),
            list(getattr(self, "_pending_loads", {}) or {}),
        )
        return sending, recving

    def take_cancelled_recving(self) -> set[str] | None:
        cancelled = getattr(self, "_cancelled_recving", None)
        if not cancelled:
            return None
        out = set(cancelled)
        cancelled.clear()
        return out or None

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        """Flush in-flight D2H / abandon H2D before preempted GPU pages reuse."""
        if getattr(self, "_role", None) != KVConnectorRole.WORKER:
            return
        self._ingest_aborted_req_ids(kv_connector_metadata)
        need_flush = (
            isinstance(kv_connector_metadata, LMCacheConnectorMetadata)
            and kv_connector_metadata.need_flush_before_forward
        )
        if not need_flush:
            return
        for req_id in list(self._pending_req_ids()):
            self._abandon_req_loads(req_id)
        if self._async_store_enabled():
            self._flush_pending_stores()

    def get_block_ids_with_load_errors(self) -> set[int]:
        invalid_blocks = self._invalid_block_ids.copy()
        self._invalid_block_ids.clear()
        return invalid_blocks

    @_lmcache_nvtx_annotate
    def shutdown(self):
        """Shutdown the connector by delegating to LMCacheManager."""
        logger.info("Starting LMCacheConnector shutdown...")
        try:
            self._drain_pending_stores()
            for req_id in list(self._pending_req_ids()):
                self._abandon_req_loads(req_id)
        except Exception:
            logger.exception("drain pending async stores during shutdown failed")
        executor = getattr(self, "_put_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
            self._put_executor = None
        self._manager.stop_services()

    ###################
    # Scheduler side APIs
    ####################

    @_lmcache_nvtx_annotate
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        """
        Check for external KV cache hit.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        # Ignore DP attention mock requests
        if request.request_id.startswith("mock_req"):
            return 0, False
        if request.request_id in getattr(self, "_hbm_cancelled_req_ids", set()):
            return 0, False
        # to handle preempted requests, we want `get_num_new_matched_tokens` to be
        # idempotent under the condition that `update_state_after_alloc` is NOT called
        # then the two side-effects that must be idempotent are:
        # 1. lookup_client caches a result
        #     uncached in `update_state_after_alloc` if this request can be scheduled
        # 2. cache engine will pin the KV caches for the request
        #     unpinned in `wait_for_save` if this request can be scheduled
        if self.kv_role == "kv_producer" and not hasattr(
            self.lookup_client, "supports_producer_reuse"
        ):
            return 0, False

        req_id = request.request_id

        # Degraded mode (LMCache init failed): no lookup client is available, so
        # report no external hits and let vLLM recompute instead of asserting and
        # crashing EngineCore during scheduling.
        if self.lookup_client is None:
            return 0, False

        if self.kv_io_mode() == "dram":
            lookup_cache = getattr(self.lookup_client, "lookup_cache", None)
            cached = lookup_cache(lookup_id=req_id) if callable(lookup_cache) else -1
            if cached is None:
                return None, True
            if cached == -1 or int(cached) <= 0:
                return 0, False
            num_external_hit_tokens = int(cached)
        else:
            num_external_hit_tokens = self._poll_or_start_lookup(request)
            if num_external_hit_tokens is None:
                logger.debug(
                    "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                    "LMCache hit tokens: None.",
                    req_id,
                    request.num_tokens,
                    num_computed_tokens,
                )
                if self._gds_lookup_should_fallback(req_id):
                    return 0, False
                return None, True
            if int(num_external_hit_tokens) > 0:
                self._gds_lookup_note_late_hit(req_id)

        # When prompt length is divisible by the block size and all
        # blocks are cached, we need to recompute the last token.
        # This will be removed in the future if vLLM's scheduler provides
        # a better support for this case.
        need_to_allocate = num_external_hit_tokens - num_computed_tokens

        # In, full-prompt-hit case, we need to recompute the last token
        if num_external_hit_tokens == request.num_tokens:
            need_to_allocate -= 1

        # Check if hit tokens meet the minimum for retrieve
        # If below minimum, skip retrieve but still record hit tokens
        # for skip_leading_tokens to avoid re-storing existing chunks
        min_retrieve = self.config.min_retrieve_tokens
        below_min_retrieve = min_retrieve > 0 and need_to_allocate < min_retrieve

        if below_min_retrieve:
            logger.info(
                "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                "LMCache hit tokens: %d, but need to load: %d < min_retrieve %d, "
                "skip retrieve but record for save skip",
                req_id,
                request.num_tokens,
                num_computed_tokens,
                num_external_hit_tokens,
                max(need_to_allocate, 0),
                min_retrieve,
            )
        else:
            logger.info(
                "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                "LMCache hit tokens: %d, need to load: %d, "
                "lmcache_lookup_tokens_uncapped: %d, h2d_capped_tokens: %d",
                req_id,
                request.num_tokens,
                num_computed_tokens,
                num_external_hit_tokens,
                max(need_to_allocate, 0),
                int(num_external_hit_tokens),
                int(num_external_hit_tokens),
            )

        # DRAM-only chunked loading. GDS must report the full confirmed prefix;
        # stock vLLM waits on HBM instead of truncating to 256 tokens.
        capped_lmcache_tokens = num_external_hit_tokens
        if (
            self.kv_io_mode() != "gds"
            and self._max_tokens_per_load > 0
            and need_to_allocate > self._max_tokens_per_load
        ):
            # Align cap to LMCache chunk boundaries so that the
            # retrieve path receives chunk-aligned token ranges.
            cap = (
                self._max_tokens_per_load
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            need_to_allocate = cap
            # Align capped_lmcache_tokens to chunk boundary so that
            # session.get_hashes() receives a chunk-aligned end value.
            capped_lmcache_tokens = (
                (num_computed_tokens + cap)
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            logger.debug(
                "Reqid: %s, Chunked KV loading: capped from "
                "%d to %d external tokens "
                "(max_tokens_per_load=%d)",
                req_id,
                num_external_hit_tokens - num_computed_tokens,
                need_to_allocate,
                self._max_tokens_per_load,
            )

        self.load_specs[req_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            lmcache_cached_tokens=capped_lmcache_tokens,
            can_load=False,
        )

        if below_min_retrieve or need_to_allocate <= 0:
            return 0, False

        # TODO: Align to vLLM block size. Should test whether it can be removed
        # need_to_allocate = need_to_allocate // self._block_size * \
        #        self._block_size

        return need_to_allocate, (
            self.kv_io_mode() in ("dram", "gds") or self._hbm_prefetch_enabled()
        )

    def _lookup_token_ids_and_configs(
        self, request: "Request"
    ) -> tuple[Any, Optional[dict]]:
        """Token ids + request_configs for lookup(); shared by both scheduler entries."""
        token_ids = request.all_token_ids
        mm_hashes, mm_positions = extract_mm_features(request)
        if mm_hashes and mm_positions:
            token_ids = torch.tensor(request.prompt_token_ids)
            apply_mm_hashes_to_token_ids(token_ids, mm_hashes, mm_positions)
            token_ids = token_ids.tolist()
        request_configs = None
        if hasattr(request, "sampling_params"):
            request_configs = extract_request_configs(request.sampling_params)
        if self.skip_last_n_tokens > 0:
            token_ids = token_ids[: -self.skip_last_n_tokens]
        return token_ids, request_configs

    def _gds_lookup_should_fallback(self, req_id: str) -> bool:
        """Return True when a GDS in-flight lookup has exceeded the deferral bound.

        The scheduler must not sleep. After timeout the request is scheduled as
        a miss while lookup continues in the background.

        Args:
            req_id: vLLM request id.

        Returns:
            True if this call should return ``(0, False)`` instead of waiting.
        """
        if self.kv_io_mode() != "gds":
            return False
        extra = getattr(self.config, "extra_config", None) or {}
        raw = extra.get(
            "kv.gds_lookup_deferral_timeout_ms",
            getattr(self.config, "gds_lookup_deferral_timeout_ms", 50),
        )
        timeout_ms = int(raw or 0)
        if timeout_ms <= 0:
            return False
        now = time.monotonic()
        started = getattr(self, "_gds_lookup_defer_started", None)
        if started is None:
            self._gds_lookup_defer_started = {}
            started = self._gds_lookup_defer_started
        counts = getattr(self, "_gds_lookup_defer_counts", None)
        if counts is None:
            self._gds_lookup_defer_counts = {}
            counts = self._gds_lookup_defer_counts
        started.setdefault(req_id, now)
        counts[req_id] = int(counts.get(req_id, 0) or 0) + 1
        elapsed_ms = (now - float(started[req_id])) * 1000.0
        if elapsed_ms < timeout_ms:
            return False
        self._gds_lookup_fallback_count = int(
            getattr(self, "_gds_lookup_fallback_count", 0) or 0
        ) + 1
        fallback_ids = getattr(self, "_gds_lookup_fallback_ids", None)
        if fallback_ids is None:
            self._gds_lookup_fallback_ids = set()
            fallback_ids = self._gds_lookup_fallback_ids
        fallback_ids.add(req_id)
        logger.info(
            "GDS lookup deferral timeout req=%s elapsed_ms=%.1f count=%s "
            "fallback=miss",
            req_id,
            elapsed_ms,
            counts[req_id],
        )
        return True

    def _gds_lookup_note_late_hit(self, req_id: str) -> None:
        """Count a hit that arrived after the request was scheduled as a miss."""
        fallback_ids = getattr(self, "_gds_lookup_fallback_ids", None)
        if not fallback_ids or req_id not in fallback_ids:
            started = getattr(self, "_gds_lookup_defer_started", None)
            if isinstance(started, dict):
                started.pop(req_id, None)
            return
        fallback_ids.discard(req_id)
        self._gds_lookup_late_hit_count = int(
            getattr(self, "_gds_lookup_late_hit_count", 0) or 0
        ) + 1
        logger.info(
            "GDS lookup late-hit req=%s late_hit=%s",
            req_id,
            self._gds_lookup_late_hit_count,
        )

    def _poll_or_start_lookup(self, request: "Request") -> Optional[int]:
        """Start or poll async lookup. None = in-flight; int >= 0 is cached hits.

        lookup_cache: -1 starts/restarts lookup(); None = in-flight; int>=0 is
        cached. A cached miss (0) may return -1 after relookup_min_interval_ms
        when relookup_on_schedule is on. Never treat -1 as a 0-hit miss.
        """
        req_id = request.request_id
        lookup_cache = getattr(self.lookup_client, "lookup_cache", None)
        if callable(lookup_cache):
            cached = lookup_cache(lookup_id=req_id)
            if cached != -1:
                logger.debug(
                    "Found %s hit tokens for request %s in the lookup cache.",
                    cached,
                    req_id,
                )
                return cached
            logger.debug("Looking up cache for the first time for request %s!", req_id)
        self._requests_priority[req_id] = getattr(request, "priority", 0)
        token_ids, request_configs = self._lookup_token_ids_and_configs(request)
        return self.lookup_client.lookup(
            token_ids,
            lookup_id=req_id,
            request_configs=request_configs,
        )

    def cancel_and_clear_lookup(self, request_id: str) -> None:
        lookup = getattr(self, "lookup_client", None)
        if lookup is None:
            return
        cancel_lookup = getattr(lookup, "cancel_lookup", None)
        if callable(cancel_lookup):
            cancel_lookup(request_id)
        clear = getattr(lookup, "clear_lookup_status", None)
        if callable(clear):
            clear(request_id)

    @_lmcache_nvtx_annotate
    def update_state_after_alloc(
        self, request: "Request", blocks: Any = None, num_external_tokens: int = 0
    ):
        """
        Update KVConnector state after temporary buffer alloc.

        ``blocks`` is optional for older wrappers; prefetch requires it to
        record GPU block ids for retrieve slot_mapping.
        """
        if blocks is not None and isinstance(blocks, int) and num_external_tokens == 0:
            # Old signature: update_state_after_alloc(request, num_external_tokens)
            num_external_tokens = blocks
            blocks = None

        # Degraded mode (LMCache init failed): there is no lookup client to clear
        # and no load spec was recorded, so nothing to do.
        if self.lookup_client is None:
            return

        tracker = (getattr(self, "_request_trackers", {}) or {}).get(
            request.request_id
        )
        is_second = tracker is not None and getattr(tracker, "load_state", "") in (
            "waiting_for_load",
            "ready",
        )
        if not is_second:
            lookup_cache = getattr(self.lookup_client, "lookup_cache", None)
            cached = -1
            if callable(lookup_cache):
                cached = lookup_cache(lookup_id=request.request_id)
            if cached == -1:
                self.lookup_client.clear_lookup_status(request.request_id)
            else:
                logger.info(
                    "dram_op request=%s phase=keep_lookup cached=%s",
                    request.request_id,
                    cached,
                )

        kv_transfer_params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )

        if kv_transfer_params is not None and "disagg_spec" in kv_transfer_params:
            req_disagg_spec = kv_transfer_params["disagg_spec"]

            receiver_id = req_disagg_spec["receiver_host"] + str(
                req_disagg_spec["receiver_init_port"]
            )

            disagg_spec = DisaggSpec(
                req_id=req_disagg_spec["req_id"],
                receiver_id=receiver_id,
                receiver_host=req_disagg_spec["receiver_host"],
                receiver_init_port=req_disagg_spec["receiver_init_port"],
                receiver_alloc_port=req_disagg_spec["receiver_alloc_port"],
                receiver_query_port=req_disagg_spec.get("receiver_query_port"),
            )

            tmp_disagg_tracker[request.request_id] = disagg_spec
        self._unfinished_requests[request.request_id] = request

        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            return

        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = False
            return

        spec = self.load_specs[request.request_id]
        recalc_last = (
            1
            if spec.lmcache_cached_tokens == request.num_tokens
            else 0
        )
        expected = (
            spec.lmcache_cached_tokens - spec.vllm_cached_tokens - recalc_last
        )
        assert num_external_tokens == expected, (
            f"Mismatch in tokens to load: {num_external_tokens} vs "
            f"{spec.lmcache_cached_tokens} "
            "(tokens in lmcache) - "
            f"{spec.vllm_cached_tokens} "
            "(tokens in vllm) - "
            f"{recalc_last} "
            "(full lmcache hits subtracts last token to recalculate logits)"
            f" for request {request.request_id}"
        )

        self.load_specs[request.request_id].can_load = True
        if tracker is None:
            tracker = self._tracker_from_request(request, blocks)
            self._request_trackers[request.request_id] = tracker
        else:
            tracker.append_allocated_blocks(blocks)
        if not is_second and (
            self._hbm_prefetch_enabled() or self.kv_io_mode() in ("dram", "gds")
        ):
            tracker.load_state = "waiting_for_load"

    def _tracker_from_request(self, request: "Request", blocks: Any) -> RequestTracker:
        load_spec = self.load_specs.get(request.request_id)
        lmcache_cached = load_spec.lmcache_cached_tokens if load_spec is not None else 0
        ntok = max(int(lmcache_cached), 1)
        all_ids = list(getattr(request, "all_token_ids", []) or [])
        token_ids = all_ids[:ntok] if all_ids else list(range(ntok))
        tracker = RequestTracker(
            req_id=request.request_id,
            prompt_len=int(getattr(request, "num_prompt_tokens", ntok) or ntok),
            token_ids=token_ids,
            allocated_block_ids=[],
            num_saved_tokens=lmcache_cached,
            skip_save=True,
            num_lmcache_cached_tokens=lmcache_cached,
            load_state="idle",
        )
        tracker.append_allocated_blocks(blocks)
        return tracker

    def _inc_restore_stat(self, name: str, n: int = 1) -> None:
        setattr(self, name, int(getattr(self, name, 0) or 0) + n)

    def _log_load_spec(
        self, req_id: str, reason: str, spec: Optional[LoadSpec]
    ) -> None:
        generation = 0 if spec is None else int(getattr(spec, "generation", 0) or 0)
        issued = 0 if spec is None else int(bool(getattr(spec, "prefetch_issued", False)))
        can_load = 0 if spec is None else int(bool(spec.can_load))
        logger.info(
            "LOAD_SPEC req_id=%s generation=%s reason=%s prefetch_issued=%s can_load=%s",
            req_id,
            generation,
            reason,
            issued,
            can_load,
        )

    def _mark_hbm_load_emitted(self, req_id: str, reason: str) -> None:
        spec = self.load_specs.get(req_id)
        if spec is None:
            return
        # Wire metadata already copied this generation; keep it for matching.
        spec.issued_generation = int(spec.generation or 0)
        spec.prefetch_issued = True
        if reason == "prefetch":
            self._inc_restore_stat("_load_spec_emit_prefetch")
        elif reason == "new":
            self._inc_restore_stat("_load_spec_emit_new")
        elif reason == "resumed":
            self._inc_restore_stat("_load_spec_emit_resumed")
        self._log_load_spec(req_id, reason, spec)

    def _release_load_spec(self, req_id: str, reason: str) -> None:
        spec = self.load_specs.pop(req_id, None)
        if spec is None:
            return
        if reason == "abort":
            self._inc_restore_stat("_load_spec_abort")
        else:
            self._inc_restore_stat("_load_spec_finish")
        self._log_load_spec(req_id, reason, spec)

    def _rebuild_load_spec(
        self, req_id: str, request: Optional["Request"]
    ) -> LoadSpec:
        tracker = self._request_trackers.get(req_id)
        cached = 0
        if tracker is not None:
            cached = int(getattr(tracker, "num_lmcache_cached_tokens", 0) or 0)
        if cached > 0:
            return LoadSpec(
                vllm_cached_tokens=0,
                lmcache_cached_tokens=cached,
                can_load=True,
                prefetch_issued=False,
                generation=0,
            )
        computed = 0
        if request is not None:
            computed = int(getattr(request, "num_computed_tokens", 0) or 0)
        # No restore plan: force local recompute rather than crashing EngineCore.
        return LoadSpec(
            vllm_cached_tokens=computed,
            lmcache_cached_tokens=computed,
            can_load=False,
            prefetch_issued=True,
            generation=0,
        )

    def _load_spec_for_preempt_resume(
        self, req_id: str, request: Optional["Request"]
    ) -> Optional[LoadSpec]:
        self._inc_restore_stat("_hbm_preempt_resume")
        stored = self.load_specs.get(req_id)
        if stored is None:
            self._inc_restore_stat("_load_spec_missing")
            self._log_load_spec(req_id, "missing", None)
            rebuilt = self._rebuild_load_spec(req_id, request)
            self.load_specs[req_id] = rebuilt
            self._inc_restore_stat("_load_spec_rebuilt")
            self._log_load_spec(req_id, "rebuilt", rebuilt)
            stored = rebuilt
        copied = replace(stored)
        if copied.can_load:
            self._mark_hbm_load_emitted(req_id, "resumed")
        return copied

    @_lmcache_nvtx_annotate
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        # Degraded mode (LMCache init failed): no lookup client means no load specs
        # were ever recorded, so return empty metadata for the worker-side hooks.
        if self.lookup_client is None:
            return LMCacheConnectorMetadata()

        force_skip_save = self.kv_role == "kv_consumer" or self.force_skip_save

        meta = LMCacheConnectorMetadata()
        meta.need_flush_before_forward = _has_preemption_reqs(scheduler_output)
        aborted: set[str] = set()
        for finished_req_id in scheduler_output.finished_req_ids:
            vllm_req = self._unfinished_requests.get(finished_req_id)
            if (
                vllm_req is not None
                and getattr(vllm_req, "status", None) == RequestStatus.FINISHED_ABORTED
            ):
                aborted.add(finished_req_id)
        aborted |= getattr(self, "_pending_worker_abort_ids", set())
        self._pending_worker_abort_ids = set()
        meta.aborted_req_ids = aborted

        for finished_req_id in scheduler_output.finished_req_ids:
            reason = "abort" if finished_req_id in aborted else "finish"
            self._release_load_spec(finished_req_id, reason)
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)

        # Prefetch / async-load requests are not in scheduled_new_reqs.
        # Borrow/copy the restore spec; do not pop. A later HBM preemption
        # still needs the same load_spec to resume.
        prefetch_emitted: set[str] = set()
        for req_id, tracker in list(self._request_trackers.items()):
            if getattr(tracker, "load_state", "") != "waiting_for_load":
                continue
            stored = self.load_specs.get(req_id)
            if stored is None or not stored.can_load:
                continue
            if getattr(stored, "prefetch_issued", False):
                continue
            load_spec = replace(stored)
            req_meta = ReqMeta.from_request_tracker(
                tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self.config.save_decode_cache,
            )
            if req_meta is not None:
                meta.add_request(req_meta)
                prefetch_emitted.add(req_id)
                self._mark_hbm_load_emitted(req_id, "prefetch")
                tracker.load_state = "ready"

        # We should load KV for:
        # 1. new requests
        # 2. preempted requests (once per recovery)
        # can_load will only be True if `update_state_after_alloc` has been called
        # which only happens when vLLM's KV manager has space to receive KV from LMCache
        for request in scheduler_output.scheduled_new_reqs:
            # Ignore DP attention mock requests
            if request.req_id.startswith("mock_req"):
                continue
            if request.req_id in prefetch_emitted:
                continue
            stored = self.load_specs.get(request.req_id)
            load_spec = None
            if stored is not None and not getattr(stored, "prefetch_issued", False):
                load_spec = replace(stored)
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            lmcache_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
            request_priority = self._requests_priority.pop(request.req_id, 0)

            skip_save = force_skip_save or (
                self.config.priority_limit is not None
                and request_priority > self.config.priority_limit
            )

            request_tracker = RequestTracker.from_new_request(
                self.config,
                request,
                num_tokens_to_compute,
                lmcache_cached_tokens,
                skip_save,
            )
            self._request_trackers[request.req_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self.config.save_decode_cache,
            )
            if req_meta is not None:
                meta.add_request(req_meta)
                if load_spec is not None and load_spec.can_load:
                    self._mark_hbm_load_emitted(request.req_id, "new")

        cached_reqs = scheduler_output.scheduled_cached_reqs

        # NOTE: For backward compatibility with vllm version < 0.9.2,
        # In the latest vllm version, the type of scheduled_cached_reqs has
        # changed from list to object `CachedRequestData`
        if isinstance(cached_reqs, list):
            for i, req in enumerate(cached_reqs):
                vllm_request = self._unfinished_requests.get(req.req_id)
                load_spec = None
                if req.resumed_from_preemption:
                    load_spec = self._load_spec_for_preempt_resume(
                        req.req_id, vllm_request
                    )
                lmcache_cached_tokens = 0
                vllm_cached_tokens = 0
                if load_spec is not None:
                    lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                    vllm_cached_tokens = load_spec.vllm_cached_tokens
                request_tracker = self._request_trackers[req.req_id]

                # Pass all_token_ids for preempted requests to restore
                # token_ids correctly for chunk key computation
                all_token_ids = None
                if req.resumed_from_preemption:
                    assert vllm_request is not None, (
                        f"Preempted request {req.req_id} not found "
                        "in _unfinished_requests"
                    )
                    all_token_ids = list(vllm_request.all_token_ids)

                request_tracker.update(
                    req.new_token_ids,
                    req.new_block_ids,
                    req.resumed_from_preemption,
                    lmcache_cached_tokens=lmcache_cached_tokens,
                    vllm_cached_tokens=vllm_cached_tokens,
                    all_token_ids=all_token_ids,
                )

                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    self._lmcache_chunk_size,
                    load_spec=load_spec,
                    discard_partial_chunks=self._discard_partial_chunks,
                    save_decode_cache=self.config.save_decode_cache,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)
            return meta

        for i, req_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._request_trackers[req_id]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # TODO: this is a dangerous reference to the request object inside vllm
            if request := self._unfinished_requests.get(req_id):
                num_current_tokens = request.num_computed_tokens
                # tracker_len < num_computed_tokens during decode
                #   (important for save_decode_cache).
                # num_computed_tokens < tracker_len after preemption.
                tracker_len = len(request_tracker.token_ids)
                slice_base = min(num_current_tokens, tracker_len)
                new_token_ids = request.all_token_ids[
                    slice_base : slice_base + num_new_tokens
                ]
            else:
                raise ValueError(
                    f"Request {req_id} is not in _unfinished_requests, "
                    f"but it is scheduled to be cached"
                )
            new_block_ids = cached_reqs.new_block_ids[i]

            preempted = False
            # Handle both old and new versions of CachedRequestData
            if hasattr(cached_reqs, "resumed_req_ids"):
                # New version with resumed_req_ids
                preempted = req_id in cached_reqs.resumed_req_ids
            elif hasattr(cached_reqs, "resumed_from_preemption"):
                # Old version with resumed_from_preemption
                preempted = cached_reqs.resumed_from_preemption[i]
            else:
                # This case should not be reached with supported vLLM versions.
                # Raising an error is safer than assuming not preempted.
                raise AttributeError(
                    f"Unable to determine preemption status for request {req_id}. "
                    f"This might be due to an unsupported vLLM version."
                )
            load_spec = None
            if preempted:
                load_spec = self._load_spec_for_preempt_resume(req_id, request)
            lmcache_cached_tokens = 0
            vllm_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                vllm_cached_tokens = load_spec.vllm_cached_tokens
            if preempted:
                if load_spec is None or not load_spec.can_load:
                    logger.error(
                        "Request %s resumed from preemption without a usable "
                        "load_spec; falling back to local recompute",
                        req_id,
                    )
                else:
                    # num_computed_tokens should be reset to 0 during preemption
                    # and then set to the number of already cached tokens (maxxing
                    # prefix caching and lmcache)
                    # this assumption is crucial for the update() call of RequestTracker
                    # On full cache hit, get_num_new_matched_tokens subtracts 1
                    # to force last-token recomputation. This only affects
                    # num_computed_tokens when lmcache has all tokens AND
                    # provides more than vLLM's local cache.
                    expected = max(lmcache_cached_tokens, load_spec.vllm_cached_tokens)
                    full_hit_adj = (
                        lmcache_cached_tokens == len(request.all_token_ids)
                        and lmcache_cached_tokens > load_spec.vllm_cached_tokens
                    )
                    if full_hit_adj:
                        expected -= 1
                    if request.num_computed_tokens != expected:
                        logger.error(
                            "Preempted request %s has num_computed_tokens %s "
                            "but expected %s (full_hit_adj=%s); continuing "
                            "without killing EngineCore",
                            req_id,
                            request.num_computed_tokens,
                            expected,
                            full_hit_adj,
                        )

            # When retrieve fail, vllm will call _handle_invalid_blocks to
            # reset request.num_computed_tokens, this will lead to
            # request_tracker.token_ids being not matched with vllm
            if num_current_tokens < len(request_tracker.token_ids):
                logger.warning(
                    "Request %s rolled back from %d to %d tokens; "
                    "truncating tracker state.",
                    req_id,
                    len(request_tracker.token_ids),
                    num_current_tokens,
                )
                num_token_slots = (
                    len(request_tracker.allocated_block_ids) * self._block_size
                )
                tokens_to_keep = num_current_tokens
                if num_token_slots < num_current_tokens:
                    logger.warning(
                        "Request %s tracker has %d token slots but %d tokens; "
                        "capping token_ids to slot capacity.",
                        req_id,
                        num_token_slots,
                        num_current_tokens,
                    )
                    tokens_to_keep = num_token_slots

                request_tracker.token_ids = list(request.all_token_ids[:tokens_to_keep])
                request_tracker.num_saved_tokens = min(
                    request_tracker.num_saved_tokens, tokens_to_keep
                )

            # Pass all_token_ids for preempted requests to restore
            # token_ids correctly for chunk key computation
            all_token_ids = list(request.all_token_ids) if preempted else None

            request_tracker.update(
                new_token_ids,
                new_block_ids,
                preempted=preempted,
                lmcache_cached_tokens=lmcache_cached_tokens,
                vllm_cached_tokens=vllm_cached_tokens,
                all_token_ids=all_token_ids,
            )

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self.config.save_decode_cache,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        return meta

    @_lmcache_nvtx_annotate
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # Layerwise save uses request-scoped generators. If request finishes
        # without entering wait_for_save (abort/error/evict path), make sure
        # we release the generator entry to avoid leaking state.
        if getattr(self, "use_layerwise", False) and hasattr(
            self, "_layerwise_save_storers"
        ):
            self._layerwise_save_storers.pop(request.request_id, None)

        abort = request.status == RequestStatus.FINISHED_ABORTED
        self._hbm_cancelled_req_ids.discard(request.request_id)
        self._release_load_spec(
            request.request_id, "abort" if abort else "finish"
        )

        # Cleanup if request was aborted. The per-branch logic below already
        # degrades gracefully when ``lmcache_engine`` and/or ``lookup_client``
        # are missing (it warns and skips only the unavailable backend), so we
        # must not short-circuit the whole method here -- doing so would drop
        # the storage/lookup cancels that still need to run when just one of
        # the two is present. See LMCache#3337.
        if request.status == RequestStatus.FINISHED_ABORTED:
            # ``request_finished`` is a Scheduler-side connector API.
            # The Scheduler typically does not initialize the storage
            # engine (unless ``enable_scheduler_bypass_lookup`` is set);
            # only the Worker role builds it by default. The Scheduler
            # *does* own the lookup_client though, so the async lookup
            # cancel below must run independently of the engine check
            # to avoid leaking in-flight async lookups on Scheduler-side
            # aborts. See LMCache#3337.
            if self.lmcache_engine is None:
                logger.warning(
                    "Skipping abort-time backend cleanup for request %s: "
                    "lmcache_engine is not initialized (Scheduler role "
                    "without enable_scheduler_bypass_lookup).",
                    request.request_id,
                )
            else:
                # Notify storage backends of aborted requests
                sm = self.lmcache_engine.storage_manager
                if sm is not None:
                    sm.cancel_request(request.request_id)

            if self.async_loading:
                # Cancel any ongoing async lookup and prefetch tasks on
                # workers. Independent of ``lmcache_engine`` because the
                # Scheduler owns ``lookup_client`` even when it does not
                # build an engine.
                lookup_id = request.request_id
                if self.lookup_client is None:
                    logger.warning(
                        "Skipping abort-time async lookup cancel for "
                        "request %s: lookup_client is not initialized "
                        "while async_loading is enabled. Engine stays "
                        "alive; this request's lookup is dropped.",
                        request.request_id,
                    )
                else:
                    self.lookup_client.cancel_lookup(lookup_id)  # type: ignore[attr-defined]

        params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )
        return_params = None

        # NOTE: Used to stream back the first token
        # for disagg prefill
        if params is not None and "ret_first_tok" in params:
            return_params = {
                "first_tok": request._output_token_ids[0],
            }

        if self.config.get_extra_config_value(
            "enable_cache_usage_details_in_response", False
        ):
            request_tracker = self._request_trackers.get(request.request_id)
            if request_tracker:
                return_params = return_params or {}
                return_params["num_lmcache_cached_tokens"] = (
                    request_tracker.num_lmcache_cached_tokens
                )

        if self._async_store_enabled() and self.kv_role != "kv_consumer":
            # Delay-free GPU pages until get_finished, including abort: worker
            # must drain in-flight D2H before those pages can be reused.
            return True, return_params
        return False, return_params

    @_lmcache_nvtx_annotate
    def get_kv_events(self) -> Iterable[CacheStoreEvent]:
        if self.lmcache_engine is not None:
            return self.lmcache_engine.get_kv_events()
        return []
