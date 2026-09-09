# SPDX-License-Identifier: Apache-2.0
"""
LMCache Engine Configuration

Configuration system for LMCache Engine that:
- Loads configuration from YAML file or environment variables
- Supports command-line parameter overrides
- Provides convenient access to configuration values
"""

# Standard
from typing import Any, Dict, Optional, cast
import json
import os

# Third Party
import yaml

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config_base import (
    _parse_local_disk,
    _parse_quoted_string,
    _resolve_config_aliases,
    _to_bool,
    _to_float_list,
    _to_int_list,
    _to_str_list,
    create_config_class,
    load_config_with_overrides,
    validate_and_set_config_value,
)
from lmcache.v1.gds_store_policy import (
    gds_direct_store_max_inflight,
    gds_store_overload_policy,
)

logger = init_logger(__name__)


def _to_hidden_states_retrieve_mode(value: Any) -> str:
    """Normalize hidden_states_retrieve_mode from YAML/env."""
    if value is None:
        return "prefix_strict"
    s = str(value).strip().lower().replace("-", "_")
    if s in ("prefix_strict", "prefixstrict"):
        return "prefix_strict"
    if s in (
        "skip_missing_chunks",
        "skipmissingchunks",
        "legacy",
    ):
        return "skip_missing_chunks"
    raise ValueError(
        "hidden_states_retrieve_mode must be 'prefix_strict' or "
        f"'skip_missing_chunks', got {value!r}"
    )


# Configuration aliases and deprecated mappings
_CONFIG_ALIASES = {
    # Maps deprecated names to current names
    "enable_xpyd": "enable_pd",
    "nixl_peer_host": "pd_peer_host",
    "nixl_peer_init_port": "pd_peer_init_port",
    "nixl_peer_alloc_port": "pd_peer_alloc_port",
    "nixl_proxy_host": "pd_proxy_host",
    "nixl_proxy_port": "pd_proxy_port",
    "nixl_buffer_size": "pd_buffer_size",
    "nixl_role": "pd_role",
    "controller_url": "controller_pull_url",
    "lmcache_worker_port": "lmcache_worker_ports",
    "plugin_locations": "runtime_plugin_locations",
    "external_backends": "storage_plugins",
}

_DEPRECATED_CONFIGS = {
    # Maps deprecated names to warning messages
    "nixl_peer_port": "nixl_peer_port is deprecated, use nixl_receiver_port instead",
    "plugin_locations": (
        "plugin_locations is deprecated, use runtime_plugin_locations instead"
    ),
    "external_backends": (
        "external_backends is deprecated, use storage_plugins instead"
    ),
}

_EC_ENV_PREFIX = "LMCACHE_EC_"
_EC_FILE_PREFIX = "ec_"

# Single configuration definition center - add new config items only here
_CONFIG_DEFINITIONS: dict[str, dict[str, Any]] = {
    # Basic configurations
    "chunk_size": {"type": int, "default": 256, "env_converter": int},
    "local_cpu": {
        "type": bool,
        "default": True,
        "env_converter": _to_bool,
    },
    "max_local_cpu_size": {"type": float, "default": 5.0, "env_converter": float},
    "local_cpu_use_hugepages": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "reserve_local_cpu_size": {"type": float, "default": 0.0, "env_converter": float},
    "local_disk": {
        "type": Optional[str],
        "default": None,
        "env_converter": _parse_local_disk,
    },
    "local_disk_path_sharding": {
        "type": str,
        "default": "by_gpu",
        "env_converter": str,
    },
    "max_local_disk_size": {"type": float, "default": 0.0, "env_converter": float},
    "remote_url": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "remote_serde": {"type": Optional[str], "default": "naive", "env_converter": str},
    # Feature toggles
    "use_layerwise": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "save_decode_cache": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "pre_caching_hash_algorithm": {
        "type": str,
        "default": "builtin",
        "env_converter": str,
    },
    # Blending configurations
    "enable_blending": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "blend_recompute_ratios": {
        "type": Optional[list[float]],
        "default": None,
        "env_converter": _to_float_list,
    },
    "blend_thresholds": {
        "type": Optional[list[float]],
        "default": None,
        "env_converter": _to_float_list,
    },
    "blend_check_layers": {
        "type": list[int],
        "default": None,
        "env_converter": _to_int_list,
    },
    "blend_min_tokens": {"type": int, "default": 256, "env_converter": int},
    "blend_special_str": {"type": str, "default": " # # ", "env_converter": str},
    "retrieve_locations": {
        "type": Optional[list[str]],
        "default": None,
        "env_converter": _to_str_list,
    },
    "store_location": {"type": Optional[str], "default": None, "env_converter": str},
    # P2P configurations
    "enable_p2p": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "p2p_host": {"type": Optional[str], "default": None, "env_converter": str},
    "p2p_init_ports": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    "p2p_lookup_ports": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    # MP-server configurations required by SGLang
    "mp_host": {"type": Optional[str], "default": None, "env_converter": str},
    "mp_port": {"type": int, "default": 5555, "env_converter": int},
    # Controller configurations
    "enable_controller": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "lmcache_instance_id": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "controller_pull_url": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "controller_reply_url": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "lmcache_worker_ports": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    "lmcache_worker_ids": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    # LMCache Worker heartbeat
    # the lmcache_worker_heartbeat_delay_time means that delay a period of time
    # before starting, ensures that the heartbeat starts working only after the
    # service is fully ready(such as, waiting register).
    "lmcache_worker_heartbeat_delay_time": {
        "type": int,
        "default": 10,
        "env_converter": int,
    },
    # the lmcache_worker_heartbeat_time means that sending heartbeat periodically.
    "lmcache_worker_heartbeat_time": {
        "type": Optional[int],
        "default": None,
        "env_converter": int,
    },
    # PD-related configurations
    "enable_pd": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "pd_role": {"type": Optional[str], "default": None, "env_converter": str},
    "pd_buffer_size": {"type": Optional[int], "default": None, "env_converter": int},
    "pd_buffer_device": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "pd_peer_host": {"type": Optional[str], "default": None, "env_converter": str},
    "pd_peer_init_port": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    "pd_peer_alloc_port": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    "pd_peer_query_port": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    "pd_proxy_host": {"type": Optional[str], "default": None, "env_converter": str},
    "pd_proxy_port": {"type": Optional[int], "default": None, "env_converter": int},
    "pd_allocation_timeout_sec": {
        "type": float,
        "default": float("inf"),
        "env_converter": float,
        "description": "Maximum seconds to retry memory allocation before giving up.",
    },
    "pd_shutdown_timeout_sec": {
        "type": float,
        "default": 5.0,
        "env_converter": float,
        "description": (
            "Maximum seconds to wait for event loop shutdown and thread join."
        ),
    },
    "pd_condition_poll_interval_sec": {
        "type": float,
        "default": 0.005,
        "env_converter": float,
        "description": (
            "Polling interval in seconds when waiting on a threading/asyncio "
            "Condition. Small enough to be responsive, large enough not to "
            "spin-waste CPU."
        ),
    },
    "pd_max_prefill_len": {
        "type": int,
        "default": 0,
        "env_converter": int,
        "description": (
            "Maximum prefill token length that the PD buffer must be able to "
            "hold. If > 0, initialization raises ValueError when the buffer "
            "capacity (in tokens) is smaller than this value. "
            "Set to 0 (default) to skip the check."
        ),
    },
    "pd_backend_mode": {
        "type": Optional[str],
        "default": "async",
        "env_converter": str,
        "description": (
            "Select the PD backend implementation: 'async' (default) uses the "
            "asyncio-based implementation; 'sync' uses the original "
            "thread-based synchronous implementation."
        ),
    },
    "pd_skip_proxy_notification": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "pd_bidirectional": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    # Transfer-related configurations
    "transfer_channel": {"type": Optional[str], "default": None, "env_converter": str},
    # Nixl-related configurations
    "nixl_backends": {
        "type": Optional[list[str]],
        "default": None,
        "env_converter": _to_str_list,
    },
    "nixl_buffer_size": {
        "type": Optional[int],
        "default": None,
        "env_converter": int,
    },
    "nixl_buffer_device": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    # Storage paths
    "gds_path": {"type": Optional[str], "default": None, "env_converter": str},
    "gds_path_sharding": {
        "type": str,
        "default": "by_gpu",
        "env_converter": str,
    },
    "gds_buffer_size": {
        "type": Optional[int],
        "default": None,
        "env_converter": int,
    },
    # Maru CXL shared memory backend
    "maru_path": {"type": Optional[str], "default": None, "env_converter": str},
    "maru_pool_size": {
        "type": float,
        "default": 4.0,
        "env_converter": float,
    },
    # GDS (GPU Direct Storage) settings
    "use_gds": {
        "type": bool,
        "default": True,
        "env_converter": _to_bool,
    },
    "gds_backend": {
        "type": str,
        "default": "cufile",
        "env_converter": str,
    },
    # Other configurations
    # (Deprecated) The url of the actual remote lmcache instance for auditing.
    # Please use extra_config['audit_actual_remote_url'] instead.
    "audit_actual_remote_url": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "internal_api_server_host": {
        "type": str,
        "default": "0.0.0.0",
        "env_converter": str,
    },
    "extra_config": {
        "type": Optional[dict],
        "default": None,
        "env_converter": lambda x: (
            x if isinstance(x, dict) else json.loads(x) if x else None
        ),
    },
    "save_unfull_chunk": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "blocking_timeout_secs": {"type": int, "default": 10, "env_converter": int},
    "external_lookup_client": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "py_enable_gc": {
        "type": bool,
        "default": True,
        "env_converter": _to_bool,
    },
    "cache_policy": {
        "type": str,
        "default": "LRU",
        "env_converter": str,
    },
    "numa_mode": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "enable_async_loading": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "internal_api_server_enabled": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "internal_api_server_port_start": {
        "type": int,
        "default": 6999,
        "env_converter": int,
    },
    "priority_limit": {
        "type": Optional[int],
        "default": None,
        "env_converter": int,
    },
    "internal_api_server_include_index_list": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    "internal_api_server_socket_path_prefix": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "runtime_plugin_locations": {
        "type": Optional[list[str]],
        "default": None,
        "env_converter": lambda x: x if isinstance(x, list) else [x] if x else [],
    },
    "storage_plugins": {
        "type": Optional[list[str]],
        "default": None,
        "env_converter": _to_str_list,
    },
    "remote_storage_plugins": {
        "type": Optional[list[str]],
        "default": None,
        "env_converter": _to_str_list,
    },
    # Lookup client configurations
    "lookup_timeout_ms": {
        "type": int,
        "default": 3000,
        "env_converter": int,
    },
    "relookup_on_schedule": {
        "type": bool,
        "default": True,
        "env_converter": _to_bool,
        "description": (
            "If a previous async lookup cached 0 hits, allow a later scheduler "
            "pass to issue a new lookup. Hits >0 and in-flight lookups are not "
            "retried. Breaks one-shot lookup only for misses."
        ),
    },
    "relookup_min_interval_ms": {
        "type": int,
        "default": 1000,
        "env_converter": int,
        "description": (
            "Minimum wall time between a cached miss and a relookup_on_schedule retry."
        ),
    },
    "relookup_max_attempts": {
        "type": int,
        "default": 1,
        "env_converter": int,
        "description": (
            "Max extra lookup() calls after a cached miss. 0 disables retry "
            "(commit the miss immediately). Prevents 0→-1 loops that re-defer "
            "turn-1 prefills."
        ),
    },
    "min_retrieve_tokens": {
        "type": int,
        "default": 0,
        "env_converter": int,
        "description": (
            "Minimum number of hit tokens required to perform retrieve. "
            "If hit tokens < min_retrieve_tokens, skip retrieve but the "
            "actual hit count is still used for skip_leading_tokens to avoid "
            "re-storing existing chunks. Default is 0 (disabled)."
        ),
    },
    "hit_miss_ratio": {
        "type": Optional[float],
        "default": None,
        "env_converter": float,
    },
    "lookup_server_worker_ids": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    "enable_scheduler_bypass_lookup": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "script_allowed_imports": {
        "type": Optional[list[str]],
        "default": None,
        "env_converter": _to_str_list,
    },
    # Lazy memory allocator configurations
    "enable_lazy_memory_allocator": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": (
            "Enable lazy memory allocator to reduce initial memory footprint. "
            "Memory is allocated on-demand and expanded automatically when needed."
        ),
    },
    "lazy_memory_initial_ratio": {
        "type": float,
        "default": 0.2,
        "env_converter": float,
        "description": (
            "Initial memory allocation ratio (0.0-1.0). "
            "Determines the percentage of target memory size to allocate at startup. "
            "Default is 0.2 (20%)."
        ),
    },
    "lazy_memory_expand_trigger_ratio": {
        "type": float,
        "default": 0.5,
        "env_converter": float,
        "description": (
            "Memory usage ratio (0.0-1.0) that triggers automatic expansion. "
            "When memory usage exceeds this threshold, expansion is triggered. "
            "Default is 0.5 (50%)."
        ),
    },
    "lazy_memory_step_ratio": {
        "type": float,
        "default": 0.1,
        "env_converter": float,
        "description": (
            "Memory expansion step ratio (0.0-1.0). "
            "Determines the percentage of target memory size to add in each expansion. "
            "Default is 0.1 (10%)."
        ),
    },
    "lazy_memory_safe_size": {
        "type": float,
        "default": 0.0,
        "env_converter": float,
        "description": (
            "Safe threshold size in GB. Lazy allocator is only enabled when "
            "max_local_cpu_size exceeds this value. Default is 0.0 GB (always enabled)."
        ),
    },
    "async_store_d2h_enabled": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": (
            "Enable async D2H store: wait_for_save issues D2H and records a "
            "CUDA event; GPU pages are released after get_finished sees the event. "
            "Object put stays on the background worker. Default False for "
            "non-SurData deployments."
        ),
    },
    # These watermarks skip a store when in-flight work is too large; they
    # never block decode.
    "async_store_max_pending_bytes": {
        "type": int,
        "default": 0,
        "env_converter": int,
        "description": (
            "Max host L1 bytes held by in-flight async D2H stores. 0 = unlimited. "
            "When exceeded, skip this cache store (count a drop); never block decode."
        ),
    },
    "async_store_max_pending_requests": {
        "type": int,
        "default": 0,
        "env_converter": int,
        "description": (
            "Max requests with in-flight async D2H stores. 0 = unlimited. "
            "When exceeded, skip this cache store and count a drop."
        ),
    },
    "async_store_max_delayed_free_blocks": {
        "type": int,
        "default": 0,
        "env_converter": int,
        "description": (
            "Max GPU blocks retained by delay-free while D2H is in flight. "
            "0 = unlimited unless async_store_max_delayed_free_ratio is set. "
            "When exceeded, skip this cache store and count a drop "
            "to avoid preemption storms on small HBM (e.g. 12GiB)."
        ),
    },
    "async_store_max_delayed_free_ratio": {
        "type": float,
        "default": 0.25,
        "env_converter": float,
        "description": (
            "Cap delayed-free GPU blocks as a fraction of registered KV "
            "blocks (0 < x <= 1). 0 disables the ratio cap. Combined with a "
            "non-zero async_store_max_delayed_free_blocks by taking the min. "
            "If the ratio is enabled but total blocks cannot be inferred, "
            "every async store is refused."
        ),
    },
    "hbm_prefetch_enable": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": (
            "Enable spare-HBM prefetch: external KV hits load asynchronously "
            "(load_kv_async=True) into free GPU pages while the current step "
            "computes. Default false."
        ),
    },
    "hbm_prefetch_watermark_ratio": {
        "type": float,
        "default": 0.15,
        "env_converter": float,
        "description": (
            "Keep at least this fraction of GPU KV blocks free after prefetch "
            "admission (0 < x < 1). Default 0.15."
        ),
    },
    "hbm_prefetch_max_inflight": {
        "type": int,
        "default": 8,
        "env_converter": int,
        "description": (
            "Max requests in WAITING_FOR_REMOTE_KVS / in-flight H2D prefetch. "
            "Must be > 0. Default 8."
        ),
    },
    "hbm_prefetch_min_cuda_free_mb": {
        "type": int,
        "default": 256,
        "env_converter": int,
        "description": (
            "Extra HBM prefetch is refused when torch.cuda.mem_get_info free "
            "bytes are below this many MiB. 0 disables the CUDA-free gate. "
            "Does not skip TP retrieve broadcast for already can_load requests. "
            "Default 256."
        ),
    },
    "kv_io_mode": {
        "type": str,
        "default": "dram",
        "env_converter": str,
        "description": (
            "Process-level KV transfer mode: 'dram' (SurData→L1 then L1→HBM) "
            "or 'gds' (SurData↔GPU, no L1). Unknown values are rejected."
        ),
    },
    "kv_gds_inflight": {
        "type": int,
        "default": 8,
        "env_converter": int,
        "description": (
            "Max concurrent GDS object I/O slots. Pool slot_count is "
            "min(kv_gds_inflight, kv_gds_staging_bytes // padded_slot_bytes)."
        ),
    },
    "kv_gds_staging_bytes": {
        "type": int,
        "default": 268435456,
        "env_converter": int,
        "description": (
            "Total GDS bounce-pool byte cap (not per slot). Default 256MiB so "
            "8 × ~20MiB DMA slots fit. Too small for one padded slot fails closed."
        ),
    },
    "kv_gds_get_workers": {
        "type": int,
        "default": 4,
        "env_converter": int,
        "description": (
            "Dedicated GDS object_get_to_gpu executor threads. Must not share "
            "the lookup asyncio loop. Default 4."
        ),
    },
    "kv_gds_get_queue_depth": {
        "type": int,
        "default": 32,
        "env_converter": int,
        "description": (
            "Max queued GDS GET admissions. Worker-thread GET uses try-acquire; "
            "exhaustion falls through to DRAM+H2D instead of blocking."
        ),
    },
    "kv_gds_retrieve_slots": {
        "type": int,
        "default": 4,
        "env_converter": int,
        "description": (
            "Retrieve-only GDS bounce slots (HBM). Store cannot borrow. "
            "try-acquire only; exhaustion falls through to DRAM+H2D."
        ),
    },
    "kv_gds_store_slots": {
        "type": int,
        "default": 4,
        "env_converter": int,
        "description": (
            "Store-only GDS bounce slots (HBM). Retrieve cannot borrow. "
            "try-acquire only; exhaustion falls through to D2H+host put."
        ),
    },
    "kv_gds_dram_fallback_bytes": {
        "type": int,
        "default": 3221225472,
        "env_converter": int,
        "description": (
            "Pinned-host budget for Tier-3 DRAM fallback (not L1 cache). "
            "Independent of kv_gds_staging_bytes. Default 3GiB covers "
            "32-way store with ~8MiB chunks when GDS slots are short."
        ),
    },
    "kv_gds_watchdog_timeout_ms": {
        "type": int,
        "default": 0,
        "env_converter": int,
        "description": (
            "Stuck-hardware watchdog for GDS DMA/lease wait in milliseconds. "
            "0 disables (normal 32x2/800x2 path). Not used for flow control."
        ),
    },
    "kv_gds_staging_arena_bytes": {
        "type": int,
        "default": 67108864,
        "env_converter": int,
        "description": (
            "Registered page-direct GPU staging arena size in bytes for "
            "header/residual/read slices. Not the 32-slot full-KV bounce pool."
        ),
    },
    "kv_gds_staging_max_waiters": {
        "type": int,
        "default": 32,
        "env_converter": int,
        "description": "Max waiters blocked on page-direct staging or D2H admission.",
    },
    "kv_gds_staging_wait_timeout_ms": {
        "type": int,
        "default": 1000,
        "env_converter": int,
        "description": (
            "Wait timeout before emergency D2H / backpressure error. "
            "GDS store uses try-acquire (timeout 0) regardless of this value. "
            "0 means a single try then fail closed (no silent drop)."
        ),
    },
    "kv_gds_store_overload_policy": {
        "type": str,
        "default": "immediate_drop",
        "env_converter": str,
        "description": (
            "When page-direct GDS store cannot be admitted: "
            "immediate_drop discards this store without D2H. "
            "immediate_d2h_fallback is an explicit debug option that copies "
            "the whole request to host. Unknown names are rejected."
        ),
    },
    "kv_gds_direct_store_max_inflight": {
        "type": int,
        "default": 0,
        "env_converter": int,
        "description": (
            "Max concurrent page-direct GDS stores. 0 means unlimited. "
            "A native KVB store has multiple records/disks; do not copy "
            "Env per-disk outstanding=2 as this cap."
        ),
    },
    "gds_lookup_deferral_timeout_ms": {
        "type": int,
        "default": 50,
        "env_converter": int,
        "description": (
            "Scheduler-side bound on WAITING_FOR_REMOTE_KVS for GDS lookup. "
            "After this many milliseconds the request is scheduled as a miss "
            "(0, False) while lookup continues in the background. 0 disables. "
            "The scheduler must not sleep."
        ),
    },
    "gds_skip_exists_empty_ns": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": (
            "Benchmark-only. Skip native exists when the namespace was just "
            "formatted empty and gds_empty_ns_stamp matches the run stamp. "
            "Default false. Never infer emptiness from 'this process has not "
            "stored yet'."
        ),
    },
    "gds_empty_ns_stamp": {
        "type": str,
        "default": "",
        "env_converter": str,
        "description": (
            "Run stamp that must match LMCACHE_RUN_STAMP / kv.gds_run_stamp "
            "before gds_skip_exists_empty_ns may take effect."
        ),
    },
    "max_tokens_per_load": {
        "type": int,
        "default": 0,
        "env_converter": int,
        "description": (
            "Cap external tokens reported per scheduling step (DRAM chunked "
            "loading). 0 = no cap. GDS must keep this at 0 so a confirmed "
            "prefix is not truncated."
        ),
    },
    "dram_preload_fifo_depth": {
        "type": int,
        "default": 32,
        "env_converter": int,
        "description": "Max in-flight DRAM preloads that do not occupy HBM.",
    },
    "dram_preload_timeout_ms": {
        "type": int,
        "default": 30000,
        "env_converter": int,
        "description": "Timeout for SurData→L1 gather (not HBM assembly).",
    },
    # Chunk statistics configurations
    "enable_chunk_statistics": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": "Enable chunk statistics tracking.",
    },
    "chunk_statistics_auto_start_statistics": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": "Auto-start statistics on init.",
    },
    "chunk_statistics_auto_exit_timeout_hours": {
        "type": float,
        "default": 0.0,
        "env_converter": float,
        "description": "Auto-stop timeout in hours (0=disabled).",
    },
    "chunk_statistics_auto_exit_target_unique_chunks": {
        "type": int,
        "default": 0,
        "env_converter": int,
        "description": "Auto-stop at target unique chunks.",
    },
    "chunk_statistics_strategy": {
        "type": str,
        "default": "memory_bloom_filter",
        "env_converter": str,
        "description": "Recording strategy: memory_bloom_filter or file_hash.",
    },
    # KV events configuration
    "enable_kv_events": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    # TODO(chunxiaozheng): remove this after VLLMPagedMemGPUConnectorV3 is stable
    "use_gpu_connector_v3": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    # Memory management configurations
    "pin_timeout_sec": {
        "type": int,
        "default": 300,
        "env_converter": int,
        "description": (
            "Maximum duration in seconds that a memory object can remain pinned. "
            "If a pinned object exceeds this timeout, it will be forcibly unpinned "
            "by the PinMonitor to prevent memory leaks. Default is 300 seconds."
        ),
    },
    "pin_check_interval_sec": {
        "type": int,
        "default": 30,
        "env_converter": int,
        "description": (
            "Interval in seconds between PinMonitor timeout checks. "
            "The background thread periodically scans all pinned objects at this "
            "interval to detect and handle timeouts. Default is 30 seconds."
        ),
    },
    # Remote configuration service
    "remote_config_url": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
        "description": (
            "URL of the remote configuration service. When set, LMCache will "
            "fetch additional configuration from this URL at startup."
        ),
    },
    "app_id": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
        "description": (
            "Application ID to send to the remote configuration service. "
            "If not set, the remote service may infer it from current config "
            "and environment variables."
        ),
    },
    # Hidden state caching configurations (vLLM-Omni multi-stage pipeline)
    "enable_hidden_state_cache": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": (
            "Enable caching of thinker hidden states alongside KV cache entries "
            "for vLLM-Omni multi-stage pipelines. When enabled, hidden states "
            "are stored in a separate CPU pinned memory pool and share the same "
            "chunk keys as their corresponding KV entries."
        ),
    },
    "max_hidden_state_cpu_size": {
        "type": float,
        "default": 2.0,
        "env_converter": float,
        "description": (
            "Maximum size in GB of pinned CPU memory for the hidden state cache. "
            "Each chunk-layer tensor is [chunk_size, hidden_dim] float32. "
            "Sizing formula: num_cached_layers × chunk_size × hidden_dim × 4 bytes. "
            "Example: Qwen3-Omni with layers=[0,24], chunk_size=256, hidden_dim=5120 "
            "→ ~10 MB/chunk; 2 GB allows ≈200 chunks (~51K tokens) of prefix. "
            "If this budget is too small, hidden entries will be evicted before "
            "their KV counterparts, causing partial-prefix fallback. "
            "Relevant only when enable_hidden_state_cache=True."
        ),
    },
    "hidden_state_layers": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
        "description": (
            "Optional allowlist of **storage layer indices** accepted by "
            "HiddenStateStore.store_hidden_states. If None (recommended "
            "default), every layer index passed on store is cached. "
            "**Semantics depend on the integration:** these are storage "
            "layer_idx values, not necessarily transformer layer IDs. For "
            "example, a multi-stage pipeline may use layer_idx 0 for the main "
            "hidden-state tensor and 1, 2, … for multimodal output slots; "
            "copying unrelated examples such as [0, 24] as if they were "
            '"model layers" can **silently drop** rows stored under other '
            "indices. Leave this unset unless you have verified the exact "
            "indices your stack writes—consult your integration's docs. "
            "Relevant only when enable_hidden_state_cache=True."
        ),
    },
    "hidden_states_retrieve_mode": {
        "type": str,
        "default": "prefix_strict",
        "env_converter": _to_hidden_states_retrieve_mode,
        "description": (
            "How to assemble hidden_states_out on retrieve() when some KV-hit "
            "chunks lack a paired hidden entry. "
            "'prefix_strict' (default): stop at the first missing chunk — "
            "outputs align with a contiguous token prefix (recommended for "
            "thinker→talker). "
            "'skip_missing_chunks': legacy behavior — skip missing chunks and "
            "concatenate later chunks; tensor length may not match ret_mask. "
            "Relevant only when enable_hidden_state_cache=True."
        ),
    },
}


def enforce_gds_invariants(config) -> None:
    """Force GDS routing after extras. Keep a positive CPU allocator budget."""
    extra = getattr(config, "extra_config", None)
    if not isinstance(extra, dict):
        extra = {}
        config.extra_config = extra
    extra_mode = extra.get("kv.io_mode")
    if extra_mode is not None:
        config.kv_io_mode = extra_mode
    io_mode = str(getattr(config, "kv_io_mode", "dram") or "dram").strip().lower()
    if io_mode != "gds":
        config.kv_io_mode = io_mode
        return
    nested = extra.get("extra_config")
    if isinstance(nested, dict) and str(nested.get("kv.io_mode", "gds")).lower() != "gds":
        raise ValueError("nested extra_config kv.io_mode conflicts with GDS")
    if str(extra.get("kv.io_mode", "gds")).lower() not in ("gds",):
        raise ValueError("extra_config kv.io_mode conflicts with requested GDS")
    if getattr(config, "local_cpu", True):
        logger.warning("kv_io_mode=gds disables local_cpu / L1 DRAM hot path")
    config.local_cpu = False
    config.kv_io_mode = "gds"
    extra["surdata.host_only"] = 0
    extra["kv.io_mode"] = "gds"
    extra["surdata.l1_budget_enable"] = 0
    if bool(getattr(config, "hbm_prefetch_enable", False)):
        logger.warning(
            "kv_io_mode=gds disables hbm_prefetch_enable; use stock async load"
        )
    config.hbm_prefetch_enable = False
    extra["hbm_prefetch_enable"] = False
    max_tpl = int(getattr(config, "max_tokens_per_load", 0) or 0)
    extra_tpl = extra.get("lmcache.max_tokens_per_load", extra.get("max_tokens_per_load"))
    if extra_tpl is not None:
        max_tpl = int(extra_tpl)
    if max_tpl != 0:
        logger.warning(
            "kv_io_mode=gds forces max_tokens_per_load from %s to 0",
            max_tpl,
        )
    config.max_tokens_per_load = 0
    extra["max_tokens_per_load"] = 0
    extra["lmcache.max_tokens_per_load"] = 0
    first_rank = extra.get("first_rank_max_local_cpu_size")
    if first_rank is not None:
        extra["first_rank_max_local_cpu_size"] = min(float(first_rank), 1.0)
    max_cpu = float(getattr(config, "max_local_cpu_size", 0.0) or 0.0)
    if max_cpu <= 0.0:
        raise ValueError(
            "kv_io_mode=gds requires max_local_cpu_size > 0 so RemoteBackend "
            "can create a CPU allocator (not an L1 hot cache)"
        )
    # LocalCPUBackend still eagerly allocates this pool as RemoteBackend scratch.
    # Harness DRAM L1 leftover (e.g. 20 GB) would pin/zero-fill for minutes and
    # fail EngineCore health. Cap to a scratch budget, not an L1 cache.
    gds_cpu_scratch_gb = 1.0
    fb_bytes = int(getattr(config, "kv_gds_dram_fallback_bytes", 0) or 0)
    if fb_bytes > 0:
        gds_cpu_scratch_gb = max(gds_cpu_scratch_gb, (float(fb_bytes) / (1024.0 ** 3)) + 0.25)
    if max_cpu > gds_cpu_scratch_gb:
        logger.warning(
            "kv_io_mode=gds caps max_local_cpu_size from %.2f GB to %.2f GB "
            "(RemoteBackend allocator scratch, not L1 DRAM)",
            max_cpu,
            gds_cpu_scratch_gb,
        )
        config.max_local_cpu_size = gds_cpu_scratch_gb


# Specialized methods that are unique to LMCacheEngineConfig
def _validate_config(self):
    """Validate configuration"""

    # needed for the old async serializer implementation
    # # auto-adjust save_unfull_chunk for async loading to prevent CPU fragmentation
    # if self.enable_async_loading:
    #     logger.warning(
    #         "Automatically setting save_unfull_chunk=False because "
    #         "enable_async_loading=True or use_layerwise=True to prevent "
    #         "CPU memory fragmentation"
    #     )
    #     self.save_unfull_chunk = False

    if self.min_retrieve_tokens < 0:
        raise ValueError(
            "min_retrieve_tokens must be >= 0, got %d" % self.min_retrieve_tokens
        )

    delayed_ratio = float(getattr(self, "async_store_max_delayed_free_ratio", 0.25) or 0.0)
    if delayed_ratio < 0.0 or delayed_ratio > 1.0:
        raise ValueError(
            "async_store_max_delayed_free_ratio must be in [0, 1], got %s"
            % delayed_ratio
        )
    self.async_store_max_delayed_free_ratio = delayed_ratio

    ratio = float(getattr(self, "hbm_prefetch_watermark_ratio", 0.15))
    if not (0.0 < ratio < 1.0):
        raise ValueError(
            "hbm_prefetch_watermark_ratio must be in (0, 1), got %s" % ratio
        )
    max_inf = int(getattr(self, "hbm_prefetch_max_inflight", 8))
    if max_inf < 1:
        raise ValueError("hbm_prefetch_max_inflight must be > 0, got %d" % max_inf)
    min_cuda_mb = int(getattr(self, "hbm_prefetch_min_cuda_free_mb", 256))
    if min_cuda_mb < 0:
        raise ValueError(
            "hbm_prefetch_min_cuda_free_mb must be >= 0, got %d" % min_cuda_mb
        )
    extra = getattr(self, "extra_config", None)
    if not isinstance(extra, dict):
        extra = {}
        self.extra_config = extra
    extra_mode = extra.get("kv.io_mode")
    if extra_mode is not None:
        self.kv_io_mode = extra_mode
    extra_fb = extra.get("kv.gds_dram_fallback_bytes")
    if extra_fb is not None:
        self.kv_gds_dram_fallback_bytes = extra_fb
    extra_ds = extra.get("kv.gds_direct_store_max_inflight")
    if extra_ds is not None:
        self.kv_gds_direct_store_max_inflight = extra_ds
    extra_ol = extra.get("kv.gds_store_overload_policy")
    if extra_ol is not None:
        self.kv_gds_store_overload_policy = extra_ol
    io_mode = str(getattr(self, "kv_io_mode", "dram") or "dram").strip().lower()
    if io_mode not in ("dram", "gds"):
        raise ValueError("kv_io_mode must be dram|gds, got %r" % io_mode)
    self.kv_io_mode = io_mode
    depth = int(getattr(self, "dram_preload_fifo_depth", 32) or 32)
    if depth < 1:
        raise ValueError("dram_preload_fifo_depth must be > 0, got %d" % depth)
    if io_mode == "gds":
        enforce_gds_invariants(self)

    extra = getattr(self, "extra_config", None)
    if not isinstance(extra, dict):
        extra = {}
        self.extra_config = extra
    extra_inflight = extra.get("kv.gds_inflight")
    if extra_inflight is not None:
        self.kv_gds_inflight = extra_inflight
    extra_staging = extra.get("kv.gds_staging_bytes")
    if extra_staging is not None:
        self.kv_gds_staging_bytes = extra_staging
    inflight = int(getattr(self, "kv_gds_inflight", 8) or 8)
    if inflight < 1:
        raise ValueError("kv_gds_inflight must be >= 1, got %r" % inflight)
    self.kv_gds_inflight = inflight
    staging_bytes = int(getattr(self, "kv_gds_staging_bytes", 268435456) or 0)
    if staging_bytes < 1:
        raise ValueError(
            "kv_gds_staging_bytes must be >= 1 (total pool cap), got %r" % staging_bytes
        )
    self.kv_gds_staging_bytes = staging_bytes
    extra_workers = extra.get("kv.gds_get_workers")
    if extra_workers is not None:
        self.kv_gds_get_workers = extra_workers
    extra_qd = extra.get("kv.gds_get_queue_depth")
    if extra_qd is not None:
        self.kv_gds_get_queue_depth = extra_qd
    extra_wd = extra.get("kv.gds_watchdog_timeout_ms")
    if extra_wd is not None:
        self.kv_gds_watchdog_timeout_ms = extra_wd
    extra_arena = extra.get("kv.gds_staging_arena_bytes")
    if extra_arena is not None:
        self.kv_gds_staging_arena_bytes = extra_arena
    extra_mw = extra.get("kv.gds_staging_max_waiters")
    if extra_mw is not None:
        self.kv_gds_staging_max_waiters = extra_mw
    extra_swt = extra.get("kv.gds_staging_wait_timeout_ms")
    if extra_swt is not None:
        self.kv_gds_staging_wait_timeout_ms = extra_swt
    extra_rs = extra.get("kv.gds_retrieve_slots")
    if extra_rs is not None:
        self.kv_gds_retrieve_slots = extra_rs
    extra_ss = extra.get("kv.gds_store_slots")
    if extra_ss is not None:
        self.kv_gds_store_slots = extra_ss
    extra_fb = extra.get("kv.gds_dram_fallback_bytes")
    if extra_fb is not None:
        self.kv_gds_dram_fallback_bytes = extra_fb
    extra_ds = extra.get("kv.gds_direct_store_max_inflight")
    if extra_ds is not None:
        self.kv_gds_direct_store_max_inflight = extra_ds
    extra_ol = extra.get("kv.gds_store_overload_policy")
    if extra_ol is not None:
        self.kv_gds_store_overload_policy = extra_ol
    extra_ld = extra.get("kv.gds_lookup_deferral_timeout_ms")
    if extra_ld is not None:
        self.gds_lookup_deferral_timeout_ms = extra_ld
    extra_se = extra.get("kv.gds_skip_exists_empty_ns")
    if extra_se is not None:
        self.gds_skip_exists_empty_ns = extra_se
    extra_ns = extra.get("kv.gds_empty_ns_stamp")
    if extra_ns is not None:
        self.gds_empty_ns_stamp = extra_ns
    extra_run = extra.get("kv.gds_run_stamp")
    if extra_run is not None and not str(
        getattr(self, "gds_empty_ns_stamp", "") or ""
    ).strip():
        self.gds_empty_ns_stamp = extra_run
    defer_ms = int(getattr(self, "gds_lookup_deferral_timeout_ms", 50) or 0)
    if defer_ms < 0:
        raise ValueError(
            "gds_lookup_deferral_timeout_ms must be >= 0, got %r" % defer_ms
        )
    self.gds_lookup_deferral_timeout_ms = defer_ms
    skip_exists = getattr(self, "gds_skip_exists_empty_ns", False)
    if isinstance(skip_exists, bool):
        self.gds_skip_exists_empty_ns = skip_exists
    else:
        self.gds_skip_exists_empty_ns = str(skip_exists).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    self.gds_empty_ns_stamp = str(getattr(self, "gds_empty_ns_stamp", "") or "")
    fb_bytes = int(getattr(self, "kv_gds_dram_fallback_bytes", 0) or 0)
    if fb_bytes < 0:
        raise ValueError("kv_gds_dram_fallback_bytes must be >= 0, got %r" % fb_bytes)
    self.kv_gds_dram_fallback_bytes = fb_bytes
    self.kv_gds_direct_store_max_inflight = gds_direct_store_max_inflight(
        getattr(self, "kv_gds_direct_store_max_inflight", 0)
    )
    self.kv_gds_store_overload_policy = gds_store_overload_policy(
        getattr(self, "kv_gds_store_overload_policy", None)
    )
    extra_tpl = extra.get("lmcache.max_tokens_per_load", extra.get("max_tokens_per_load"))
    if extra_tpl is not None:
        self.max_tokens_per_load = extra_tpl
    max_tpl = int(getattr(self, "max_tokens_per_load", 0) or 0)
    if max_tpl < 0:
        raise ValueError("max_tokens_per_load must be >= 0, got %r" % max_tpl)
    self.max_tokens_per_load = max_tpl
    if str(getattr(self, "kv_io_mode", "dram") or "dram").lower() == "gds":
        self.max_tokens_per_load = 0
        self.hbm_prefetch_enable = False
    get_workers = int(getattr(self, "kv_gds_get_workers", 4) or 4)
    if get_workers < 1:
        raise ValueError("kv_gds_get_workers must be >= 1, got %r" % get_workers)
    self.kv_gds_get_workers = get_workers
    get_qd = int(getattr(self, "kv_gds_get_queue_depth", 32) or 32)
    if get_qd < 1:
        raise ValueError("kv_gds_get_queue_depth must be >= 1, got %r" % get_qd)
    self.kv_gds_get_queue_depth = get_qd
    wd_ms = int(getattr(self, "kv_gds_watchdog_timeout_ms", 0) or 0)
    if wd_ms < 0:
        raise ValueError("kv_gds_watchdog_timeout_ms must be >= 0, got %r" % wd_ms)
    self.kv_gds_watchdog_timeout_ms = wd_ms
    arena = int(getattr(self, "kv_gds_staging_arena_bytes", 67108864) or 0)
    if arena < 1:
        raise ValueError("kv_gds_staging_arena_bytes must be >= 1, got %r" % arena)
    self.kv_gds_staging_arena_bytes = arena
    max_w = int(getattr(self, "kv_gds_staging_max_waiters", 32) or 0)
    if max_w < 1:
        raise ValueError("kv_gds_staging_max_waiters must be >= 1, got %r" % max_w)
    self.kv_gds_staging_max_waiters = max_w
    swt = int(getattr(self, "kv_gds_staging_wait_timeout_ms", 1000) or 0)
    if swt < 0:
        raise ValueError(
            "kv_gds_staging_wait_timeout_ms must be >= 0, got %r" % swt
        )
    self.kv_gds_staging_wait_timeout_ms = swt

    if self.enable_blending:
        if not self.save_unfull_chunk:
            logger.warning(
                "Automatically setting save_unfull_chunk=True because "
                "enable_blending=True"
            )
            self.save_unfull_chunk = True

    if self.enable_controller:
        if self.lmcache_instance_id is None:
            raise ValueError(
                "lmcache_instance_id is required when enable_controller=True"
            )
        if self.controller_pull_url is None:
            raise ValueError(
                "controller_pull_url is required when enable_controller=True"
            )
        if self.controller_reply_url is None:
            raise ValueError(
                "controller_reply_url is required when enable_controller=True"
            )
        if not self.lmcache_worker_ports:
            raise ValueError(
                "lmcache_worker_ports is required and cannot be "
                "empty when enable_controller=True"
            )

    if self.enable_p2p:
        assert self.enable_controller
        assert self.controller_pull_url is not None
        assert self.controller_reply_url is not None
        assert self.lmcache_worker_ports is not None
        assert self.p2p_host is not None
        assert self.p2p_init_ports is not None
        assert self.p2p_lookup_ports is not None
        assert self.transfer_channel is not None

    enable_nixl_storage = self.extra_config is not None and self.extra_config.get(
        "enable_nixl_storage"
    )
    if self.enable_pd:
        assert self.pd_role is not None
        assert self.pd_buffer_size is not None
        assert self.pd_buffer_device is not None
        assert self.enable_p2p is False, "PD only supports enable_p2p=False"
        if self.pd_backend_mode not in ("sync", "async"):
            raise ValueError(
                f"pd_backend_mode must be 'sync' or 'async', "
                f"got {self.pd_backend_mode!r}"
            )

        # PD requires save_unfull_chunk=True for complete KV cache transfer
        # from prefill node to decode node. Without this, partial chunks would
        # be discarded, causing incomplete KV cache transfer and wrong results
        # on the decode node.
        if not self.save_unfull_chunk:
            logger.warning(
                "PD (Peer-to-Peer Disaggregation) requires save_unfull_chunk=True "
                "for complete KV cache transfer. Automatically setting "
                "save_unfull_chunk=True."
            )
            self.save_unfull_chunk = True
        else:
            logger.info(
                "PD mode enabled with save_unfull_chunk=True - all KV cache "
                "including partial chunks will be transferred to decode node"
            )

        # for receiver, PDBackend is for retrieve location
        # can't take PDBackend as store location
        # as PDBackend is now one way from producer to receiver only
        if self.pd_role == "receiver":
            assert self.store_location != "PDBackend", (
                "store_location cannot be PDBackend for receiver"
            )
            assert self.retrieve_locations in (None, ["PDBackend"]), (
                "for pd receiver, "
                'retrieve_locations are expected to be ["PDBackend"], '
                f"now, it is {self.retrieve_locations}"
            )

    if enable_nixl_storage:
        assert self.extra_config.get("nixl_backend") is not None
        assert self.extra_config.get("nixl_pool_size") is not None
        if self.extra_config.get(
            "nixl_presence_cache_only"
        ) and not self.extra_config.get("nixl_presence_cache"):
            raise ValueError(
                "nixl_presence_cache must be true when nixl_presence_cache_only is true"
            )
        assert self.nixl_buffer_device is not None
        if self.nixl_buffer_device == "cpu":
            # CPU mode shares LocalCPUBackend's pinned pool; nixl_buffer_size
            # has no effect there. Reject the combo so users don't silently
            # carry a stale GPU-mode value into a CPU-mode deployment.
            if self.nixl_buffer_size is not None:
                raise ValueError(
                    "nixl_buffer_size must not be set when "
                    "nixl_buffer_device='cpu'. In CPU mode NIXL shares "
                    "LocalCPUBackend's pinned pool, which is sized by "
                    "max_local_cpu_size."
                )
            if self.max_local_cpu_size <= 0:
                raise ValueError(
                    "nixl_buffer_device='cpu' requires max_local_cpu_size > 0 "
                    "(LocalCPUBackend's pinned pool is the NIXL staging buffer)."
                )
            # With enable_p2p=True, both the P2P backend and the NIXL
            # storage backend would run their own NIXL agents over
            # LocalCPUBackend's pinned pool. The pieces are structurally
            # supported — NIXL allows registering the same memory from
            # multiple agents, and both backends already allocate via
            # LocalCPUBackend.allocate() so any contention runs inside
            # LocalCPUBackend's allocator rather than across backends.
            # But the combined configuration has no CI coverage and has
            # not been exercised end-to-end. Reject until it has been.
            if self.enable_p2p:
                raise ValueError(
                    "enable_p2p=True together with enable_nixl_storage=True "
                    "+ nixl_buffer_device='cpu' has not been validated "
                    "end-to-end and has no CI coverage. Use enable_p2p=True "
                    "with nixl_buffer_device='cuda', or disable enable_p2p "
                    "when using the NIXL CPU shared pool. This rejection "
                    "can be lifted once the combination is exercised by "
                    "integration tests."
                )
        else:
            assert self.nixl_buffer_size is not None

        # Deprecation: extra_config.nixl_use_hugepages → local_cpu_use_hugepages.
        # The flag was always a no-op for GPU buffers (hugepages don't apply);
        # in CPU mode the pinned pool is now owned by LocalCPUBackend, so
        # local_cpu_use_hugepages is the only effective knob. Alias the value
        # in CPU mode, warn in GPU mode, then pop the deprecated key so
        # downstream readers see a single source of truth.
        if "nixl_use_hugepages" in self.extra_config:
            nixl_huge = bool(self.extra_config["nixl_use_hugepages"])
            user_set = getattr(self, "_user_set_keys", set())
            if self.nixl_buffer_device == "cpu":
                if (
                    "local_cpu_use_hugepages" in user_set
                    and self.local_cpu_use_hugepages != nixl_huge
                ):
                    raise ValueError(
                        f"Conflicting hugepage settings: "
                        f"extra_config.nixl_use_hugepages={nixl_huge!r} vs "
                        f"local_cpu_use_hugepages={self.local_cpu_use_hugepages!r}. "
                        "extra_config.nixl_use_hugepages is deprecated; "
                        "remove it and set local_cpu_use_hugepages only."
                    )
                logger.warning(
                    "extra_config.nixl_use_hugepages is deprecated; applying "
                    "value (%r) to local_cpu_use_hugepages. Update your "
                    "config to set local_cpu_use_hugepages directly.",
                    nixl_huge,
                )
                self.local_cpu_use_hugepages = nixl_huge
            else:
                logger.warning(
                    "extra_config.nixl_use_hugepages is deprecated and has no "
                    "effect for nixl_buffer_device=%r (hugepages apply only to "
                    "the CPU shared pool, controlled by local_cpu_use_hugepages).",
                    self.nixl_buffer_device,
                )
            del self.extra_config["nixl_use_hugepages"]

    return self


def _log_config(self):
    """Log configuration"""
    config_dict = {}
    for name in _CONFIG_DEFINITIONS:
        value = getattr(self, name)
        if name in ["max_local_cpu_size", "max_local_disk_size"]:
            value = f"{value} GB"
        config_dict[name] = value

    logger.info("LMCache Configuration: %s", config_dict)
    return self


def _get_extra_config_value(self, key, default_value=None):
    if hasattr(self, "extra_config") and self.extra_config is not None:
        return self.extra_config.get(key, default_value)
    else:
        return default_value


def _get_lmcache_worker_ids(self, use_mla, world_size):
    if not self.lmcache_worker_ids:
        # if mla is not enabled, return all worker ids, which means start
        # lmcache worker on all ranks as default;
        # if mla is enabled, return [0], which means start lmcache
        # worker on worker 0 as default.
        return [0] if use_mla else list(range(world_size))

    # check the input
    for worker_id in self.lmcache_worker_ids:
        assert -1 < worker_id < world_size
    return self.lmcache_worker_ids


def _get_lookup_server_worker_ids(self, use_mla, world_size):
    if not self.lookup_server_worker_ids:
        # if mla is not enabled, return all worker ids, which means start
        # lookup server on all worker as default;
        # if mla is enabled, return [0], which means start lookup
        # server on worker 0 as default.
        return [0] if use_mla else list(range(world_size))

    # check the input
    for worker_id in self.lookup_server_worker_ids:
        assert -1 < worker_id < world_size
    return self.lookup_server_worker_ids


def _from_legacy(cls, **kwargs):
    """Create configuration from legacy format"""
    backend = kwargs.pop("backend", "cpu")

    # Define backend mappings
    backend_configs = {
        "cpu": {
            "local_cpu": True,
            "max_local_cpu_size": 2,
            "local_disk": None,
            "max_local_disk_size": 0,
            "remote_url": None,
        },
        "local_disk": {
            "local_cpu": False,
            "max_local_cpu_size": 3,
            "local_disk": "local/disk_test/local_disk/",
            "max_local_disk_size": 2,
            "remote_url": None,
        },
        "local_cpu_disk": {
            "local_cpu": True,
            "max_local_cpu_size": 2,
            "local_disk": "local/disk_test/local_disk/",
            "max_local_disk_size": 5,
            "remote_url": None,
        },
        "remote": {"local_cpu": False, "max_local_cpu_size": 2, "local_disk": None},
        "local_cpu_remote": {
            "local_cpu": True,
            "max_local_cpu_size": 2,
            "local_disk": None,
        },
        "local_disk_remote": {
            "local_cpu": False,
            "max_local_cpu_size": 2,
            "local_disk": "local/disk_test/local_disk/",
            "max_local_disk_size": 5,
        },
        "local_cpu_disk_remote": {
            "local_cpu": True,
            "max_local_cpu_size": 2,
            "local_disk": "local/disk_test/local_disk/",
            "max_local_disk_size": 5,
        },
    }

    if backend not in backend_configs:
        raise ValueError(f"Invalid backend: {backend}")

    # Merge configurations
    config_values = {}
    for name, config in _CONFIG_DEFINITIONS.items():
        if name in backend_configs[backend]:
            config_values[name] = backend_configs[backend][name]
        elif name in kwargs:
            config_values[name] = kwargs[name]
        else:
            config_values[name] = config["default"]

    instance = cls(**config_values)
    instance.validate()
    return instance


def _update_config_from_env(self):
    """Update an existing config object with environment variable configurations."""

    def get_env_name(attr_name: str) -> str:
        return f"LMCACHE_{attr_name.upper()}"

    # Collect environment variables
    env_config = {}
    for name in _CONFIG_DEFINITIONS:
        env_name = get_env_name(name)
        env_value = os.getenv(env_name)
        if env_value is not None:
            env_config[name] = env_value

    # Handle deprecated environment variables
    for deprecated_name, new_name in _CONFIG_ALIASES.items():
        env_name = get_env_name(deprecated_name)
        env_value = os.getenv(env_name)
        if env_value is not None:
            env_config[deprecated_name] = env_value

    # Resolve aliases and handle deprecated configurations
    resolved_config = _resolve_config_aliases(
        env_config,
        "environment variables",
        _CONFIG_DEFINITIONS,
        _CONFIG_ALIASES,
        _DEPRECATED_CONFIGS,
    )

    # Ensure _user_set_keys exists
    if not hasattr(self, "_user_set_keys"):
        object.__setattr__(self, "_user_set_keys", set())

    # Update config object with environment values
    for name, config in _CONFIG_DEFINITIONS.items():
        if name in resolved_config:
            try:
                # Parse quoted strings and handle escape characters
                raw_value = resolved_config[name]  # Keep original value for logging
                value = _parse_quoted_string(raw_value)
                converted_value = config["env_converter"](value)
                setattr(self, name, converted_value)
                # Mark as user-set
                self._user_set_keys.add(name)
            except (ValueError, json.JSONDecodeError) as e:
                logger.warning(
                    f"Failed to parse {get_env_name(name)}={raw_value!r}: {e}"
                )
                # Keep existing value if conversion fails
    self.validate()
    return self


# Create configuration class using the base utility
LMCacheEngineConfig = create_config_class(
    config_name="LMCacheEngineConfig",
    config_definitions=_CONFIG_DEFINITIONS,
    config_aliases=_CONFIG_ALIASES,
    deprecated_configs=_DEPRECATED_CONFIGS,
    namespace_extras={
        "validate": _validate_config,
        "log_config": _log_config,
        "get_extra_config_value": _get_extra_config_value,
        "get_lmcache_worker_ids": _get_lmcache_worker_ids,
        "get_lookup_server_worker_ids": _get_lookup_server_worker_ids,
        "from_legacy": classmethod(_from_legacy),
        "update_config_from_env": _update_config_from_env,
    },
)


def load_engine_config_with_overrides(
    config_file_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> "LMCacheEngineConfig":  # type: ignore[valid-type]
    """
    Load engine configuration with support for file, env vars, and overrides.

    This function uses the generic load_config_with_overrides utility from
    config_base.py to reduce code duplication.

    Args:
        config_file_path: Optional direct path to config file
        overrides: Optional dictionary of configuration overrides

    Returns:
        Loaded and validated LMCacheEngineConfig instance
    """

    return load_config_with_overrides(
        config_class=LMCacheEngineConfig,
        config_file_env_var="LMCACHE_CONFIG_FILE",
        config_file_path=config_file_path,
        overrides=overrides,
    )


def _normalize_ec_config_key(raw_key: str, source: str) -> Optional[str]:
    """Normalize one EC-prefixed config key to a LMCache config key."""
    key = raw_key.strip().lower()
    if not key:
        logger.warning("Empty EC config key from %s", source)
        return None

    key = _CONFIG_ALIASES.get(key, key)
    if key not in _CONFIG_DEFINITIONS:
        logger.warning("Unknown EC config key '%s' from %s", raw_key, source)
        return None
    return key


def _collect_ec_overrides_from_env() -> dict[str, Any]:
    """Collect EC-specific overrides from environment variables."""
    overrides: dict[str, Any] = {}
    for env_name, env_value in os.environ.items():
        if not env_name.startswith(_EC_ENV_PREFIX):
            continue
        stripped = env_name[len(_EC_ENV_PREFIX) :]

        normalized_key = _normalize_ec_config_key(
            stripped,
            source=f"environment variable {env_name}",
        )
        if normalized_key is not None:
            overrides[normalized_key] = env_value
    return overrides


def _collect_ec_overrides_from_file(
    config_file_path: Optional[str],
) -> dict[str, Any]:
    """Collect EC-specific overrides from the LMCache YAML config file."""
    if not config_file_path:
        return {}

    try:
        with open(config_file_path, "r", encoding="utf-8") as fin:
            loaded = yaml.safe_load(fin) or {}
    except FileNotFoundError:
        logger.warning(
            "LMCache config file %s not found while loading EC overrides",
            config_file_path,
        )
        return {}

    if not isinstance(loaded, dict):
        logger.warning(
            "LMCache config file %s is not a mapping; skipping EC overrides",
            config_file_path,
        )
        return {}

    overrides: dict[str, Any] = {}
    for raw_key, value in loaded.items():
        if not isinstance(raw_key, str):
            continue

        lower_key = raw_key.lower()
        if lower_key == "ec" and isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if not isinstance(nested_key, str):
                    continue
                normalized_key = _normalize_ec_config_key(
                    nested_key,
                    source=f"config file nested key ec.{nested_key}",
                )
                if normalized_key is not None:
                    overrides[normalized_key] = nested_value
            continue

        if not lower_key.startswith(_EC_FILE_PREFIX):
            continue

        normalized_key = _normalize_ec_config_key(
            lower_key[len(_EC_FILE_PREFIX) :],
            source=f"config file key {raw_key}",
        )
        if normalized_key is not None:
            overrides[normalized_key] = value

    return overrides


def _clone_lmcache_engine_config(
    base_config: "LMCacheEngineConfig",  # type: ignore[valid-type]
) -> "LMCacheEngineConfig":  # type: ignore[valid-type]
    """Create a detached LMCacheEngineConfig copy from an existing config."""
    base_cfg = cast(Any, base_config)
    cloned_config = LMCacheEngineConfig.from_dict(base_cfg.to_dict())
    object.__setattr__(
        cloned_config,
        "_user_set_keys",
        set(getattr(base_cfg, "_user_set_keys", set())),
    )
    return cloned_config


def _apply_ec_storage_defaults(
    config: "LMCacheEngineConfig",  # type: ignore[valid-type]
) -> None:
    """Apply EC-only storage defaults after EC override ingestion."""
    cfg = cast(Any, config)

    if not cfg.enable_pd:
        if not cfg.local_cpu:
            logger.info("EC config enabling local_cpu allocator backend")
            cfg.local_cpu = True
        if cfg.max_local_cpu_size <= 0:
            logger.info("EC config setting max_local_cpu_size to 1 GB")
            cfg.max_local_cpu_size = 1

    if cfg.local_disk and cfg.max_local_disk_size <= 0:
        logger.info("EC config setting max_local_disk_size to 64 GB")
        cfg.max_local_disk_size = 64


def load_ec_engine_config(
    base_config: Optional["LMCacheEngineConfig"] = None,  # type: ignore[valid-type]
    config_file_path: Optional[str] = None,
) -> "LMCacheEngineConfig":  # type: ignore[valid-type]
    """Build EC config from base config plus EC-prefixed overrides.

    Precedence is:
    1) base LMCache config,
    2) YAML keys prefixed with ``ec_`` (or ``ec:`` nested map),
    3) environment variables prefixed with ``LMCACHE_EC_``.
    """
    resolved_base_config = base_config
    if resolved_base_config is None:
        resolved_base_config = load_engine_config_with_overrides(
            config_file_path=config_file_path,
        )

    ec_config = _clone_lmcache_engine_config(resolved_base_config)
    resolved_config_file = config_file_path or os.getenv("LMCACHE_CONFIG_FILE")

    file_overrides = _collect_ec_overrides_from_file(resolved_config_file)
    env_overrides = _collect_ec_overrides_from_env()
    merged_overrides = {**file_overrides, **env_overrides}

    for key, value in merged_overrides.items():
        if not validate_and_set_config_value(ec_config, key, value, override=True):
            logger.warning("Failed to apply EC override %s=%r", key, value)

    _apply_ec_storage_defaults(ec_config)
    cast(Any, ec_config).validate()
    return ec_config
