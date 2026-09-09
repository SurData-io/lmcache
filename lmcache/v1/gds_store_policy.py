# SPDX-License-Identifier: Apache-2.0
"""GDS store policy helpers (no vLLM import)."""

GDS_STORE_OVERLOAD_IMMEDIATE_DROP = "immediate_drop"
GDS_STORE_OVERLOAD_IMMEDIATE_D2H = "immediate_d2h_fallback"
_GDS_STORE_OVERLOAD_POLICIES = (
    GDS_STORE_OVERLOAD_IMMEDIATE_DROP,
    GDS_STORE_OVERLOAD_IMMEDIATE_D2H,
)


def gds_store_overload_policy(raw: object) -> str:
    """Normalize the GDS store overload policy string.

    Args:
        raw: Config value. Empty/None uses ``immediate_drop``.

    Returns:
        A known policy name.

    Raises:
        ValueError: Unknown policy name.
    """
    name = str(raw or GDS_STORE_OVERLOAD_IMMEDIATE_DROP).strip().lower()
    if not name:
        name = GDS_STORE_OVERLOAD_IMMEDIATE_DROP
    if name not in _GDS_STORE_OVERLOAD_POLICIES:
        raise ValueError(
            "kv_gds_store_overload_policy must be one of %s, got %r"
            % (_GDS_STORE_OVERLOAD_POLICIES, raw)
        )
    return name


def gds_direct_store_max_inflight(raw: object) -> int:
    """Parse the page-direct GDS store permit cap.

    Args:
        raw: Config value. ``0`` means unlimited (permit always granted).

    Returns:
        Non-negative inflight cap.

    Raises:
        ValueError: Negative value.
    """
    cap = int(raw or 0)
    if cap < 0:
        raise ValueError("kv_gds_direct_store_max_inflight must be >= 0, got %r" % raw)
    return cap


def gds_defer_partial_prefill_store(kv_io_mode: object, is_last_prefill: bool) -> bool:
    """Return True when a GDS store should wait for the last prefill slice.

    Args:
        kv_io_mode: Engine I/O mode (``gds`` or ``dram``).
        is_last_prefill: True when this save covers the finished prompt.

    Returns:
        True if this save must be skipped so hashes stay in one prefix file.

    Notes:
        Chunked prefill issues a short GPU IOV store, then ``pending_saves``
        skips later slices while serialize I/O is still in flight. MN then
        only publishes the first slice (GPU 32×2: ``direct=8..1``,
        ``token_hit~50%``). One store on last prefill keeps the chain intact.
    """
    return str(kv_io_mode or "dram").lower() == "gds" and not is_last_prefill


def gds_skip_exists_for_empty_ns(
    enabled: object, configured_stamp: object, run_stamp: object
) -> bool:
    """Return True only for a stamp-checked empty-namespace exists skip.

    This is a benchmark knob. A process that has not stored yet is not a
    valid skip condition: another process or a reused namespace may already
    hold KV. Both stamps must be non-empty and equal.

    Args:
        enabled: Config/extra flag. Default-false strings and None skip.
        configured_stamp: ``gds_empty_ns_stamp`` from config.
        run_stamp: Current run stamp (env or extra_config).

    Returns:
        True when exists RPCs may return a synthetic miss without I/O.
    """
    if isinstance(enabled, bool):
        flag = enabled
    else:
        flag = str(enabled or "").strip().lower() in ("1", "true", "yes", "on")
    if not flag:
        return False
    stamp = str(configured_stamp or "").strip()
    run = str(run_stamp or "").strip()
    return bool(stamp) and stamp == run


def gds_io_pool_workers(get_workers: int, store_slots: int) -> int:
    """Thread count for the GDS GPU-IOV executor.

    Args:
        get_workers: ``kv_gds_get_workers`` (retrieve).
        store_slots: ``kv_gds_store_slots`` (store bounce slots).

    Returns:
        At least 32 workers so last-prefill stores are not queued behind a
        4-thread pool (STAMP ``20260905-154343``: 3/32 durable commits).
    """
    get_w = max(1, int(get_workers or 4))
    store_w = max(0, int(store_slots or 0))
    return max(get_w, store_w, 32)

