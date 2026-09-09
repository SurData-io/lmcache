"""Miss-only lookup retry policy (no engine / ZMQ imports)."""


def should_relookup_cached_miss(
    cached_hits,
    first_lookup_ts,
    now: float,
    *,
    enabled: bool,
    min_interval_ms: int,
) -> bool:
    """True when a terminal 0-hit lookup is old enough to retry.

    Hits >0 and in-flight lookups must not be restarted.
    """
    return cached_miss_action(
        cached_hits,
        first_lookup_ts,
        now,
        enabled=enabled,
        min_interval_ms=min_interval_ms,
        attempts=0,
        max_attempts=1,
    ) == "retry"


def cached_miss_action(
    cached_hits,
    first_lookup_ts,
    now: float,
    *,
    enabled: bool,
    min_interval_ms: int,
    attempts: int,
    max_attempts: int,
) -> str:
    """Scheduler action for a cached miss: ``commit``, ``hold``, or ``retry``.

    ``commit``: return 0 so vLLM can prefill (do not bounce back to lookup).
    ``hold``: interval not elapsed; keep the request deferred without a new lookup.
    ``retry``: drop the cached 0 and issue lookup() once more.

    Infinite 0→-1 loops starve turn-1 prefills: every waiting miss would be
    converted back to in-flight after ``relookup_min_interval_ms``.
    """
    if not enabled or cached_hits != 0:
        return "commit"
    if int(attempts) >= max(0, int(max_attempts)):
        return "commit"
    if first_lookup_ts is None:
        return "commit"
    interval_s = max(0, int(min_interval_ms)) / 1000.0
    if (now - float(first_lookup_ts)) < interval_s:
        return "hold"
    return "retry"
