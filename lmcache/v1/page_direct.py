# SPDX-License-Identifier: Apache-2.0
"""Page-direct views, zero-residual probe, and GDS range fusion.

Connector-facing geometry: a page is one contiguous vLLM block slot on one
layer (or fused K/V tensor). ``slot_mapping`` values are token slot ids, not
pointers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

# First Party
from lmcache.v1.gds_layout import lcm, split_pages
from lmcache.v1.kvb_v2 import required_block_multiple


@dataclass(frozen=True)
class PageView:
    """One contiguous GPU page covering a full vLLM block on one layer."""

    gpu_addr: int
    page_bytes: int
    layer_index: int
    block_id: int
    group_id: int
    is_contiguous: bool
    is_gds_registered: bool
    allocation_id: int
    range_id: int = -1

    @property
    def end_addr(self) -> int:
        return int(self.gpu_addr) + int(self.page_bytes)


@dataclass(frozen=True)
class RegisteredRun:
    """One cuFile-registered contiguous VA run inside a single allocation."""

    allocation_id: int
    start: int
    nbytes: int
    range_id: int


@dataclass
class PageDirectProbe:
    """Runtime geometry probe. Never mutates vLLM ``--block-size``."""

    page_bytes: int
    block_size: int
    bytes_per_token_per_layer: int
    sub_stripe: int
    sector: int
    align: int
    required_block_multiple: int
    zero_residual: bool
    warn: str = ""


@dataclass
class ChunkPageAdmission:
    """Whether a token span maps to complete, non-overlapping vLLM pages."""

    complete_pages: bool
    page_views: List[PageView]
    reason: str = ""


def compute_align(sub_stripe: int, sector: int = 4096) -> int:
    return lcm(int(sub_stripe), int(sector))


def probe_zero_residual(
    pages: Sequence[PageView],
    *,
    block_size: int,
    sub_stripe: int,
    sector: int = 4096,
) -> PageDirectProbe:
    """Evaluate the zero-residual fast path from enumerated page views.

    Args:
        pages: Connector-enumerated page views (may be a sample of all pages).
        block_size: vLLM tokens per page (already allocated; not mutated).
        sub_stripe: SDK sub-stripe bytes.
        sector: Disk sector bytes.

    Returns:
        :class:`PageDirectProbe`. ``zero_residual`` is true only when every
        page is contiguous, registered, and both address and length are
        aligned. Failure never attempts to change block_size.
    """
    ss = int(sub_stripe)
    sec = int(sector)
    align = compute_align(ss, sec)
    bs = int(block_size)
    if not pages:
        return PageDirectProbe(
            page_bytes=0,
            block_size=bs,
            bytes_per_token_per_layer=0,
            sub_stripe=ss,
            sector=sec,
            align=align,
            required_block_multiple=1,
            zero_residual=False,
            warn="no page views enumerated",
        )
    page_bytes = int(pages[0].page_bytes)
    bpt = page_bytes // bs if bs > 0 else 0
    req = required_block_multiple(align, bpt) if bpt > 0 else align
    ok = True
    reasons: List[str] = []
    for p in pages:
        if int(p.page_bytes) != page_bytes:
            ok = False
            reasons.append("heterogeneous page_bytes")
            break
        if not p.is_contiguous:
            ok = False
            reasons.append("page not contiguous")
            break
        if int(p.gpu_addr) % align != 0:
            ok = False
            reasons.append("page gpu_addr not align")
            break
        if int(p.page_bytes) % align != 0:
            ok = False
            reasons.append("page_bytes not multiple of align")
            break
        if not p.is_gds_registered:
            ok = False
            reasons.append("page not GDS-registered")
            break
    warn = ""
    if not ok:
        warn = (
            "zero_residual unavailable page_bytes=%s align=%s "
            "required_block_multiple=%s current_block_size=%s (%s)"
            % (page_bytes, align, req, bs, "; ".join(reasons) or "unspecified")
        )
    return PageDirectProbe(
        page_bytes=page_bytes,
        block_size=bs,
        bytes_per_token_per_layer=bpt,
        sub_stripe=ss,
        sector=sec,
        align=align,
        required_block_multiple=req,
        zero_residual=ok,
        warn=warn,
    )


def launcher_block_size(current: int, required_multiple: int) -> int:
    """Round ``current`` up to a multiple of ``required_multiple``.

    Used by the bench launcher *before* vLLM allocates KV cache.
    """
    cur = int(current)
    mult = int(required_multiple)
    if cur <= 0 or mult <= 0:
        raise ValueError("block_size and required_multiple must be positive")
    if cur % mult == 0:
        return cur
    return ((cur + mult - 1) // mult) * mult


def gds_serve_block_size(
    *,
    sub_stripe: int,
    bytes_per_token_per_layer: int,
    current_block_size: int = 16,
    sector: int = 4096,
) -> int:
    """Return the vLLM ``--block-size`` that makes one page a multiple of align.

    ``align = lcm(sub_stripe, sector)``. The result is the smallest multiple of
    ``required_block_multiple(align, bytes_per_token_per_layer)`` that is at
    least ``current_block_size``. Call this before vLLM allocates KV cache.

    Args:
        sub_stripe: SurFS/SDK sub-stripe bytes (must match serve geometry).
        bytes_per_token_per_layer: Physical page bytes / tokens per page.
        current_block_size: vLLM default or user-requested tokens per page.
        sector: Disk sector bytes.

    Returns:
        Positive block_size to pass as ``--block-size``.

    Raises:
        ValueError: If any size argument is not positive.
    """
    align = compute_align(int(sub_stripe), int(sector))
    req = required_block_multiple(align, int(bytes_per_token_per_layer))
    return launcher_block_size(int(current_block_size), req)


def slot_ids_to_block_coverage(
    slot_ids: Sequence[int],
    block_size: int,
) -> Tuple[bool, List[int], str]:
    """Return whether slots cover complete unique vLLM pages.

    Args:
        slot_ids: Token slot ids for one KVB chunk (length == chunk tokens).
        block_size: vLLM tokens per page.

    Returns:
        (complete, sorted unique block_ids, reason).
    """
    bs = int(block_size)
    if bs <= 0:
        return False, [], "block_size <= 0"
    if not slot_ids:
        return False, [], "empty slot_ids"
    n = len(slot_ids)
    if n % bs != 0:
        return False, [], "chunk_len %% vllm_block_size != 0"
    seen: dict[int, set[int]] = {}
    for slot in slot_ids:
        s = int(slot)
        if s < 0:
            return False, [], "negative/prefix-cached slot"
        block_id = s // bs
        off = s % bs
        bucket = seen.setdefault(block_id, set())
        if off in bucket:
            return False, [], "duplicate slot in page"
        bucket.add(off)
    for block_id, offs in seen.items():
        if offs != set(range(bs)):
            return False, [], "partial page coverage"
    # Detect alias: same physical page appearing twice as different runs is
    # already covered by duplicate slot. Cross-chunk alias is the caller's
    # pin problem; we only require unique block ids in this span.
    block_ids = sorted(seen.keys())
    return True, block_ids, ""


def enumerate_page_views(
    *,
    layer_bases: Sequence[Tuple[int, int, int]],
    block_ids: Sequence[int],
    page_bytes: int,
    num_blocks: int,
    registered_runs: Sequence[RegisteredRun],
) -> List[PageView]:
    """Build page views for ``block_ids`` across layer bases.

    ``layer_bases`` entries are ``(base_ptr, allocation_id, group_id)``.
    Page address is ``base + block_id * page_bytes`` (contiguous paged pool).
    A page is GDS-registered iff some run with the same allocation_id covers
    ``[gpu_addr, gpu_addr+page_bytes)``.
    """
    pb = int(page_bytes)
    nb = int(num_blocks)
    views: List[PageView] = []
    for layer_index, (base, alloc_id, group_id) in enumerate(layer_bases):
        base_i = int(base)
        for block_id in block_ids:
            bid = int(block_id)
            if bid < 0 or (nb > 0 and bid >= nb):
                views.append(
                    PageView(
                        gpu_addr=0,
                        page_bytes=pb,
                        layer_index=layer_index,
                        block_id=bid,
                        group_id=int(group_id),
                        is_contiguous=False,
                        is_gds_registered=False,
                        allocation_id=int(alloc_id),
                    )
                )
                continue
            addr = base_i + bid * pb
            covered = False
            range_id = -1
            for run in registered_runs:
                if int(run.allocation_id) != int(alloc_id):
                    continue
                if addr >= int(run.start) and addr + pb <= int(run.start) + int(
                    run.nbytes
                ):
                    covered = True
                    range_id = int(run.range_id)
                    break
            views.append(
                PageView(
                    gpu_addr=addr,
                    page_bytes=pb,
                    layer_index=layer_index,
                    block_id=bid,
                    group_id=int(group_id),
                    is_contiguous=True,
                    is_gds_registered=covered,
                    allocation_id=int(alloc_id),
                    range_id=range_id,
                )
            )
    return views


def fuse_allocation_runs(
    ranges: Sequence[Tuple[int, int, int]],
    *,
    sector: int,
    registration_limit: int,
) -> Tuple[List[RegisteredRun], str]:
    """Fuse contiguous ranges that share an allocation id.

    Args:
        ranges: ``(start, nbytes, allocation_id)`` for each known allocation
            slice. Adjacent same-allocation ranges with no VA gap may merge.
        sector: Required alignment of start and length.
        registration_limit: Max registered runs (staging arena uses +1).

    Returns:
        (runs, error). On error, runs may be empty and page-direct is disabled.
    """
    sec = int(sector)
    items = sorted(
        ((int(s), int(n), int(a)) for s, n, a in ranges),
        key=lambda t: (t[2], t[0]),
    )
    fused: List[Tuple[int, int, int]] = []
    for start, nbytes, alloc in items:
        if nbytes <= 0:
            return [], "empty range"
        if start % sec != 0 or nbytes % sec != 0:
            return [], "range not sector aligned"
        if not fused or fused[-1][2] != alloc:
            fused.append((start, nbytes, alloc))
            continue
        prev_s, prev_n, prev_a = fused[-1]
        prev_end = prev_s + prev_n
        if start < prev_end:
            return [], "overlapping ranges in allocation"
        if start == prev_end:
            fused[-1] = (prev_s, prev_n + nbytes, prev_a)
        else:
            # Gap inside the same allocation is a separate run, not a merge.
            fused.append((start, nbytes, alloc))
    if len(fused) + 1 > int(registration_limit):
        return [], "registered_runs + staging_arena exceed gds_registration_limit"
    out = [
        RegisteredRun(
            allocation_id=a, start=s, nbytes=n, range_id=i
        )
        for i, (s, n, a) in enumerate(fused)
    ]
    return out, ""


def gpu_addrs_from_views(views: Sequence[PageView]) -> List[int]:
    return [int(v.gpu_addr) for v in views]


def split_views(views: Sequence[PageView], sub_stripe: int, sector: int = 4096):
    """Run ``split_pages`` on enumerated views (same page_bytes required)."""
    if not views:
        raise ValueError("no page views")
    pb = int(views[0].page_bytes)
    for v in views:
        if int(v.page_bytes) != pb:
            raise ValueError("heterogeneous page_bytes in split_views")
    return split_pages(gpu_addrs_from_views(views), pb, int(sub_stripe), int(sector))
