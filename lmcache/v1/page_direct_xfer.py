# SPDX-License-Identifier: Apache-2.0
"""Pack KVB v2 GPU IOV records from page views and residual copies."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from lmcache.v1.gds_layout import split_pages
from lmcache.v1.kvb_v2 import (
    PAYLOAD_ENCODING_RAW_PAGES,
    KvbV2Profile,
    header_from_layout,
    pack_header,
)
from lmcache.v1.page_direct import PageView


def profile_as_sdk_dict(profile: KvbV2Profile, kv_layout: int = 4) -> Dict[str, int]:
    """Map a Python profile to native ``kvb_store_*`` dict keys."""
    return {
        "header_region_bytes": int(profile.header_region_bytes),
        "direct_capacity": int(profile.direct_capacity),
        "residual_capacity": int(profile.residual_capacity),
        "record_extent_bytes": int(profile.record_extent_bytes),
        "max_pages_per_record": int(profile.max_pages_per_record),
        "max_descriptors": int(profile.max_descriptors),
        "sub_stripe_bytes": int(profile.sub_stripe),
        "sector_bytes": int(profile.sector),
        "align_bytes": int(profile.align),
        "physical_page_format_id": int(profile.physical_page_format_id),
        "kv_layout": int(kv_layout),
    }


def build_raw_page_record(
    views: Sequence[PageView],
    profile: KvbV2Profile,
    *,
    header_ptr: int,
    residual_ptr: int,
    residual_len: int,
    kv_block_index: int,
) -> Tuple[Dict[str, Any], Any]:
    """Split page views and return a GPU IOV record plus the layout.

    Returns:
        ``(record_dict, layout)``. Caller copies ``pack_header`` into
        ``header_ptr`` and gathers residuals into ``residual_ptr``.
    """
    addrs = [int(v.gpu_addr) for v in views]
    layout = split_pages(
        addrs, int(views[0].page_bytes), int(profile.sub_stripe), int(profile.sector)
    )
    header = header_from_layout(layout, profile, PAYLOAD_ENCODING_RAW_PAGES)
    packed = pack_header(header)
    gpu_ptrs: List[int] = []
    record_offs: List[int] = []
    lens: List[int] = []
    for sl in layout.direct_slices:
        gpu_ptrs.append(int(sl.gpu_addr))
        record_offs.append(int(profile.header_region_bytes) + int(sl.object_off))
        lens.append(int(sl.length))
    rec = {
        "kv_block_index": int(kv_block_index),
        "payload_encoding": PAYLOAD_ENCODING_RAW_PAGES,
        "header_ptr": int(header_ptr),
        "header_is_device": 1,
        "gpu_ptrs": gpu_ptrs,
        "record_offs": record_offs,
        "lens": lens,
        "residual_ptr": int(residual_ptr) if residual_len else 0,
        "residual_len": int(residual_len),
        "packed_header": packed,
        "layout": layout,
    }
    return rec, layout
