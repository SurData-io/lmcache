# SPDX-License-Identifier: Apache-2.0
"""KVB v2 split-region record header, encodings, and profile sizing.

``gds_layout.LAYOUT_VERSION`` is the record-header format version. SDK/MN
layout identity is ``SURDATA_SDK_KV_LAYOUT_SPLIT_REGION`` (value 4).
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

# First Party
from lmcache.v1.gds_layout import pad_to

SURDATA_SDK_KV_LAYOUT_CONTIG = 3
SURDATA_SDK_KV_LAYOUT_SPLIT_REGION = 4

HEADER_MAGIC = b"KVB2SR01"
HEADER_LAYOUT_VERSION = 1
FIXED_HEADER_BYTES = 96
DESCRIPTOR_BYTES = 40

PAYLOAD_ENCODING_RAW_PAGES = 1
PAYLOAD_ENCODING_CANONICAL_KV2LTD = 2

_FIXED_FMT = "<8sIIQ7Q4I"
# magic, layout_version, payload_encoding, format_id,
# sub_stripe, sector, align, header_region, direct_cap, residual_cap,
# record_extent, page_count, descriptor_count, max_descriptors, reserved
assert struct.calcsize(_FIXED_FMT) == FIXED_HEADER_BYTES
_DESC_FMT = "<5Q"
assert struct.calcsize(_DESC_FMT) == DESCRIPTOR_BYTES


def gcd_positive(a: int, b: int) -> int:
    a = int(a)
    b = int(b)
    if a <= 0 or b <= 0:
        raise ValueError("gcd requires positive integers")
    return math.gcd(a, b)


def required_block_multiple(align: int, bytes_per_token_per_layer: int) -> int:
    """Smallest block_size multiple that makes page_bytes a multiple of align.

    Args:
        align: lcm(sub_stripe, sector).
        bytes_per_token_per_layer: page_bytes / vllm_block_size.

    Returns:
        Positive integer ``align / gcd(align, bytes_per_token_per_layer)``.
    """
    align_i = int(align)
    bpt = int(bytes_per_token_per_layer)
    if align_i <= 0 or bpt <= 0:
        raise ValueError("align and bytes_per_token_per_layer must be positive")
    return align_i // gcd_positive(align_i, bpt)


def physical_page_format_id(
    *,
    format_name: str,
    kv_layout: str,
    dtype_itemsize: int,
    vllm_block_size: int,
    page_bytes: int,
    layer_count: int,
    kv_factor: int,
    enum_order: str,
) -> int:
    """Stable 64-bit identity for the raw-page byte layout.

    Args:
        format_name: EngineKVFormat / KVFormatSpec name.
        kv_layout: ``NHD`` or ``HND``.
        dtype_itemsize: Element size in bytes.
        vllm_block_size: Tokens per vLLM page.
        page_bytes: Physical bytes per page view.
        layer_count: Number of layers in the enumeration.
        kv_factor: 1 for fused MLA, 2 for split K/V.
        enum_order: Deterministic page enumeration order id.

    Returns:
        Non-zero 64-bit truncated SHA-256 of the canonical signature.
    """
    blob = "|".join(
        (
            str(format_name),
            str(kv_layout),
            str(int(dtype_itemsize)),
            str(int(vllm_block_size)),
            str(int(page_bytes)),
            str(int(layer_count)),
            str(int(kv_factor)),
            str(enum_order),
        )
    ).encode("utf-8")
    digest = hashlib.sha256(blob).digest()
    return int.from_bytes(digest[:8], "little") or 1


@dataclass(frozen=True)
class PageDescriptor:
    page_payload_off: int
    body_record_off: int
    body_len: int
    residual_record_off: int
    residual_len: int


@dataclass
class KvbV2Header:
    layout_version: int
    payload_encoding: int
    physical_page_format_id: int
    sub_stripe: int
    sector: int
    align: int
    header_region_bytes: int
    direct_capacity: int
    residual_capacity: int
    record_extent_bytes: int
    page_count: int
    descriptor_count: int
    max_descriptors: int
    descriptors: List[PageDescriptor]


@dataclass(frozen=True)
class KvbV2Profile:
    header_region_bytes: int
    direct_capacity: int
    residual_capacity: int
    record_extent_bytes: int
    max_pages_per_record: int
    max_descriptors: int
    sub_stripe: int
    sector: int
    align: int
    page_bytes: int
    physical_page_format_id: int


def header_region_bytes(max_descriptors: int, sub_stripe: int) -> int:
    """Pad fixed header + descriptor table to sub-stripe.

    Args:
        max_descriptors: Locked descriptor slots in the profile.
        sub_stripe: Sub-stripe size in bytes.

    Returns:
        Physical header region length, a multiple of ``sub_stripe``.
    """
    n = int(max_descriptors)
    if n < 0:
        raise ValueError("max_descriptors must be >= 0")
    raw = FIXED_HEADER_BYTES + n * DESCRIPTOR_BYTES
    return pad_to(raw, int(sub_stripe))


def compute_profile(
    *,
    max_pages: int,
    page_bytes: int,
    sub_stripe: int,
    sector: int,
    align: int,
    physical_page_format_id: int,
    canonical_payload_bytes: int = 0,
) -> KvbV2Profile:
    """Lock a fixed-stride KVB v2 record profile.

    Direct capacity is the worst-case packed body (all pages fully aligned).
    Residual capacity is the worst-case residual (all pages fully residual)
    padded to sub-stripe. Profile extent also covers a canonical KV_2LTD
    fallback payload of ``canonical_payload_bytes``.

    Args:
        max_pages: Maximum page views in one record.
        page_bytes: Bytes per page view.
        sub_stripe: Sub-stripe bytes.
        sector: Sector bytes.
        align: lcm(sub_stripe, sector).
        physical_page_format_id: Immutable format signature.
        canonical_payload_bytes: Max host fallback payload (logical).

    Returns:
        Locked :class:`KvbV2Profile`.

    Raises:
        ValueError: If geometry is invalid or max_pages is too large.
    """
    max_pages_i = int(max_pages)
    page_i = int(page_bytes)
    ss = int(sub_stripe)
    sec = int(sector)
    al = int(align)
    if max_pages_i <= 0:
        raise ValueError("max_pages must be positive")
    if page_i <= 0 or ss <= 0 or sec <= 0 or al <= 0:
        raise ValueError("page/sub_stripe/sector/align must be positive")
    if al % ss != 0 or al % sec != 0:
        raise ValueError("align must be a multiple of sub_stripe and sector")
    max_desc = max_pages_i
    hdr = header_region_bytes(max_desc, ss)
    direct_logical = max_pages_i * page_i
    residual_logical = max_pages_i * page_i
    direct_cap = pad_to(direct_logical, ss)
    residual_cap = pad_to(residual_logical, ss)
    canon = pad_to(int(canonical_payload_bytes), ss) if canonical_payload_bytes else 0
    payload_cap = max(direct_cap + residual_cap, canon)
    if payload_cap == canon and canon > 0:
        residual_cap = 0
        direct_cap = canon
        payload_cap = direct_cap + residual_cap
    # Always keep room for both encodings: max of (direct+residual) and canon.
    if canon > direct_cap + residual_cap:
        extra = canon - (direct_cap + residual_cap)
        residual_cap += extra
    extent = hdr + direct_cap + residual_cap
    if extent % ss != 0:
        raise ValueError("record_extent_bytes must be a multiple of sub_stripe")
    return KvbV2Profile(
        header_region_bytes=hdr,
        direct_capacity=direct_cap,
        residual_capacity=residual_cap,
        record_extent_bytes=extent,
        max_pages_per_record=max_pages_i,
        max_descriptors=max_desc,
        sub_stripe=ss,
        sector=sec,
        align=al,
        page_bytes=page_i,
        physical_page_format_id=int(physical_page_format_id),
    )


def pack_header(header: KvbV2Header) -> bytes:
    """Serialize a header into a header-region-sized buffer (zero padded).

    Args:
        header: Validated header.

    Returns:
        Bytes of length ``header.header_region_bytes``.

    Raises:
        ValueError: If descriptor count exceeds max or region is too small.
    """
    if header.descriptor_count != len(header.descriptors):
        raise ValueError("descriptor_count does not match descriptors")
    if header.descriptor_count > header.max_descriptors:
        raise ValueError("descriptor_count exceeds max_descriptors")
    if header.layout_version != HEADER_LAYOUT_VERSION:
        raise ValueError("unsupported header layout_version")
    if header.payload_encoding not in (
        PAYLOAD_ENCODING_RAW_PAGES,
        PAYLOAD_ENCODING_CANONICAL_KV2LTD,
    ):
        raise ValueError("unknown payload_encoding")
    need = FIXED_HEADER_BYTES + header.max_descriptors * DESCRIPTOR_BYTES
    if header.header_region_bytes < need:
        raise ValueError("header_region_bytes too small for descriptor table")
    fixed = struct.pack(
        _FIXED_FMT,
        HEADER_MAGIC,
        int(header.layout_version),
        int(header.payload_encoding),
        int(header.physical_page_format_id),
        int(header.sub_stripe),
        int(header.sector),
        int(header.align),
        int(header.header_region_bytes),
        int(header.direct_capacity),
        int(header.residual_capacity),
        int(header.record_extent_bytes),
        int(header.page_count),
        int(header.descriptor_count),
        int(header.max_descriptors),
        0,
    )
    parts = [fixed]
    for desc in header.descriptors:
        parts.append(
            struct.pack(
                _DESC_FMT,
                int(desc.page_payload_off),
                int(desc.body_record_off),
                int(desc.body_len),
                int(desc.residual_record_off),
                int(desc.residual_len),
            )
        )
    unused = header.max_descriptors - header.descriptor_count
    if unused:
        parts.append(b"\x00" * (unused * DESCRIPTOR_BYTES))
    blob = b"".join(parts)
    if len(blob) > header.header_region_bytes:
        raise ValueError("packed header exceeds header_region_bytes")
    if len(blob) < header.header_region_bytes:
        blob += b"\x00" * (header.header_region_bytes - len(blob))
    return blob


def header_needed_bytes(fixed_header: bytes) -> int:
    """Return how many bytes ``unpack_header`` needs from a 96-byte prefix.

    Args:
        fixed_header: At least ``FIXED_HEADER_BYTES`` of the packed header.

    Returns:
        ``max(header_region_bytes, 96 + 40 * descriptor_count)``.

    Raises:
        ValueError: Buffer too small or ``header_region_bytes`` implausible.
    """
    if len(fixed_header) < FIXED_HEADER_BYTES:
        raise ValueError("header buffer too small")
    unpacked = struct.unpack(_FIXED_FMT, fixed_header[:FIXED_HEADER_BYTES])
    header_region = int(unpacked[7])
    descriptor_count = int(unpacked[12])
    need = FIXED_HEADER_BYTES + int(descriptor_count) * DESCRIPTOR_BYTES
    if header_region > 0:
        need = max(need, header_region)
    if need > 1 << 20:
        raise ValueError("header_region_bytes implausible")
    return need


def header_blob_from_buffer(view: Union[bytes, bytearray, memoryview]) -> bytes:
    """Copy the full header region (not just 96 bytes) for ``unpack_header``.

    Args:
        view: Packed KVB v2 record prefix (header region, possibly followed
            by payload).

    Returns:
        ``bytes`` of length ``header_needed_bytes``.

    Raises:
        ValueError: Truncated before the descriptor table or header region.
    """
    if len(view) < FIXED_HEADER_BYTES:
        raise ValueError("header buffer too small")
    fixed = bytes(view[:FIXED_HEADER_BYTES])
    need = header_needed_bytes(fixed)
    if len(view) < need:
        raise ValueError("header truncated before descriptors")
    return bytes(view[:need])


def unpack_header(buf: bytes, expected_format_id: int = 0) -> KvbV2Header:
    """Parse and validate a KVB v2 header.

    Args:
        buf: At least ``FIXED_HEADER_BYTES`` of header-region bytes.
        expected_format_id: If non-zero, must match stored format id.

    Returns:
        Parsed :class:`KvbV2Header`.

    Raises:
        ValueError: Corrupt, unknown encoding, or format mismatch.
    """
    if len(buf) < FIXED_HEADER_BYTES:
        raise ValueError("header buffer too small")
    unpacked = struct.unpack(_FIXED_FMT, buf[:FIXED_HEADER_BYTES])
    magic = unpacked[0]
    if magic != HEADER_MAGIC:
        raise ValueError("bad KVB v2 magic")
    layout_version = unpacked[1]
    encoding = unpacked[2]
    format_id = unpacked[3]
    if layout_version != HEADER_LAYOUT_VERSION:
        raise ValueError("unsupported header layout_version")
    if encoding not in (
        PAYLOAD_ENCODING_RAW_PAGES,
        PAYLOAD_ENCODING_CANONICAL_KV2LTD,
    ):
        raise ValueError("unknown payload_encoding")
    if expected_format_id and int(format_id) != int(expected_format_id):
        raise ValueError("physical_page_format_id mismatch")
    (
        _magic,
        _lv,
        _enc,
        _fid,
        sub_stripe,
        sector,
        align,
        header_region,
        direct_cap,
        residual_cap,
        record_extent,
        page_count,
        descriptor_count,
        max_descriptors,
        _reserved,
    ) = unpacked
    if descriptor_count > max_descriptors:
        raise ValueError("descriptor_count exceeds max_descriptors")
    table_bytes = max_descriptors * DESCRIPTOR_BYTES
    if len(buf) < FIXED_HEADER_BYTES + descriptor_count * DESCRIPTOR_BYTES:
        raise ValueError("header truncated before descriptors")
    descs: List[PageDescriptor] = []
    off = FIXED_HEADER_BYTES
    for _ in range(descriptor_count):
        vals = struct.unpack(_DESC_FMT, buf[off : off + DESCRIPTOR_BYTES])
        descs.append(
            PageDescriptor(
                page_payload_off=vals[0],
                body_record_off=vals[1],
                body_len=vals[2],
                residual_record_off=vals[3],
                residual_len=vals[4],
            )
        )
        off += DESCRIPTOR_BYTES
    _ = table_bytes
    return KvbV2Header(
        layout_version=layout_version,
        payload_encoding=encoding,
        physical_page_format_id=format_id,
        sub_stripe=sub_stripe,
        sector=sector,
        align=align,
        header_region_bytes=header_region,
        direct_capacity=direct_cap,
        residual_capacity=residual_cap,
        record_extent_bytes=record_extent,
        page_count=page_count,
        descriptor_count=descriptor_count,
        max_descriptors=max_descriptors,
        descriptors=descs,
    )


def descriptors_from_split(
    layout,
    header_region_bytes_v: int,
) -> List[PageDescriptor]:
    """Build page descriptors from a :class:`LayoutDescriptor`.

    Direct slices are packed at ``header_region_bytes + object_off``.
    Residuals are packed after ``header_region + direct_phys_len``.
    ``split_pages`` emits at most one body DirectSlice per page, in page
    order, and only when that page has a non-zero aligned body.

    Args:
        layout: Result of ``split_pages``.
        header_region_bytes_v: Locked header region size.

    Returns:
        One descriptor per page (body and/or residual may be zero).
    """
    residual_base = int(header_region_bytes_v) + int(layout.direct_phys_len)
    page_count = int(layout.page_count)
    body_by_page: List[Tuple[int, int]] = [(0, 0)] * page_count
    body_iter = iter(layout.direct_slices)
    residual_len = [0] * page_count
    residual_off = [0] * page_count
    for res in layout.residuals:
        idx = int(res.page_index)
        if residual_len[idx] == 0:
            residual_off[idx] = residual_base + int(res.staging_off)
        residual_len[idx] += int(res.length)
    for i in range(page_count):
        covered = residual_len[i]
        if covered < int(layout.page_bytes):
            sl = next(body_iter, None)
            if sl is None:
                raise ValueError("direct slice missing for page %s" % i)
            body_by_page[i] = (
                int(header_region_bytes_v) + int(sl.object_off),
                int(sl.length),
            )
            if body_by_page[i][1] + covered != int(layout.page_bytes):
                raise ValueError("page %s body+residual does not cover page" % i)
        elif covered != int(layout.page_bytes):
            raise ValueError("page %s residual exceeds page_bytes" % i)
    leftover = next(body_iter, None)
    if leftover is not None:
        raise ValueError("unexpected extra direct slice")
    out: List[PageDescriptor] = []
    for i in range(page_count):
        boff, blen = body_by_page[i]
        head = 0
        for res in layout.residuals:
            if int(res.page_index) == i and int(res.page_off) == 0:
                head = int(res.length)
                break
        out.append(
            PageDescriptor(
                page_payload_off=head if blen else 0,
                body_record_off=boff,
                body_len=blen,
                residual_record_off=residual_off[i] if residual_len[i] else 0,
                residual_len=residual_len[i],
            )
        )
    return out


def header_from_layout(
    layout,
    profile: KvbV2Profile,
    payload_encoding: int,
) -> KvbV2Header:
    """Build a KVB v2 header for a split layout under a locked profile.

    Args:
        layout: ``split_pages`` result.
        profile: Locked record profile.
        payload_encoding: RAW_PAGES or CANONICAL_KV2LTD.

    Returns:
        Header with descriptors sized to the profile.

    Raises:
        ValueError: If page count exceeds profile max_pages.
    """
    if int(layout.page_count) > profile.max_pages_per_record:
        raise ValueError("page_count exceeds max_pages_per_record")
    descs = descriptors_from_split(layout, profile.header_region_bytes)
    return KvbV2Header(
        layout_version=HEADER_LAYOUT_VERSION,
        payload_encoding=int(payload_encoding),
        physical_page_format_id=profile.physical_page_format_id,
        sub_stripe=profile.sub_stripe,
        sector=profile.sector,
        align=profile.align,
        header_region_bytes=profile.header_region_bytes,
        direct_capacity=profile.direct_capacity,
        residual_capacity=profile.residual_capacity,
        record_extent_bytes=profile.record_extent_bytes,
        page_count=int(layout.page_count),
        descriptor_count=len(descs),
        max_descriptors=profile.max_descriptors,
        descriptors=descs,
    )
