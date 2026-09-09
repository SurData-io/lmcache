"""Parameterized split-region layout for page-direct GDS.

Splitter is geometry-driven: no assumed residual size (e.g. 2048 B).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

LAYOUT_VERSION = 1


def lcm(a: int, b: int) -> int:
    a = int(a)
    b = int(b)
    if a <= 0 or b <= 0:
        raise ValueError("lcm requires positive integers")
    return a // math.gcd(a, b) * b


@dataclass(frozen=True)
class DirectSlice:
    gpu_addr: int
    object_off: int
    length: int


@dataclass(frozen=True)
class ResidualDesc:
    page_index: int
    page_off: int
    staging_off: int
    length: int


@dataclass
class LayoutDescriptor:
    layout_version: int
    page_bytes: int
    page_count: int
    sub_stripe: int
    sector: int
    align: int
    direct_logical_len: int
    residual_logical_len: int
    direct_phys_len: int
    residual_phys_len: int
    direct_slices: List[DirectSlice] = field(default_factory=list)
    residuals: List[ResidualDesc] = field(default_factory=list)
    published: bool = False

    def geometry_ok(self) -> bool:
        if self.layout_version != LAYOUT_VERSION:
            return False
        if self.sub_stripe <= 0 or self.sector <= 0 or self.align <= 0:
            return False
        if self.direct_phys_len % self.sub_stripe != 0:
            return False
        if self.residual_phys_len % self.sub_stripe != 0:
            return False
        if self.page_count < 0 or self.page_bytes < 0:
            return False
        covered = 0
        last_off = -1
        for sl in self.direct_slices:
            if sl.length <= 0:
                return False
            if sl.gpu_addr % self.align != 0:
                return False
            if sl.length % self.align != 0:
                return False
            if sl.object_off < last_off:
                return False
            last_off = sl.object_off + sl.length
            covered += sl.length
        if covered != self.direct_logical_len:
            return False
        res_cov = 0
        for res in self.residuals:
            if res.length <= 0:
                return False
            if res.page_index < 0 or res.page_index >= self.page_count:
                return False
            res_cov += res.length
        if res_cov != self.residual_logical_len:
            return False
        if self.page_count and self.page_bytes:
            if covered + res_cov != self.page_count * self.page_bytes:
                return False
        return True


def pad_to(n: int, align: int) -> int:
    if align <= 0:
        raise ValueError("align must be positive")
    if n < 0:
        raise ValueError("n must be non-negative")
    rem = n % align
    return n if rem == 0 else n + (align - rem)


def split_page(d: int, page_bytes: int, align: int) -> Tuple[int, int, int]:
    """Return (head, body, tail) byte lengths for one page pointer."""
    if page_bytes < 0 or align <= 0:
        raise ValueError("invalid page split inputs")
    head = (align - (d % align)) % align
    if head >= page_bytes:
        return page_bytes, 0, 0
    body = ((page_bytes - head) // align) * align
    tail = page_bytes - head - body
    return head, body, tail


def split_pages(
    gpu_addrs: Sequence[int],
    page_bytes: int,
    sub_stripe: int,
    sector: int = 4096,
) -> LayoutDescriptor:
    """Build a layout for a chunk of independent page pointers."""
    align = lcm(int(sub_stripe), int(sector))
    n = len(gpu_addrs)
    slices: List[DirectSlice] = []
    residuals: List[ResidualDesc] = []
    direct_off = 0
    staging_off = 0
    for i, addr in enumerate(gpu_addrs):
        d = int(addr)
        head, body, tail = split_page(d, int(page_bytes), align)
        if head:
            residuals.append(
                ResidualDesc(page_index=i, page_off=0, staging_off=staging_off, length=head)
            )
            staging_off += head
        if body:
            gpu_body = d + head
            if (
                gpu_body % align == 0
                and body % align == 0
                and direct_off % align == 0
            ):
                slices.append(
                    DirectSlice(gpu_addr=gpu_body, object_off=direct_off, length=body)
                )
                direct_off += body
            else:
                residuals.append(
                    ResidualDesc(
                        page_index=i,
                        page_off=head,
                        staging_off=staging_off,
                        length=body,
                    )
                )
                staging_off += body
        if tail:
            residuals.append(
                ResidualDesc(
                    page_index=i,
                    page_off=head + body,
                    staging_off=staging_off,
                    length=tail,
                )
            )
            staging_off += tail
    ss = int(sub_stripe)
    return LayoutDescriptor(
        layout_version=LAYOUT_VERSION,
        page_bytes=int(page_bytes),
        page_count=n,
        sub_stripe=ss,
        sector=int(sector),
        align=align,
        direct_logical_len=direct_off,
        residual_logical_len=staging_off,
        direct_phys_len=pad_to(direct_off, ss) if direct_off else 0,
        residual_phys_len=pad_to(staging_off, ss) if staging_off else 0,
        direct_slices=slices,
        residuals=residuals,
    )
