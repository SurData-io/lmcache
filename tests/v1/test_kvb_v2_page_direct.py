# SPDX-License-Identifier: Apache-2.0
"""KVB v2 geometry, zero-residual probe, and page-view tests (no GPU)."""

from __future__ import annotations

import unittest

from lmcache.v1.gds_layout import split_pages
from lmcache.v1.kvb_v2 import (
    HEADER_LAYOUT_VERSION,
    PAYLOAD_ENCODING_CANONICAL_KV2LTD,
    PAYLOAD_ENCODING_RAW_PAGES,
    compute_profile,
    descriptors_from_split,
    header_from_layout,
    pack_header,
    physical_page_format_id,
    required_block_multiple,
    unpack_header,
)
from lmcache.v1.page_direct import (
    PageView,
    fuse_allocation_runs,
    gds_serve_block_size,
    launcher_block_size,
    probe_zero_residual,
    slot_ids_to_block_coverage,
)


class TestRequiredBlockMultiple(unittest.TestCase):
    def test_fp16_mla_align8192(self) -> None:
        self.assertEqual(required_block_multiple(8192, 1152), 64)

    def test_fp8_mla_align8192(self) -> None:
        self.assertEqual(required_block_multiple(8192, 576), 128)

    def test_align4096_fp16(self) -> None:
        # gcd(4096, 1152) = 128 → 32
        self.assertEqual(required_block_multiple(4096, 1152), 32)


class TestZeroResidualProbe(unittest.TestCase):
    def _views(self, addrs, page_bytes, registered=True, contiguous=True):
        return [
            PageView(
                gpu_addr=a,
                page_bytes=page_bytes,
                layer_index=0,
                block_id=i,
                group_id=0,
                is_contiguous=contiguous,
                is_gds_registered=registered,
                allocation_id=1,
            )
            for i, a in enumerate(addrs)
        ]

    def test_bs16_mla_not_zero_residual(self) -> None:
        page = 16 * 1152
        views = self._views([0, page], page)
        probe = probe_zero_residual(views, block_size=16, sub_stripe=8192)
        self.assertEqual(probe.required_block_multiple, 64)
        self.assertFalse(probe.zero_residual)
        self.assertIn("required_block_multiple=64", probe.warn)

    def test_bs64_aligned_zero_residual(self) -> None:
        page = 64 * 1152
        align = 8192
        self.assertEqual(page % align, 0)
        views = self._views([0, page], page)
        probe = probe_zero_residual(views, block_size=64, sub_stripe=8192)
        self.assertTrue(probe.zero_residual)
        self.assertEqual(probe.warn, "")

    def test_unregistered_disables_fastpath(self) -> None:
        page = 64 * 1152
        views = self._views([0, page], page, registered=False)
        probe = probe_zero_residual(views, block_size=64, sub_stripe=8192)
        self.assertFalse(probe.zero_residual)

    def test_noncontiguous_disables_fastpath(self) -> None:
        page = 64 * 1152
        views = self._views([0], page, contiguous=False)
        probe = probe_zero_residual(views, block_size=64, sub_stripe=8192)
        self.assertFalse(probe.zero_residual)

    def test_launcher_rounds_up_without_runtime_mutate(self) -> None:
        self.assertEqual(launcher_block_size(16, 64), 64)
        self.assertEqual(launcher_block_size(64, 64), 64)
        self.assertEqual(launcher_block_size(16, 32), 32)

    def test_gds_serve_block_size_from_sub_stripe(self) -> None:
        self.assertEqual(
            gds_serve_block_size(
                sub_stripe=4096, bytes_per_token_per_layer=1152, current_block_size=16
            ),
            32,
        )
        self.assertEqual(
            gds_serve_block_size(
                sub_stripe=8192, bytes_per_token_per_layer=1152, current_block_size=16
            ),
            64,
        )
        self.assertEqual(
            gds_serve_block_size(
                sub_stripe=8192, bytes_per_token_per_layer=1152, current_block_size=64
            ),
            64,
        )


class TestSlotCoverage(unittest.TestCase):
    def test_complete_pages(self) -> None:
        bs = 16
        slots = list(range(32, 64))  # two full pages: blocks 2,3
        ok, ids, reason = slot_ids_to_block_coverage(slots, bs)
        self.assertTrue(ok)
        self.assertEqual(ids, [2, 3])
        self.assertEqual(reason, "")

    def test_partial_page(self) -> None:
        ok, ids, reason = slot_ids_to_block_coverage(list(range(8)), 16)
        self.assertFalse(ok)
        self.assertIn("chunk_len", reason)

    def test_duplicate_slot(self) -> None:
        slots = list(range(16))
        slots[3] = 0
        ok, _, reason = slot_ids_to_block_coverage(slots, 16)
        self.assertFalse(ok)
        self.assertIn("duplicate", reason)


class TestFuseRuns(unittest.TestCase):
    def test_merge_adjacent_same_alloc(self) -> None:
        runs, err = fuse_allocation_runs(
            [(0, 4096, 1), (4096, 4096, 1)],
            sector=4096,
            registration_limit=8,
        )
        self.assertEqual(err, "")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].nbytes, 8192)

    def test_no_merge_across_alloc(self) -> None:
        runs, err = fuse_allocation_runs(
            [(0, 4096, 1), (4096, 4096, 2)],
            sector=4096,
            registration_limit=8,
        )
        self.assertEqual(err, "")
        self.assertEqual(len(runs), 2)

    def test_gap_same_alloc_stays_separate(self) -> None:
        runs, err = fuse_allocation_runs(
            [(0, 4096, 1), (8192, 4096, 1)],
            sector=4096,
            registration_limit=8,
        )
        self.assertEqual(err, "")
        self.assertEqual(len(runs), 2)

    def test_limit_includes_staging(self) -> None:
        _, err = fuse_allocation_runs(
            [(0, 4096, 1), (8192, 4096, 2)],
            sector=4096,
            registration_limit=2,
        )
        self.assertIn("exceed", err)


class TestHeaderRoundTrip(unittest.TestCase):
    def test_zero_residual_header(self) -> None:
        page = 64 * 1152
        addrs = [0, page]
        layout = split_pages(addrs, page, 8192, sector=4096)
        self.assertEqual(layout.residual_logical_len, 0)
        self.assertEqual(len(layout.direct_slices), 2)
        self.assertTrue(layout.geometry_ok())
        fmt = physical_page_format_id(
            format_name="NL_X_NB_BS_HS",
            kv_layout="NHD",
            dtype_itemsize=2,
            vllm_block_size=64,
            page_bytes=page,
            layer_count=1,
            kv_factor=1,
            enum_order="layer_then_block",
        )
        profile = compute_profile(
            max_pages=4,
            page_bytes=page,
            sub_stripe=8192,
            sector=4096,
            align=8192,
            physical_page_format_id=fmt,
            canonical_payload_bytes=page * 4,
        )
        self.assertEqual(profile.header_region_bytes % 8192, 0)
        hdr = header_from_layout(layout, profile, PAYLOAD_ENCODING_RAW_PAGES)
        blob = pack_header(hdr)
        self.assertEqual(len(blob), profile.header_region_bytes)
        back = unpack_header(blob, expected_format_id=fmt)
        self.assertEqual(back.payload_encoding, PAYLOAD_ENCODING_RAW_PAGES)
        self.assertEqual(back.page_count, 2)
        self.assertEqual(sum(d.body_len for d in back.descriptors), 2 * page)
        self.assertEqual(sum(d.residual_len for d in back.descriptors), 0)

    def test_residual_header_bs16(self) -> None:
        page = 16 * 1152
        addrs = [0, page]
        layout = split_pages(addrs, page, 4096, sector=4096)
        self.assertGreater(layout.residual_logical_len, 0)
        self.assertTrue(layout.geometry_ok())
        fmt = physical_page_format_id(
            format_name="NL_X_NB_BS_HS",
            kv_layout="NHD",
            dtype_itemsize=2,
            vllm_block_size=16,
            page_bytes=page,
            layer_count=1,
            kv_factor=1,
            enum_order="layer_then_block",
        )
        profile = compute_profile(
            max_pages=8,
            page_bytes=page,
            sub_stripe=4096,
            sector=4096,
            align=4096,
            physical_page_format_id=fmt,
        )
        hdr = header_from_layout(layout, profile, PAYLOAD_ENCODING_RAW_PAGES)
        blob = pack_header(hdr)
        back = unpack_header(blob)
        self.assertEqual(back.layout_version, HEADER_LAYOUT_VERSION)
        covered = sum(d.body_len + d.residual_len for d in back.descriptors)
        self.assertEqual(covered, 2 * page)

    def test_unknown_encoding_rejected(self) -> None:
        page = 4096
        layout = split_pages([0], page, 4096, sector=4096)
        fmt = 1
        profile = compute_profile(
            max_pages=1,
            page_bytes=page,
            sub_stripe=4096,
            sector=4096,
            align=4096,
            physical_page_format_id=fmt,
        )
        hdr = header_from_layout(layout, profile, PAYLOAD_ENCODING_CANONICAL_KV2LTD)
        blob = bytearray(pack_header(hdr))
        blob[12:16] = (99).to_bytes(4, "little")
        with self.assertRaises(ValueError):
            unpack_header(bytes(blob))

    def test_format_mismatch_rejected(self) -> None:
        page = 4096
        layout = split_pages([0], page, 4096, sector=4096)
        profile = compute_profile(
            max_pages=1,
            page_bytes=page,
            sub_stripe=4096,
            sector=4096,
            align=4096,
            physical_page_format_id=1,
        )
        hdr = header_from_layout(layout, profile, PAYLOAD_ENCODING_RAW_PAGES)
        blob = pack_header(hdr)
        with self.assertRaises(ValueError):
            unpack_header(blob, expected_format_id=2)

    def test_header_blob_from_buffer_reads_descriptors(self) -> None:
        from lmcache.v1.kvb_v2 import (
            FIXED_HEADER_BYTES,
            header_blob_from_buffer,
        )

        page = 64 * 1152
        layout = split_pages([0, page], page, 8192, sector=4096)
        fmt = physical_page_format_id(
            format_name="NL_X_NB_BS_HS",
            kv_layout="NHD",
            dtype_itemsize=2,
            vllm_block_size=64,
            page_bytes=page,
            layer_count=1,
            kv_factor=1,
            enum_order="layer_then_block",
        )
        profile = compute_profile(
            max_pages=4,
            page_bytes=page,
            sub_stripe=8192,
            sector=4096,
            align=8192,
            physical_page_format_id=fmt,
            canonical_payload_bytes=page * 4,
        )
        hdr = header_from_layout(layout, profile, PAYLOAD_ENCODING_RAW_PAGES)
        blob = pack_header(hdr)
        with self.assertRaises(ValueError):
            unpack_header(bytes(blob[:FIXED_HEADER_BYTES]))
        full = header_blob_from_buffer(blob)
        self.assertEqual(len(full), profile.header_region_bytes)
        back = unpack_header(full, expected_format_id=fmt)
        self.assertEqual(back.payload_encoding, PAYLOAD_ENCODING_RAW_PAGES)
        self.assertEqual(back.page_count, 2)


class TestResidualCopyAndPhase(unittest.TestCase):
    def test_cpu_residual_copy_roundtrip(self) -> None:
        import ctypes

        from lmcache.v1.platform.torch_ops import residual_copy_bytes

        src_ba = bytearray(b"abcdefghij")
        dst_ba = bytearray(10)
        src = (ctypes.c_uint8 * 10).from_buffer(src_ba)
        dst = (ctypes.c_uint8 * 10).from_buffer(dst_ba)
        residual_copy_bytes(ctypes.addressof(dst), ctypes.addressof(src), 10, 0)
        self.assertEqual(bytes(dst_ba), b"abcdefghij")

    def test_split_phase_mismatch_preserves_bytes(self) -> None:
        from lmcache.v1.gds_layout import split_page

        page = 16 * 1152
        src_addrs = [2048, 2048 + page]
        dst_addrs = [0, page]
        src = split_pages(src_addrs, page, 4096, sector=4096)
        dst = split_pages(dst_addrs, page, 4096, sector=4096)
        self.assertTrue(src.geometry_ok())
        self.assertTrue(dst.geometry_ok())
        self.assertNotEqual(src_addrs[0] % src.align, dst_addrs[0] % dst.align)
        from lmcache.v1.kvb_v2 import descriptors_from_split

        descs = descriptors_from_split(src, 8192)
        self.assertEqual(len(descs), 2)
        self.assertGreater(descs[0].page_payload_off, 0)
        for desc in descs:
            self.assertEqual(desc.body_len + desc.residual_len, page)
            head = desc.page_payload_off
            self.assertEqual(desc.residual_len, head + (page - head - desc.body_len))
        for addr in src_addrs + dst_addrs:
            head, body, tail = split_page(addr, page, src.align)
            self.assertEqual(head + body + tail, page)


if __name__ == "__main__":
    unittest.main()
