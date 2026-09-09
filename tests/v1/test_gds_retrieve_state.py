# SPDX-License-Identifier: Apache-2.0
"""GDS retrieve state machine: wait for DMA, never fail-open a confirmed hit."""

from __future__ import annotations

import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import torch
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.cache_engine import (
        GDS_COMPLETE,
        GDS_HBM_ASSEMBLED,
        GdsPipelineMetrics,
        LMCacheEngine,
        PendingLoad,
    )
except ImportError:
    torch = None  # type: ignore
    CacheEngineKey = None  # type: ignore


class TestOverlayApisRemoved(unittest.TestCase):
    def test_adapter_source_has_no_overlay_entrypoints(self):
        root = Path(__file__).resolve().parents[2]
        src = (root / "lmcache/integration/vllm/vllm_v1_adapter.py").read_text()
        for name in (
            "def lookup_after_hbm_alloc",
            "def start_dram_preload",
            "def dram_preload_enabled",
            "def bind_gpu_block_pool",
            "def prefetch_admission_ok",
            "def take_xfer_records",
            "def cancel_async_load",
        ):
            self.assertNotIn(name, src, name)
        self.assertIn("def handle_preemptions", src)
        self.assertIn("def get_num_new_matched_tokens", src)

    def test_wrapper_source_forwards_standard_tuple(self):
        root = Path(__file__).resolve().parents[2]
        src = (root / "lmcache/integration/vllm/lmcache_connector_v1.py").read_text()
        self.assertIn("def handle_preemptions", src)
        self.assertIn(
            "self._lmcache_engine.update_state_after_alloc(\n"
            "            request, blocks, num_external_tokens\n"
            "        )",
            src,
        )
        self.assertNotIn(
            "self._lmcache_engine.update_state_after_alloc(request, num_external_tokens)",
            src,
        )


class TestGdsPartialPrefillStore(unittest.TestCase):
    def test_gds_defers_until_last_prefill(self) -> None:
        try:
            from lmcache.v1.gds_store_policy import gds_defer_partial_prefill_store
        except ImportError as exc:
            self.skipTest(f"gds_store_policy unavailable: {exc}")
        self.assertTrue(gds_defer_partial_prefill_store("gds", False))
        self.assertFalse(gds_defer_partial_prefill_store("gds", True))
        self.assertFalse(gds_defer_partial_prefill_store("dram", False))
        self.assertFalse(gds_defer_partial_prefill_store("DRAM", True))

    def test_gds_io_pool_workers_covers_last_prefill_herd(self) -> None:
        try:
            from lmcache.v1.gds_store_policy import gds_io_pool_workers
        except ImportError as exc:
            self.skipTest(str(exc))

        self.assertEqual(gds_io_pool_workers(4, 4), 32)
        self.assertEqual(gds_io_pool_workers(64, 4), 64)
        self.assertEqual(gds_io_pool_workers(4, 48), 48)


class TestGdsStoreOverloadPolicy(unittest.TestCase):
    def test_policy_helpers(self) -> None:
        try:
            from lmcache.v1.gds_store_policy import (
                GDS_STORE_OVERLOAD_IMMEDIATE_DROP,
                GDS_STORE_OVERLOAD_IMMEDIATE_D2H,
                gds_direct_store_max_inflight,
                gds_store_overload_policy,
            )
        except ImportError as exc:
            self.skipTest(str(exc))

        self.assertEqual(
            gds_store_overload_policy(None), GDS_STORE_OVERLOAD_IMMEDIATE_DROP
        )
        self.assertEqual(
            gds_store_overload_policy("immediate_d2h_fallback"),
            GDS_STORE_OVERLOAD_IMMEDIATE_D2H,
        )
        self.assertEqual(gds_direct_store_max_inflight(0), 0)
        self.assertEqual(gds_direct_store_max_inflight(8), 8)
        with self.assertRaises(ValueError):
            gds_store_overload_policy("drop_silent")
        with self.assertRaises(ValueError):
            gds_direct_store_max_inflight(-1)

    def test_skip_exists_requires_matching_stamp(self) -> None:
        from lmcache.v1.gds_store_policy import gds_skip_exists_for_empty_ns

        self.assertFalse(gds_skip_exists_for_empty_ns(False, "a", "a"))
        self.assertFalse(gds_skip_exists_for_empty_ns(True, "", "run"))
        self.assertFalse(gds_skip_exists_for_empty_ns(True, "a", "b"))
        self.assertFalse(gds_skip_exists_for_empty_ns("false", "stamp", "stamp"))
        self.assertTrue(gds_skip_exists_for_empty_ns(True, "stamp", "stamp"))
        self.assertTrue(gds_skip_exists_for_empty_ns("1", "stamp", "stamp"))


class _Lease:
    def __init__(self, idx: int):
        self.idx = idx
        self.memory_obj = SimpleNamespace(
            get_size=lambda: 8,
            raw_data=None,
        )


@unittest.skipUnless(torch is not None, "torch required")
class TestGdsRetrieveState(unittest.TestCase):
    def test_metrics_p99(self):
        m = GdsPipelineMetrics()
        for i in range(100):
            m.note_submit_dma(i)
        snap = m.snapshot()
        self.assertEqual(snap["submit_to_dma_n"], 100)
        self.assertGreaterEqual(snap["submit_to_dma_p99_us"], 90)

    def test_try_issue_h2d_marks_dma_done(self):
        engine = mock.Mock(spec=LMCacheEngine)
        engine.config = SimpleNamespace(kv_io_mode="gds")
        engine._gds_metrics = GdsPipelineMetrics()
        pending = PendingLoad(req_id="r1", chunks=[])
        fut = mock.Mock()
        fut.done.return_value = True
        lease = _Lease(0)
        mo = SimpleNamespace()
        fut.result.return_value = (
            [(mock.Mock(), mo, 0, 8)],
            torch.ones(8, dtype=torch.bool),
            [lease],
        )
        pending.gather_future = fut
        pending.kwargs_keepalive = {}

        class _Conn:
            def batched_to_gpu_async(self, *args, **kwargs):
                return mock.Mock()

        engine.gpu_connector = _Conn()
        result = LMCacheEngine.try_issue_h2d(engine, pending)
        self.assertEqual(pending.gds_state, GDS_HBM_ASSEMBLED)
        self.assertEqual(result, "issued")

    def test_try_issue_h2d_gds_page_direct_empty_chunks_completes(self):
        engine = mock.Mock(spec=LMCacheEngine)
        engine.config = SimpleNamespace(kv_io_mode="gds")
        engine._gds_metrics = GdsPipelineMetrics()
        pending = PendingLoad(req_id="r1", chunks=[])
        fut = mock.Mock()
        fut.done.return_value = True
        fut.result.return_value = ([], torch.ones(8, dtype=torch.bool), [])
        pending.gather_future = fut
        pending.kwargs_keepalive = {"_gather_tokens": torch.arange(8)}
        result = LMCacheEngine.try_issue_h2d(engine, pending)
        self.assertEqual(result, "issued")
        self.assertTrue(pending.h2d_issued)
        self.assertTrue(bool(pending.ret_mask.any()))
        self.assertEqual(pending.gather_retries, 0)

    def test_try_issue_h2d_gds_empty_mask_fails_closed(self):
        engine = mock.Mock(spec=LMCacheEngine)
        engine.config = SimpleNamespace(kv_io_mode="gds")
        pending = PendingLoad(req_id="r1", chunks=[])
        fut = mock.Mock()
        fut.done.return_value = True
        fut.result.return_value = ([], torch.zeros(8, dtype=torch.bool), [])
        pending.gather_future = fut
        pending.kwargs_keepalive = {"_gather_tokens": torch.arange(8)}
        result = LMCacheEngine.try_issue_h2d(engine, pending)
        self.assertEqual(result, "error")
        self.assertIsNotNone(pending.error)
        self.assertEqual(pending.gather_retries, 0)

    def test_submit_retrieve_gather_gds_does_not_block(self):
        engine = mock.Mock(spec=LMCacheEngine)
        engine.config = SimpleNamespace(kv_io_mode="gds")
        engine.storage_manager = SimpleNamespace(loop=None)
        engine._get_req_id = mock.Mock(return_value="r1")
        engine._release_gds_leases = mock.Mock()

        def _gather(tokens, mask, ret_mask, **kwargs):
            pl = kwargs["_pending_load"]
            pl.pd_prefix = 2
            pl.pd_hit = object()
            pl.pd_adapter = SimpleNamespace()
            pl.pd_spans = [(0, 2), (2, 4)]
            return ([], ret_mask, [])

        engine._gather_pinned_chunks = _gather
        engine._submit_page_direct_reads = mock.Mock()
        pending = LMCacheEngine.submit_retrieve_gather(
            engine,
            torch.arange(4),
            None,
            slot_mapping=torch.arange(4),
        )
        self.assertIsNone(pending.gather_future)
        engine._submit_page_direct_reads.assert_called_once()
        self.assertIsNone(pending.error)

    def test_complete_load_sets_complete(self):
        engine = mock.Mock()
        engine.save_only_first_rank = False
        engine._gds_metrics = GdsPipelineMetrics()
        engine._release_gds_leases = mock.Mock()
        engine._release_retrieve_chunks = mock.Mock()
        engine.metadata = SimpleNamespace(is_first_rank=lambda: False)
        engine.stats_monitor = mock.Mock()
        pending = PendingLoad(req_id="r1", chunks=[])
        pending.gds_leases = [_Lease(0)]
        pending.ret_mask = torch.ones(4, dtype=torch.bool)
        LMCacheEngine.complete_load(engine, pending)
        self.assertEqual(pending.gds_state, GDS_COMPLETE)

    def test_complete_load_releases_mixed_dram_chunks(self):
        engine = mock.Mock()
        engine.save_only_first_rank = False
        engine._gds_metrics = GdsPipelineMetrics()
        engine._release_gds_leases = mock.Mock()
        engine._release_retrieve_chunks = mock.Mock()
        engine._gds_lease_memory_ids = LMCacheEngine._gds_lease_memory_ids.__get__(
            engine
        )
        engine.metadata = SimpleNamespace(is_first_rank=lambda: False)
        engine.stats_monitor = mock.Mock()
        gds_mo = SimpleNamespace()
        dram_mo = SimpleNamespace()
        lease = _Lease(0)
        lease.memory_obj = gds_mo
        pending = PendingLoad(
            req_id="r1",
            chunks=[
                ("k0", gds_mo, 0, 2),
                ("k1", dram_mo, 2, 4),
            ],
        )
        pending.gds_leases = [lease]
        pending.ret_mask = torch.ones(4, dtype=torch.bool)
        LMCacheEngine.complete_load(engine, pending)
        engine._release_gds_leases.assert_called_once()
        engine._release_retrieve_chunks.assert_called_once()
        released = engine._release_retrieve_chunks.call_args[0][0]
        self.assertEqual(len(released), 1)
        self.assertIs(released[0][1], dram_mo)

    def test_gds_process_does_not_truncate_on_pool_wait(self):
        engine = mock.Mock(spec=LMCacheEngine)
        engine.config = SimpleNamespace(kv_io_mode="gds", kv_gds_watchdog_timeout_ms=0)
        engine._gds_metrics = GdsPipelineMetrics()
        engine._pd_profile = None
        engine._gds_remote_adapter = mock.Mock(
            return_value=SimpleNamespace(
                prefix_contains_tokens=lambda tokens, n_keys=-1: 2,
                kvb_get_prefix=mock.Mock(side_effect=AssertionError("host GET")),
            )
        )
        engine.storage_manager = SimpleNamespace(
            batched_get=mock.Mock(side_effect=AssertionError("host GET"))
        )
        engine.token_database = SimpleNamespace(
            process_tokens=lambda **kwargs: [
                (0, 8, CacheEngineKey("m", 1, 0, 0, torch.float16)),
                (8, 16, CacheEngineKey("m", 1, 0, 1, torch.float16)),
            ]
        )
        ret_mask = torch.zeros(16, dtype=torch.bool)
        chunks, _sz, leases = LMCacheEngine._gds_process_tokens_internal(
            engine, torch.arange(16), None, ret_mask
        )
        self.assertEqual(len(chunks), 0)
        self.assertEqual(len(leases), 0)
        self.assertFalse(bool(ret_mask.any()))

    def test_gds_one_gather_for_multi_chunk_prefix(self):
        engine = mock.Mock(spec=LMCacheEngine)
        engine.config = SimpleNamespace(kv_io_mode="gds", kv_gds_watchdog_timeout_ms=0)
        engine._gds_metrics = GdsPipelineMetrics()
        engine._pd_profile = None
        engine._gds_remote_adapter = mock.Mock(
            return_value=SimpleNamespace(
                prefix_contains_tokens=lambda tokens, n_keys=-1: 8,
                kvb_get_prefix=mock.Mock(side_effect=AssertionError("host GET")),
            )
        )
        engine.storage_manager = SimpleNamespace(
            batched_get=mock.Mock(side_effect=AssertionError("host GET"))
        )
        engine.token_database = SimpleNamespace(
            process_tokens=lambda **kwargs: [
                (i * 8, (i + 1) * 8, CacheEngineKey("m", 1, 0, i, torch.float16))
                for i in range(8)
            ]
        )
        ret_mask = torch.zeros(64, dtype=torch.bool)
        chunks, _sz, leases = LMCacheEngine._gds_process_tokens_internal(
            engine, torch.arange(64), None, ret_mask
        )
        self.assertEqual(len(chunks), 0)
        self.assertEqual(len(leases), 0)
        self.assertFalse(bool(ret_mask.any()))


@unittest.skipUnless(torch is not None, "torch required")
class TestGdsPageDirectRetrieve(unittest.TestCase):
    def test_gds_lookup_without_slot_mapping_skips_host_get(self):
        engine = mock.Mock(spec=LMCacheEngine)
        engine.config = SimpleNamespace(kv_io_mode="gds", kv_gds_watchdog_timeout_ms=0)
        engine._gds_metrics = GdsPipelineMetrics()
        engine._pd_profile = None
        adapter = SimpleNamespace(
            prefix_contains_tokens=lambda tokens, n_keys=-1: 2,
            kvb_get_prefix=mock.Mock(side_effect=AssertionError("host GET")),
            kvb_read_to_gpu=mock.Mock(side_effect=AssertionError("gpu GET")),
        )
        engine._gds_remote_adapter = mock.Mock(return_value=adapter)
        engine.storage_manager = SimpleNamespace(
            batched_get=mock.Mock(side_effect=AssertionError("host GET"))
        )
        engine.token_database = SimpleNamespace(
            process_tokens=lambda **kwargs: [
                (0, 8, CacheEngineKey("m", 1, 0, 0, torch.float16)),
                (8, 16, CacheEngineKey("m", 1, 0, 1, torch.float16)),
            ]
        )
        ret_mask = torch.zeros(16, dtype=torch.bool)
        chunks, _sz, leases = LMCacheEngine._gds_process_tokens_internal(
            engine, torch.arange(16), None, ret_mask
        )
        self.assertEqual(len(chunks), 0)
        self.assertEqual(len(leases), 0)
        self.assertFalse(bool(ret_mask.any()))
        adapter.kvb_get_prefix.assert_not_called()

    def test_gds_page_direct_with_mapping_records_lookup(self):
        from lmcache.v1.cache_engine import PendingLoad

        engine = mock.Mock(spec=LMCacheEngine)
        engine.config = SimpleNamespace(kv_io_mode="gds", kv_gds_watchdog_timeout_ms=0)
        engine._gds_metrics = GdsPipelineMetrics()
        engine._pd_profile = object()
        hit = SimpleNamespace(found=1)
        adapter = SimpleNamespace(
            lookup_prefix_hit=lambda tokens, n_keys=-1: (2, hit),
            kvb_get_prefix=mock.Mock(side_effect=AssertionError("host GET")),
            kvb_read_to_gpu=mock.Mock(),
            kvb_read_to_gpu_submit=mock.Mock(),
        )
        engine._gds_remote_adapter = mock.Mock(return_value=adapter)
        engine.gpu_connector = SimpleNamespace(page_direct_enabled=lambda: True)
        engine.storage_manager = SimpleNamespace(
            batched_get=mock.Mock(side_effect=AssertionError("host GET"))
        )
        engine.token_database = SimpleNamespace(
            process_tokens=lambda **kwargs: [
                (0, 8, CacheEngineKey("m", 1, 0, 0, torch.float16)),
                (8, 16, CacheEngineKey("m", 1, 0, 1, torch.float16)),
            ]
        )
        ret_mask = torch.zeros(16, dtype=torch.bool)
        pending = PendingLoad(req_id="r1", chunks=[])
        slot_mapping = torch.arange(16)
        chunks, prefix, leases = LMCacheEngine._gds_process_tokens_internal(
            engine,
            torch.arange(16),
            None,
            ret_mask,
            slot_mapping=slot_mapping,
            _pending_load=pending,
        )
        self.assertEqual(len(chunks), 0)
        self.assertEqual(len(leases), 0)
        self.assertEqual(prefix, 2)
        self.assertIs(pending.pd_hit, hit)
        adapter.kvb_get_prefix.assert_not_called()
        engine.storage_manager.batched_get.assert_not_called()

    def test_gds_page_direct_failure_does_not_fallback_host(self):
        engine = mock.Mock(spec=LMCacheEngine)
        engine.config = SimpleNamespace(kv_io_mode="gds", kv_gds_watchdog_timeout_ms=0)
        engine._pd_profile = object()
        hit = SimpleNamespace(found=1)
        adapter = SimpleNamespace(
            lookup_prefix_hit=lambda tokens, n_keys=-1: (2, hit),
            kvb_get_prefix=mock.Mock(side_effect=AssertionError("host GET")),
            kvb_read_to_gpu=mock.Mock(),
        )
        engine._gds_remote_adapter = mock.Mock(return_value=adapter)
        engine.gpu_connector = SimpleNamespace(page_direct_enabled=lambda: True)
        engine._retrieve_page_direct = mock.Mock(
            side_effect=RuntimeError("gds read failed")
        )
        engine.storage_manager = SimpleNamespace(
            batched_get=mock.Mock(side_effect=AssertionError("host GET"))
        )
        engine.token_database = SimpleNamespace(
            process_tokens=lambda **kwargs: [
                (0, 8, CacheEngineKey("m", 1, 0, 0, torch.float16)),
                (8, 16, CacheEngineKey("m", 1, 0, 1, torch.float16)),
            ]
        )
        ret_mask = torch.zeros(16, dtype=torch.bool)
        pending = PendingLoad(req_id="r1", chunks=[])
        chunks, prefix, leases = LMCacheEngine._gds_process_tokens_internal(
            engine,
            torch.arange(16),
            None,
            ret_mask,
            slot_mapping=torch.arange(16),
            _pending_load=pending,
        )
        self.assertEqual(prefix, 2)
        self.assertEqual(len(chunks), 0)
        adapter.kvb_get_prefix.assert_not_called()
        engine.storage_manager.batched_get.assert_not_called()

    def test_gpu_iov_truncates_hashes_to_record_count(self):
        import concurrent.futures
        import threading

        try:
            from surdata_lmcache.inproc_adapter import SurDataRemoteConnector
        except ImportError:
            self.skipTest("SurData SDK adapter is not part of this public tree")

        captured: dict[str, object] = {}

        class NativeFut:
            def done(self) -> bool:
                return True

            def rc(self) -> int:
                return 0

            def error_message(self) -> str:
                return ""

            def result(self) -> int:
                return 4

        def store_fn(hashes, profile, records, bt, eb, te):
            captured["hashes"] = list(hashes)
            captured["nrec"] = len(records)
            del profile, bt, eb, te
            return NativeFut()

        class Adapter:
            _closing = False
            _block_tokens = 16
            _elem_bytes = 2
            _tail_elems = 1

            def __init__(self, pool) -> None:
                self._stats_lock = threading.Lock()
                self._gds_iov_submit_started = 0
                self._gds_iov_native_submitted = 0
                self._gds_iov_submit_error = 0
                self._gds_iov_native_done = 0
                produced = [bytes([i]) * 16 for i in range(6)]
                self.client = SimpleNamespace(
                    surdata_native=SimpleNamespace(
                        prefix_hashes_from_tokens=lambda ids, bt: produced,
                        kvb_store_from_gpu_iov=store_fn,
                    )
                )
                self._gds_io_pool = pool

            def _gds_io_executor(self):
                return self._gds_io_pool

            def _gds_iov_count(self, name: str) -> None:
                with self._stats_lock:
                    setattr(self, name, int(getattr(self, name, 0) or 0) + 1)

        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="t-gds-nchunk"
        )
        adapter = Adapter(pool)
        records = [{"kv_block_index": i} for i in range(4)]
        fut = SurDataRemoteConnector.kvb_store_from_gpu_iov(
            adapter, list(range(96)), records, {"align_bytes": 8192}
        )
        self.assertEqual(fut.result(timeout=2), 4)
        pool.shutdown(wait=True)
        self.assertEqual(len(captured["hashes"]), 4)
        self.assertEqual(captured["nrec"], 4)

    def test_gpu_iov_hashes_on_worker_not_caller(self):
        import concurrent.futures
        import threading

        try:
            from surdata_lmcache.inproc_adapter import SurDataRemoteConnector
        except ImportError:
            self.skipTest("SurData SDK adapter is not part of this public tree")

        caller = threading.current_thread().name
        hashed_on: list[str] = []

        class NativeFut:
            def done(self) -> bool:
                return True

            def rc(self) -> int:
                return 0

            def error_message(self) -> str:
                return ""

            def result(self) -> int:
                return 1

        def hash_fn(ids, bt):
            hashed_on.append(threading.current_thread().name)
            del ids, bt
            return [b"h0"]

        def store_fn(*args, **kwargs):
            del args, kwargs
            return NativeFut()

        class Adapter:
            _closing = False
            _block_tokens = 16
            _elem_bytes = 2
            _tail_elems = 1

            def __init__(self, pool) -> None:
                self._stats_lock = threading.Lock()
                self._gds_iov_submit_started = 0
                self._gds_iov_native_submitted = 0
                self._gds_iov_submit_error = 0
                self._gds_iov_native_done = 0
                self.client = SimpleNamespace(
                    surdata_native=SimpleNamespace(
                        prefix_hashes_from_tokens=hash_fn,
                        kvb_store_from_gpu_iov=store_fn,
                    )
                )
                self._gds_io_pool = pool

            def _gds_io_executor(self):
                return self._gds_io_pool

            def _gds_iov_count(self, name: str) -> None:
                with self._stats_lock:
                    setattr(self, name, int(getattr(self, name, 0) or 0) + 1)

        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="t-gds-hash"
        )
        adapter = Adapter(pool)
        fut = SurDataRemoteConnector.kvb_store_from_gpu_iov(
            adapter, [1, 2, 3], [{"kv_block_index": 0}], {"align_bytes": 64}
        )
        self.assertEqual(fut.result(timeout=2), 1)
        pool.shutdown(wait=True)
        self.assertEqual(len(hashed_on), 1)
        self.assertNotEqual(hashed_on[0], caller)
        self.assertTrue(hashed_on[0].startswith("t-gds-hash"))


@unittest.skipUnless(torch is not None, "torch required")
class TestGdsStoreSplit(unittest.TestCase):
    def _engine(self, store_slots=2, alloc_fail_after=None, admit_bytes=1 << 30):
        from contextlib import nullcontext

        from lmcache.v1.cache_engine import PendingStore

        engine = mock.Mock(spec=LMCacheEngine)
        engine.config = SimpleNamespace(
            kv_io_mode="gds",
            kv_gds_dram_fallback_bytes=admit_bytes,
            kv_gds_staging_wait_timeout_ms=0,
            kv_gds_staging_max_waiters=32,
            kv_gds_direct_store_max_inflight=0,
            kv_gds_store_overload_policy="immediate_d2h_fallback",
        )
        engine._gds_metrics = GdsPipelineMetrics()
        engine._gds_d2h_lock = threading.Condition()
        engine._gds_d2h_bytes = 0
        engine._gds_d2h_inflight = 0
        engine._gds_direct_lock = threading.Lock()
        engine._gds_direct_inflight = 0
        engine._pd_profile = None
        engine._pd_header_pin_lock = threading.Lock()
        engine._pd_header_pins = []
        engine.fmt = None
        engine.metadata = SimpleNamespace(
            get_shapes=lambda ntok: [torch.Size([1, 1, int(ntok), 8])],
            get_dtypes=lambda: [torch.float16],
        )
        engine._nbytes_for_tokens = lambda ntok: int(ntok) * 16
        engine._admit_store_d2h = LMCacheEngine._admit_store_d2h.__get__(engine)
        engine._release_store_d2h_admit = LMCacheEngine._release_store_d2h_admit.__get__(
            engine
        )
        engine._release_gds_leases = LMCacheEngine._release_gds_leases.__get__(engine)
        engine._harvest_gds_store = LMCacheEngine._harvest_gds_store.__get__(engine)
        engine._wait_host_allocate = LMCacheEngine._wait_host_allocate.__get__(engine)
        engine._try_host_allocate = LMCacheEngine._try_host_allocate.__get__(engine)
        engine._try_acquire_direct_permit = LMCacheEngine._try_acquire_direct_permit.__get__(
            engine
        )
        engine._release_direct_permit = LMCacheEngine._release_direct_permit.__get__(
            engine
        )
        engine._gds_wait_timeout_s = LMCacheEngine._gds_wait_timeout_s.__get__(engine)
        engine._gds_max_waiters = LMCacheEngine._gds_max_waiters.__get__(engine)
        engine._gds_immediate_drop = LMCacheEngine._gds_immediate_drop.__get__(engine)
        engine._drop_gds_store = LMCacheEngine._drop_gds_store.__get__(engine)
        engine._fill_pd_header_host = LMCacheEngine._fill_pd_header_host.__get__(engine)
        engine._acquire_pd_header_host = LMCacheEngine._acquire_pd_header_host.__get__(
            engine
        )
        engine._release_pd_header_hosts = LMCacheEngine._release_pd_header_hosts.__get__(
            engine
        )
        engine._gds_store_stream = LMCacheEngine._gds_store_stream.__get__(engine)
        engine._enqueue_pd_header_copies = LMCacheEngine._enqueue_pd_header_copies.__get__(
            engine
        )
        engine._note_gds_store_split = LMCacheEngine._note_gds_store_split.__get__(
            engine
        )
        engine._maybe_prewarm_gds_pool = lambda **kw: None
        engine._gds_remote_adapter = mock.Mock()
        puts = []

        class _Fut:
            def __init__(self):
                self._exc = None

            def done(self):
                return True

            def exception(self, timeout=0):
                return self._exc

        def submit(keys, objs, cuda_event=None):
            puts.append({"keys": list(keys), "objs": list(objs), "event": cuda_event})
            return _Fut()

        engine._gds_remote_adapter.return_value = SimpleNamespace(
            submit_gds_object_put=submit
        )
        free = list(range(store_slots))

        def acquire(*, zero_padding, wait=False, timeout_s=None, purpose=""):
            self.assertFalse(wait)
            if not free:
                return None
            idx = free.pop(0)
            mo = SimpleNamespace(get_size=lambda: 8, tensor=None)
            return SimpleNamespace(idx=idx, memory_obj=mo, purpose=purpose)

        released = []

        def release(lease):
            released.append(lease.idx)
            free.append(lease.idx)

        gathers = []

        class _Ev:
            def __init__(self, name):
                self.name = name

            def query(self):
                return True

        def gather(objs, starts, ends, **kwargs):
            ev = _Ev("gds" if objs and getattr(objs[0], "tensor", None) is None else "host")
            # cuda objs from leases have tensor=None; host objs we mark tensor=host
            kind = "host" if any(getattr(o, "kind", "") == "host" for o in objs) else "gds"
            ev = _Ev(kind)
            gathers.append({"kind": kind, "n": len(objs), "event": ev, "starts": list(starts)})
            return ev

        connector = SimpleNamespace(
            _gds_slot_count=store_slots,
            _acquire_gds_staging=acquire,
            release_gds_staging=release,
            batched_from_gpu_async=gather,
            page_direct_enabled=lambda: False,
        )
        engine.gpu_connector = connector
        allocs = {"n": 0}

        def allocate(*args, **kwargs):
            if alloc_fail_after is not None and allocs["n"] >= alloc_fail_after:
                return None
            allocs["n"] += 1
            return SimpleNamespace(
                get_size=lambda: 8,
                kind="host",
                pin=lambda: None,
                unpin=lambda: None,
                ref_count_down=lambda: None,
            )

        engine.storage_manager = SimpleNamespace(allocate=allocate, batched_put=mock.Mock())
        engine.store_stats = SimpleNamespace(profile_from_gpu=lambda: nullcontext())
        engine.puts = puts
        engine.gathers = gathers
        engine.released = released
        engine.PendingStore = PendingStore
        return engine

    def _pending(self, engine, n=9):
        from lmcache.v1.cache_engine import PendingStore

        keys = [CacheEngineKey("m", 1, 0, i, torch.float16) for i in range(n)]
        starts = [i * 8 for i in range(n)]
        ends = [(i + 1) * 8 for i in range(n)]
        stats = engine.store_stats
        return PendingStore(
            req_id="r1",
            keys=keys,
            memory_objs=[],
            store_stats=stats,
            kwargs_keepalive={
                "starts": starts,
                "ends": ends,
                "_issue_store_kwargs": {},
            },
        )

    def test_partial_slots_split_not_all_or_nothing(self):
        engine = self._engine(store_slots=2)
        pending = self._pending(engine, n=9)
        out = LMCacheEngine._issue_store_gds(engine, pending)
        self.assertIs(out, pending)
        self.assertEqual(len(pending.gds_keys), 0)
        self.assertEqual(len(pending.keys), 9)
        self.assertEqual(pending.dropped_chunks, 0)
        self.assertEqual(engine._gds_metrics.store_direct_chunks, 0)
        self.assertEqual(engine._gds_metrics.store_d2h_chunks, 9)
        self.assertTrue(pending.gds_fallback_d2h)
        kinds = [g["kind"] for g in engine.gathers]
        self.assertEqual(kinds, ["host"])

    def test_zero_slots_all_d2h(self):
        engine = self._engine(store_slots=0)
        pending = self._pending(engine, n=9)
        out = LMCacheEngine._issue_store_gds(engine, pending)
        self.assertEqual(len(out.gds_keys), 0)
        self.assertEqual(len(out.keys), 9)
        self.assertTrue(out.gds_fallback_d2h)
        self.assertEqual(out.dropped_chunks, 0)

    def test_enough_slots_all_emergency_d2h(self):
        engine = self._engine(store_slots=9)
        pending = self._pending(engine, n=9)
        out = LMCacheEngine._issue_store_gds(engine, pending)
        self.assertEqual(len(out.gds_keys), 0)
        self.assertEqual(len(out.keys), 9)
        self.assertEqual(engine._gds_metrics.store_d2h_chunks, 9)
        self.assertTrue(out.gds_fallback_d2h)

    def test_host_alloc_failure_no_drop(self):
        engine = self._engine(store_slots=2, alloc_fail_after=0)
        pending = self._pending(engine, n=9)
        out = LMCacheEngine._issue_store_gds(engine, pending)
        self.assertEqual(len(out.gds_keys), 0)
        self.assertEqual(len(out.keys), 0)
        self.assertEqual(out.dropped_chunks, 0)
        self.assertEqual(engine._gds_metrics.store_dropped_chunks, 0)
        self.assertIsNotNone(out.error)

    def test_gds_future_error_keeps_host_subset(self):
        from types import SimpleNamespace
        from lmcache.v1.cache_engine import GDS_PUT_FAILED

        engine = self._engine(store_slots=2)
        pending = self._pending(engine, n=4)
        LMCacheEngine._issue_store_gds(engine, pending)
        pending.put_future = SimpleNamespace(
            done=lambda: True, exception=lambda timeout=0: RuntimeError("cuFile")
        )
        pending.keys = pending.keys[:2]
        result = LMCacheEngine._harvest_gds_store(engine, pending)
        self.assertEqual(result, "failed")
        self.assertEqual(pending.gds_state, GDS_PUT_FAILED)
        self.assertEqual(len(pending.keys), 2)
        self.assertEqual(engine._gds_metrics.store_future_error, 1)
        self.assertIsNone(pending.error)

    def test_abandon_releases_host_and_leases(self):
        engine = self._engine(store_slots=2)
        pending = self._pending(engine, n=4)
        LMCacheEngine._issue_store_gds(engine, pending)
        engine._event_is_ready = lambda ev: True
        engine.abandoned_stores = []
        downs = []
        for mo in pending.memory_objs:
            mo.ref_count_down = lambda: downs.append(1)
        LMCacheEngine.abandon_store(engine, pending, wait_event=False)
        self.assertEqual(len(downs), 4)
        self.assertEqual(pending.gds_leases, [])
        self.assertEqual(engine._gds_d2h_inflight, 0)

    def test_d2h_budget_reject_is_nonblocking_whole_request(self):
        import time

        engine = self._engine(store_slots=2, alloc_fail_after=0)
        pending = self._pending(engine, n=9)
        t0 = time.monotonic()
        out = LMCacheEngine._issue_store_gds(engine, pending)
        self.assertLess(time.monotonic() - t0, 0.2)
        self.assertIsNotNone(out.error)
        self.assertEqual(len(out.keys), 0)
        self.assertEqual(len(out.memory_objs), 0)
        self.assertEqual(out.store_reject_reason, "d2h_budget_exhausted")
        self.assertGreaterEqual(engine._gds_metrics.d2h_alloc_fail, 1)

    def test_direct_permit_full_skips_native_iov(self):
        import time

        engine = self._engine(store_slots=9)
        engine.config.kv_gds_direct_store_max_inflight = 1
        engine._gds_direct_inflight = 1
        native = mock.Mock()
        engine._gds_remote_adapter = lambda: SimpleNamespace(
            kvb_store_from_gpu_iov=native
        )
        engine.gpu_connector.page_direct_enabled = lambda: True
        engine._pd_profile = SimpleNamespace(
            align=4096, header_region_bytes=4096, residual_capacity=0
        )
        pending = self._pending(engine, n=2)
        pending.kwargs_keepalive["slot_mapping"] = torch.zeros(16, dtype=torch.int64)
        t0 = time.monotonic()
        out = LMCacheEngine._issue_store_gds(engine, pending)
        self.assertLess(time.monotonic() - t0, 0.2)
        native.assert_not_called()
        self.assertTrue(out.gds_fallback_d2h)
        self.assertEqual(engine._gds_metrics.direct_permit_rejected, 1)
        self.assertEqual(len(out.keys), 2)

    def test_harvest_releases_permit_once(self):
        engine = self._engine(store_slots=2)
        pending = self._pending(engine, n=1)
        pending.direct_permit = 1
        engine._gds_direct_inflight = 1
        pending.put_future = SimpleNamespace(done=lambda: True, exception=lambda timeout=0: None)
        pending.pd_leases = []
        result = LMCacheEngine._harvest_gds_store(engine, pending)
        self.assertEqual(result, "done")
        self.assertEqual(pending.direct_permit, 0)
        self.assertEqual(engine._gds_direct_inflight, 0)
        LMCacheEngine._harvest_gds_store(engine, pending)
        self.assertEqual(engine._gds_direct_inflight, 0)

    def test_staging_starve_demotes_whole_request_nonblocking(self):
        import time

        engine = self._engine(store_slots=9)
        native = mock.Mock()
        engine._gds_remote_adapter = lambda: SimpleNamespace(
            kvb_store_from_gpu_iov=native
        )
        engine.gpu_connector.page_direct_enabled = lambda: True
        engine.gpu_connector.page_direct_probe = lambda: SimpleNamespace(
            zero_residual=True
        )
        engine.gpu_connector.acquire_page_direct_staging = mock.Mock(return_value=None)
        engine.gpu_connector.release_page_direct_staging = mock.Mock()
        engine.gpu_connector.get_page_direct_views = lambda *a, **k: SimpleNamespace(
            complete_pages=True, page_views=[], reason=""
        )
        engine._pd_profile = SimpleNamespace(
            align=4096, header_region_bytes=4096, residual_capacity=0
        )
        pending = self._pending(engine, n=3)
        pending.kwargs_keepalive["slot_mapping"] = torch.zeros(24, dtype=torch.int64)
        t0 = time.monotonic()
        out = LMCacheEngine._issue_store_gds(engine, pending)
        self.assertLess(time.monotonic() - t0, 0.2)
        native.assert_not_called()
        self.assertTrue(out.gds_fallback_d2h)
        self.assertEqual(len(out.keys), 3)
        self.assertEqual(engine._gds_metrics.staging_lease_starved, 1)
        self.assertEqual(engine._gds_metrics.store_d2h_chunks, 3)
        self.assertEqual(engine._gds_metrics.store_direct_chunks, 0)

    def test_staging_starve_releases_acquired_permit(self):
        engine = self._engine(store_slots=9)
        engine.config.kv_gds_direct_store_max_inflight = 4
        engine._gds_direct_inflight = 0
        engine._gds_remote_adapter = lambda: SimpleNamespace(
            kvb_store_from_gpu_iov=mock.Mock()
        )
        engine.gpu_connector.page_direct_enabled = lambda: True
        engine.gpu_connector.page_direct_probe = lambda: SimpleNamespace(
            zero_residual=True
        )
        engine.gpu_connector.acquire_page_direct_staging = mock.Mock(return_value=None)
        engine.gpu_connector.get_page_direct_views = lambda *a, **k: SimpleNamespace(
            complete_pages=True, page_views=[], reason=""
        )
        engine._pd_profile = SimpleNamespace(
            align=4096, header_region_bytes=4096, residual_capacity=0
        )
        pending = self._pending(engine, n=2)
        pending.kwargs_keepalive["slot_mapping"] = torch.zeros(16, dtype=torch.int64)
        out = LMCacheEngine._issue_store_gds(engine, pending)
        self.assertTrue(out.gds_fallback_d2h)
        self.assertEqual(out.direct_permit, 0)
        self.assertEqual(engine._gds_direct_inflight, 0)

    def test_issue_store_gds_passes_cpu_mapping_to_views(self):
        seen = []
        engine = self._engine(store_slots=9)
        native = mock.Mock()
        engine._gds_remote_adapter = lambda: SimpleNamespace(
            kvb_store_from_gpu_iov=native
        )
        engine.gpu_connector.page_direct_enabled = lambda: True
        engine.gpu_connector.page_direct_probe = lambda: SimpleNamespace(
            zero_residual=True
        )
        engine.gpu_connector.acquire_page_direct_staging = mock.Mock(return_value=None)
        engine.gpu_connector.release_page_direct_staging = mock.Mock()
        engine.gpu_connector.initialize_kvcaches_ptr = lambda **kw: None
        engine.gpu_connector.get_page_direct_views = (
            lambda slot, start, end: seen.append(slot)
            or SimpleNamespace(complete_pages=True, page_views=[], reason="")
        )
        engine._pd_profile = SimpleNamespace(
            align=4096, header_region_bytes=4096, residual_capacity=0
        )
        pending = self._pending(engine, n=2)
        cpu = torch.zeros(16, dtype=torch.int64)
        pending.kwargs_keepalive["slot_mapping"] = cpu
        LMCacheEngine._issue_store_gds(engine, pending)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], cpu)
        self.assertEqual(seen[0].device.type, "cpu")

    def test_d2h_bytes_cap_rejects_nonblocking(self):
        import time

        engine = self._engine(store_slots=2, admit_bytes=1)
        pending = self._pending(engine, n=2)
        t0 = time.monotonic()
        out = LMCacheEngine._issue_store_gds(engine, pending)
        self.assertLess(time.monotonic() - t0, 0.2)
        self.assertIsNotNone(out.error)
        self.assertEqual(len(out.keys), 0)
        self.assertEqual(out.store_reject_reason, "d2h_budget_exhausted")
        self.assertGreaterEqual(engine._gds_metrics.d2h_budget_exhausted, 1)

    def test_header_pin_buffer_reused_after_release(self):
        engine = self._engine(store_slots=1)
        packed = bytes(64)
        first = LMCacheEngine._fill_pd_header_host(engine, packed)
        self.assertEqual(len(engine._pd_header_pins), 1)
        pending = SimpleNamespace(pd_keepalive=[first])
        LMCacheEngine._release_pd_header_hosts(engine, pending)
        second = LMCacheEngine._fill_pd_header_host(engine, packed)
        self.assertEqual(len(engine._pd_header_pins), 1)
        self.assertEqual(int(first.data_ptr()), int(second.data_ptr()))

    def test_store_split_metrics_record_cpu_cuda_native(self):
        engine = self._engine(store_slots=0)
        pending = self._pending(engine, n=1)
        LMCacheEngine._issue_store_gds(engine, pending)
        snap = engine._gds_metrics.snapshot()
        self.assertGreaterEqual(snap["store_split_n"], 1)
        self.assertGreaterEqual(snap["store_cpu_prep_p50_us"], 0)


@unittest.skipUnless(torch is not None, "torch required")
class TestPageDirectStoreInit(unittest.TestCase):
    def test_init_order_kvcaches_before_register(self):
        order = []
        connector = SimpleNamespace(
            page_direct_enabled=lambda: False,
            initialize_kvcaches_ptr=lambda **kw: order.append("init"),
            configure_page_direct_registration=lambda *a, **k: order.append("cfg"),
            register_page_direct_kv_ranges=lambda **kw: order.append("register"),
        )
        adapter = SimpleNamespace(
            kvb_gds_geometry=lambda: SimpleNamespace(
                gds_ok=1,
                writev_ex_ok=1,
                readv_ex_ok=1,
                sub_stripe_bytes=8192,
                sector_bytes=4096,
                gds_registration_limit=32,
            ),
            gds_buf_register=lambda *a: None,
            gds_buf_deregister=lambda *a: None,
        )
        engine = mock.Mock(spec=LMCacheEngine)
        engine.gpu_connector = connector
        engine._gds_remote_adapter = lambda: adapter
        engine._gds_metrics = GdsPipelineMetrics()
        engine.config = SimpleNamespace(kv_gds_staging_arena_bytes=64 * 1024 * 1024)
        LMCacheEngine._maybe_init_page_direct(engine, kvcaches=["kv"])
        self.assertEqual(order[0], "init")
        self.assertIn("register", order)
        self.assertLess(order.index("init"), order.index("register"))

    def test_issue_does_not_device_sync_and_keeps_leases_until_future(self):

        sync_calls = []
        released = []
        captured = {}

        class _Fut:
            def __init__(self):
                self._d = False
                self._exc = None

            def done(self):
                return self._d

            def exception(self, timeout=0):
                return self._exc

        fut = _Fut()

        def store_iov(*args, **kwargs):
            captured["event"] = kwargs.get("cuda_event")
            captured["called"] = True
            return fut

        packed = bytes(64)
        rec = {
            "kv_block_index": 0,
            "payload_encoding": 1,
            "header_ptr": 1,
            "header_is_device": 1,
            "gpu_ptrs": [],
            "record_offs": [],
            "lens": [],
            "residual_ptr": 0,
            "residual_len": 0,
            "packed_header": packed,
            "layout": SimpleNamespace(residuals=[]),
        }
        layout = SimpleNamespace(residuals=[])
        arena = torch.zeros(128, dtype=torch.uint8)
        connector = SimpleNamespace(
            page_direct_enabled=lambda: True,
            page_direct_probe=lambda: SimpleNamespace(zero_residual=True),
            get_page_direct_views=lambda *a, **k: SimpleNamespace(
                complete_pages=True, page_views=[], reason=""
            ),
            acquire_page_direct_staging=lambda *a, **k: (1, 0, 64),
            release_page_direct_staging=lambda off, ln: released.append((off, ln)),
            page_direct_staging_tensor=lambda: arena,
            initialize_kvcaches_ptr=lambda **kw: None,
        )
        engine = mock.Mock(spec=LMCacheEngine)
        engine.config = SimpleNamespace(
            kv_io_mode="gds", kv_gds_direct_store_max_inflight=0
        )
        engine.gpu_connector = connector
        engine._gds_metrics = GdsPipelineMetrics()
        engine._pd_profile = SimpleNamespace(
            align=64, header_region_bytes=64, residual_capacity=0
        )
        engine._gds_remote_adapter = lambda: SimpleNamespace(
            kvb_store_from_gpu_iov=store_iov
        )
        engine._maybe_prewarm_gds_pool = lambda **kw: None
        engine._gds_max_waiters = lambda: 32
        engine._try_acquire_direct_permit = lambda pending: True
        engine._release_direct_permit = lambda pending: None
        engine._gds_direct_lock = threading.Lock()
        engine._gds_direct_inflight = 0
        engine._pd_header_pin_lock = threading.Lock()
        engine._pd_header_pins = []
        engine._gds_immediate_drop = lambda: False
        engine._fill_pd_header_host = LMCacheEngine._fill_pd_header_host.__get__(engine)
        engine._acquire_pd_header_host = LMCacheEngine._acquire_pd_header_host.__get__(
            engine
        )
        engine._release_pd_header_hosts = LMCacheEngine._release_pd_header_hosts.__get__(
            engine
        )
        engine._gds_store_stream = LMCacheEngine._gds_store_stream.__get__(engine)
        engine._enqueue_pd_header_copies = LMCacheEngine._enqueue_pd_header_copies.__get__(
            engine
        )
        engine._note_gds_store_split = LMCacheEngine._note_gds_store_split.__get__(
            engine
        )
        engine.metadata = SimpleNamespace(
            get_shapes=lambda n: [torch.zeros(1)],
            get_dtypes=lambda: [torch.float16],
        )
        engine._nbytes_for_tokens = lambda n: 8
        engine.store_stats = SimpleNamespace(profile_from_gpu=lambda: mock.MagicMock())
        engine._release_gds_leases = lambda pending: released.append("leases")
        pending = SimpleNamespace(
            req_id="r1",
            keys=["k0"],
            memory_objs=[],
            token_ids=[1],
            kwargs_keepalive={
                "starts": [0],
                "ends": [8],
                "slot_mapping": torch.zeros(8, dtype=torch.int64),
            },
            store_stats=engine.store_stats,
            pd_leases=[],
            pd_keepalive=[],
            dropped_chunks=0,
            requested_chunks=0,
            accepted_chunks=0,
            gds_keys=[],
            gds_objs=[],
            gds_leases=[],
            put_future=None,
            gds_state="",
            error=None,
            gds_put_error=None,
            gds_fallback_d2h=False,
            tot_kv_size=0,
            tot_token_num=0,
            event=None,
            d2h_admitted=0,
            d2h_admitted_bytes=0,
            direct_permit=0,
            store_reject_reason="",
            t_gds_queue=0.0,
            t_native_first_submit=0.0,
            t_native_last_accept=0.0,
            t_native_done=0.0,
        )
        with mock.patch("torch.cuda.synchronize", side_effect=lambda: sync_calls.append(1)), mock.patch(
            "torch.cuda.is_available", return_value=False
        ), mock.patch(
            "lmcache.v1.cache_engine.build_raw_page_record",
            return_value=(dict(rec), layout),
        ), mock.patch(
            "lmcache.v1.cache_engine.profile_as_sdk_dict",
            return_value={"align_bytes": 64},
        ):
            out = LMCacheEngine._issue_store_gds(engine, pending)
        self.assertIs(out, pending)
        self.assertEqual(sync_calls, [])
        self.assertTrue(captured.get("called"))
        self.assertEqual(pending.pd_leases, [(0, 64)])
        pending.put_future = fut
        self.assertEqual(LMCacheEngine._harvest_gds_store(engine, pending), "pending")
        self.assertEqual(pending.pd_leases, [(0, 64)])
        fut._d = True
        self.assertEqual(LMCacheEngine._harvest_gds_store(engine, pending), "done")
        self.assertIn("leases", released)

    def test_kvb_gpu_iov_waits_event_on_executor(self):
        import concurrent.futures
        import threading
        import time

        try:
            from surdata_lmcache.inproc_adapter import SurDataRemoteConnector
        except ImportError:
            self.skipTest("SurData SDK adapter is not part of this public tree")

        devices: list[int] = []

        class Ev:
            def __init__(self) -> None:
                self.gate = threading.Event()
                self.n = 0

            def synchronize(self) -> None:
                self.gate.wait(timeout=2)
                self.n += 1

        class NativeFut:
            def __init__(self) -> None:
                self._d = False
                self._rc = 0
                self.result_calls = 0

            def done(self) -> bool:
                return self._d

            def rc(self) -> int:
                return self._rc

            def error_message(self) -> str:
                return "native fail" if self._rc else ""

            def result(self) -> int:
                self.result_calls += 1
                return 2

        class Adapter:
            _closing = False
            _block_tokens = 16
            _elem_bytes = 2
            _tail_elems = 1

            def __init__(self, store_fn, pool) -> None:
                self._stats_lock = threading.Lock()
                self._gds_iov_submit_started = 0
                self._gds_iov_native_submitted = 0
                self._gds_iov_submit_error = 0
                self._gds_iov_native_done = 0
                self.client = SimpleNamespace(
                    surdata_native=SimpleNamespace(
                        prefix_hashes_from_tokens=lambda ids, bt: ["h"],
                        kvb_store_from_gpu_iov=store_fn,
                    )
                )
                self._gds_io_pool = pool

            def _gds_io_executor(self):
                return self._gds_io_pool

            def _gds_iov_count(self, name: str) -> None:
                with self._stats_lock:
                    setattr(self, name, int(getattr(self, name, 0) or 0) + 1)

        native = NativeFut()
        event = Ev()

        def store_fn(*args, **kwargs):
            self.assertEqual(event.n, 1)
            return native

        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="t-gds-iov"
        )
        adapter = Adapter(store_fn, pool)
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.current_device", return_value=3
        ), mock.patch("torch.cuda.set_device", side_effect=lambda d: devices.append(d)):
            fut = SurDataRemoteConnector.kvb_store_from_gpu_iov(
                adapter,
                [1, 2, 3],
                [{"header_ptr": 1}],
                {"align_bytes": 8192},
                cuda_event=event,
            )
            self.assertFalse(fut.done())
            self.assertEqual(event.n, 0)
            event.gate.set()
            deadline = time.monotonic() + 2
            while adapter._gds_iov_native_submitted == 0 and time.monotonic() < deadline:
                fut.done()
                time.sleep(0.01)
            self.assertEqual(adapter._gds_iov_native_submitted, 1)
            self.assertFalse(fut.done())
            self.assertIsNone(fut.exception(timeout=0))
            self.assertEqual(native.result_calls, 0)
            self.assertEqual(devices, [3])
            released: list[str] = []
            engine = mock.Mock(spec=LMCacheEngine)
            engine._gds_metrics = GdsPipelineMetrics()
            engine._release_gds_leases = lambda pending: released.append("leases")
            pending = SimpleNamespace(
                req_id="r1",
                put_future=fut,
                pd_leases=[(0, 8)],
                gds_leases=[],
            )
            self.assertEqual(LMCacheEngine._harvest_gds_store(engine, pending), "pending")
            self.assertEqual(pending.pd_leases, [(0, 8)])
            native._d = True
            self.assertTrue(fut.done())
            self.assertIsNone(fut.exception(timeout=0))
            self.assertEqual(native.result_calls, 0)
            pending.put_future = fut
            pending.pd_leases = [(0, 8)]
            self.assertEqual(LMCacheEngine._harvest_gds_store(engine, pending), "done")
            self.assertIn("leases", released)
            self.assertEqual(fut.result(timeout=2), 2)
        pool.shutdown(wait=True)

    def test_kvb_gpu_iov_submit_error_and_native_rc(self):
        import concurrent.futures
        import threading

        try:
            from surdata_lmcache.inproc_adapter import SurDataRemoteConnector
        except ImportError:
            self.skipTest("SurData SDK adapter is not part of this public tree")

        class NativeFut:
            def __init__(self, rc: int) -> None:
                self._d = True
                self._rc = rc
                self.result_calls = 0

            def done(self) -> bool:
                return self._d

            def rc(self) -> int:
                return self._rc

            def error_message(self) -> str:
                return "native fail"

            def result(self) -> int:
                self.result_calls += 1
                raise RuntimeError("must not block")

        class Adapter:
            _closing = False
            _block_tokens = 16
            _elem_bytes = 2
            _tail_elems = 1

            def __init__(self, store_fn, pool) -> None:
                self._stats_lock = threading.Lock()
                self._gds_iov_submit_started = 0
                self._gds_iov_native_submitted = 0
                self._gds_iov_submit_error = 0
                self._gds_iov_native_done = 0
                self.client = SimpleNamespace(
                    surdata_native=SimpleNamespace(
                        prefix_hashes_from_tokens=lambda ids, bt: ["h"],
                        kvb_store_from_gpu_iov=store_fn,
                    )
                )
                self._gds_io_pool = pool

            def _gds_io_executor(self):
                return self._gds_io_pool

            def _gds_iov_count(self, name: str) -> None:
                with self._stats_lock:
                    setattr(self, name, int(getattr(self, name, 0) or 0) + 1)

        class BoomPool:
            def submit(self, fn):
                fut = concurrent.futures.Future()
                fut.set_exception(RuntimeError("submit boom"))
                return fut

        released: list[str] = []
        engine = mock.Mock(spec=LMCacheEngine)
        engine._gds_metrics = GdsPipelineMetrics()
        engine._release_gds_leases = lambda pending: released.append("leases")

        boom_adapter = Adapter(lambda *a, **k: None, BoomPool())
        boom_fut = SurDataRemoteConnector.kvb_store_from_gpu_iov(
            boom_adapter, [1], [{"header_ptr": 1}], {"align_bytes": 8192}
        )
        self.assertTrue(boom_fut.done())
        self.assertIsInstance(boom_fut.exception(timeout=0), RuntimeError)
        pending = SimpleNamespace(
            req_id="r-submit",
            put_future=boom_fut,
            pd_leases=[(1, 2)],
            gds_leases=[],
        )
        self.assertEqual(LMCacheEngine._harvest_gds_store(engine, pending), "failed")
        self.assertIn("leases", released)

        class CancelExec:
            def submit(self, fn):
                fut = concurrent.futures.Future()
                fut.cancel()
                return fut

        cancel_adapter = Adapter(lambda *a, **k: None, CancelExec())
        cancel_fut = SurDataRemoteConnector.kvb_store_from_gpu_iov(
            cancel_adapter, [1], [{"header_ptr": 1}], {"align_bytes": 8192}
        )
        self.assertTrue(cancel_fut.done())
        self.assertIsInstance(
            cancel_fut.exception(timeout=0), concurrent.futures.CancelledError
        )

        native = NativeFut(rc=7)
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        rc_adapter = Adapter(lambda *a, **k: native, pool)
        rc_fut = SurDataRemoteConnector.kvb_store_from_gpu_iov(
            rc_adapter, [1], [{"header_ptr": 1}], {"align_bytes": 8192}
        )
        rc_fut._exec_fut.result(timeout=2)
        self.assertTrue(rc_fut.done())
        exc = rc_fut.exception(timeout=0)
        self.assertIsInstance(exc, RuntimeError)
        self.assertIn("native fail", str(exc))
        self.assertEqual(native.result_calls, 0)
        released.clear()
        pending = SimpleNamespace(
            req_id="r-rc",
            put_future=rc_fut,
            pd_leases=[(3, 4)],
            gds_leases=[],
        )
        self.assertEqual(LMCacheEngine._harvest_gds_store(engine, pending), "failed")
        self.assertIn("leases", released)
        pool.shutdown(wait=True)


class TestGdsLookupHandshake(unittest.TestCase):
    def test_storage_manager_skips_gds_lookup_host_get(self):
        root = Path(__file__).resolve().parents[2]
        src = (
            root / "lmcache/v1/storage_backend/storage_manager.py"
        ).read_text()
        self.assertIn("GDS skip lookup-time host GET", src)

    def test_sync_lookup_client_sends_token_ids(self):
        try:
            from lmcache.v1.lookup_client.lmcache_lookup_client import (
                LMCacheLookupClient,
            )
        except ImportError as exc:
            self.skipTest(str(exc))

        captured: dict[str, object] = {}

        class Transport:
            world_size = 1

            def send_and_recv_all(self, msg_buf):
                captured["msg"] = msg_buf
                return [(8).to_bytes(4, "big")]

        client = LMCacheLookupClient.__new__(LMCacheLookupClient)
        client.enable_blending = False
        client.token_database = SimpleNamespace(
            process_tokens=lambda token_ids, make_key=False: [
                (0, 8, 11),
                (8, 16, 22),
            ]
        )
        client.transport = Transport()
        client.reqs_status = {}
        tokens = list(range(16))
        hit = LMCacheLookupClient.lookup(client, tokens, "req-1")
        self.assertEqual(hit, 8)
        msg = captured["msg"]
        self.assertEqual(msg[0], tokens)
        self.assertEqual(msg[1], [11, 22])
        self.assertEqual(msg[2], [8, 8])
        self.assertEqual(msg[3], "req-1")


@unittest.skipUnless(torch is not None, "torch required")
class TestDelayedFreeWatermark(unittest.TestCase):
    def test_infer_vllm_0201_shapes(self):
        try:
            from lmcache.integration.vllm.vllm_v1_adapter import infer_num_gpu_blocks
        except ImportError as exc:
            self.skipTest(str(exc))
        mla = torch.empty(6472, 64, 576)
        fa5 = torch.empty(6472, 2, 64, 8, 128)
        stacked = torch.empty(2, 6472, 64, 8, 128)
        self.assertEqual(infer_num_gpu_blocks({"l0": mla, "l1": mla}), 6472)
        self.assertEqual(infer_num_gpu_blocks({"l0": fa5}), 6472)
        self.assertEqual(infer_num_gpu_blocks({"l0": stacked}), 6472)
        self.assertEqual(infer_num_gpu_blocks({}), 0)

    def test_ratio_and_absolute_take_min(self):
        try:
            from lmcache.integration.vllm.vllm_v1_adapter import (
                LMCacheConnectorV1Impl,
            )
        except ImportError as exc:
            self.skipTest(str(exc))
        inst = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
        inst._async_max_pending_bytes = 0
        inst._async_max_pending_requests = 0
        inst._async_max_delayed_free_blocks = 100
        inst._async_max_delayed_free_ratio = 0.25
        inst._gpu_kv_block_total = 400
        inst._pending_bytes = 0
        inst._delayed_free_blocks = 90
        inst._pending_saves = {}
        inst._gpu_kv_block_total_error_logged = False
        # min(100, 400*0.25=100) = 100; 90+20 > 100
        self.assertTrue(
            LMCacheConnectorV1Impl._async_store_over_watermark(inst, "r", 0, 20)
        )
        self.assertEqual(inst._async_store_drop_reason, "watermark_blocks")
        inst._delayed_free_blocks = 0
        self.assertFalse(
            LMCacheConnectorV1Impl._async_store_over_watermark(inst, "r", 0, 20)
        )

    def test_ratio_unknown_total_refuses_all(self):
        try:
            from lmcache.integration.vllm.vllm_v1_adapter import (
                LMCacheConnectorV1Impl,
            )
        except ImportError as exc:
            self.skipTest(str(exc))
        inst = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
        inst._async_max_pending_bytes = 0
        inst._async_max_pending_requests = 0
        inst._async_max_delayed_free_blocks = 0
        inst._async_max_delayed_free_ratio = 0.25
        inst._gpu_kv_block_total = 0
        inst._pending_bytes = 0
        inst._delayed_free_blocks = 0
        inst._pending_saves = {}
        inst._gpu_kv_block_total_error_logged = False
        self.assertTrue(
            LMCacheConnectorV1Impl._async_store_over_watermark(inst, "r", 0, 1)
        )
        self.assertEqual(inst._async_store_drop_reason, "watermark_ratio_unknown")

    def test_zero_caps_disable(self):
        try:
            from lmcache.integration.vllm.vllm_v1_adapter import (
                LMCacheConnectorV1Impl,
            )
        except ImportError as exc:
            self.skipTest(str(exc))
        inst = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
        inst._async_max_pending_bytes = 0
        inst._async_max_pending_requests = 0
        inst._async_max_delayed_free_blocks = 0
        inst._async_max_delayed_free_ratio = 0.0
        inst._gpu_kv_block_total = 0
        inst._pending_bytes = 10**12
        inst._delayed_free_blocks = 10**6
        inst._pending_saves = {"a": [1], "b": [2]}
        self.assertFalse(
            LMCacheConnectorV1Impl._async_store_over_watermark(inst, "r", 1, 1)
        )

    def test_try_issue_h2d_polls_unread_future(self):
        engine = mock.Mock(spec=LMCacheEngine)
        engine.config = SimpleNamespace(kv_io_mode="gds")
        engine._try_complete_page_direct = (
            LMCacheEngine._try_complete_page_direct.__get__(engine)
        )
        engine._release_gds_leases = mock.Mock()
        adapter = SimpleNamespace(
            kvb_read_to_gpu_poll=lambda fut: (False, 0),
        )
        pending = PendingLoad(req_id="r1", chunks=[])
        pending.pd_adapter = adapter
        pending.gds_read_fut = object()
        pending.pd_prefix = 2
        status = LMCacheEngine.try_issue_h2d(engine, pending)
        self.assertEqual(status, "pending_gather")
        engine._release_gds_leases.assert_not_called()

    def test_read_failure_fail_closed(self):
        engine = mock.Mock(spec=LMCacheEngine)
        engine.config = SimpleNamespace(kv_io_mode="gds")
        engine._try_complete_page_direct = (
            LMCacheEngine._try_complete_page_direct.__get__(engine)
        )
        engine._release_gds_leases = mock.Mock()
        adapter = SimpleNamespace(
            kvb_read_to_gpu_poll=lambda fut: (True, -5),
        )
        pending = PendingLoad(req_id="r1", chunks=[])
        pending.pd_adapter = adapter
        pending.gds_read_fut = SimpleNamespace(
            done=lambda: True, rc=lambda: -5, error_message=lambda: "hole"
        )
        pending.pd_prefix = 2
        status = LMCacheEngine.try_issue_h2d(engine, pending)
        self.assertEqual(status, "error")
        engine._release_gds_leases.assert_called_once()


class TestGdsCompleteLoadLease(unittest.TestCase):
    def test_complete_load_tuple_leases_no_memory_obj(self):
        try:
            from lmcache.v1.cache_engine import LMCacheEngine, PendingLoad
        except ImportError as exc:
            self.skipTest(str(exc))
        engine = mock.Mock(spec=LMCacheEngine)
        engine.config = SimpleNamespace(kv_io_mode="gds")
        engine.save_only_first_rank = False
        engine.abandoned_loads = []
        engine._gds_lease_memory_ids = LMCacheEngine._gds_lease_memory_ids.__get__(
            engine
        )
        engine._release_gds_leases = mock.Mock()
        engine._release_retrieve_chunks = mock.Mock()
        pending = PendingLoad(req_id="r1", chunks=[])
        pending.gds_leases = [(1, 0, 4096)]
        pending.ret_mask = [True] * 8
        LMCacheEngine.complete_load(engine, pending)
        engine._release_gds_leases.assert_called_once()

    def test_abandon_load_waits_inflight_gds_read(self):
        try:
            from lmcache.v1.cache_engine import LMCacheEngine, PendingLoad
        except ImportError as exc:
            self.skipTest(str(exc))
        engine = mock.Mock(spec=LMCacheEngine)
        engine.config = SimpleNamespace(kv_io_mode="gds")
        engine.abandoned_loads = []
        engine._event_is_ready = LMCacheEngine._event_is_ready.__get__(engine)
        engine._gds_retrieve_still_busy = (
            LMCacheEngine._gds_retrieve_still_busy.__get__(engine)
        )
        engine.complete_load = mock.Mock()
        pending = PendingLoad(req_id="r1", chunks=[])
        pending.gds_read_fut = SimpleNamespace(done=lambda: False)
        pending.pd_adapter = SimpleNamespace(
            kvb_read_to_gpu_poll=lambda fut: (False, 0)
        )
        pending.gds_leases = [(1, 0, 4096)]
        LMCacheEngine.abandon_load(engine, pending)
        engine.complete_load.assert_not_called()
        self.assertEqual(engine.abandoned_loads, [pending])


class TestGdsHostPutNonblocking(unittest.TestCase):
    def test_async_gds_skips_future_result(self):
        try:
            from lmcache.v1.storage_backend.remote_backend import RemoteBackend
        except ImportError as exc:
            self.skipTest(str(exc))

        be = RemoteBackend.__new__(RemoteBackend)
        be.config = SimpleNamespace(
            kv_io_mode="gds",
            async_store_d2h_enabled=True,
            blocking_timeout_secs=1,
        )
        self.assertFalse(RemoteBackend._gds_blocking_host_put(be))
        be.config.async_store_d2h_enabled = False
        self.assertTrue(RemoteBackend._gds_blocking_host_put(be))


class _CudaSlotFake:
    """CUDA-marked mapping that fails closed if the hot path tries to D2H."""

    is_cuda = True
    device = SimpleNamespace(type="cuda")

    def __getitem__(self, key):
        raise AssertionError("CUDA slot_mapping must not be sliced")

    def detach(self):
        raise AssertionError("must not detach CUDA slot_mapping")

    def to(self, *args, **kwargs):
        raise AssertionError("must not .to(cpu) on CUDA slot_mapping")


@unittest.skipUnless(torch is not None, "torch required")
class TestGdsSlotMappingCpuContract(unittest.TestCase):
    def test_get_page_direct_views_rejects_cuda(self):
        from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2

        conn = VLLMPagedMemGPUConnectorV2.__new__(VLLMPagedMemGPUConnectorV2)
        conn._pd_enabled = True
        conn.kvcaches = [object()]
        with self.assertRaises(ValueError) as ctx:
            VLLMPagedMemGPUConnectorV2.get_page_direct_views(
                conn, _CudaSlotFake(), 0, 8
            )
        self.assertIn("get_page_direct_views requires CPU slot_mapping", str(ctx.exception))

    def test_get_page_direct_views_accepts_cpu(self):
        from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2

        conn = VLLMPagedMemGPUConnectorV2.__new__(VLLMPagedMemGPUConnectorV2)
        conn._pd_enabled = True
        conn.kvcaches = [object()]
        conn.block_size = 16
        conn.num_blocks = 8
        conn._pd_layer_bases = []
        conn._pd_page_bytes = 73728
        conn._pd_runs = []
        cpu = torch.arange(16, dtype=torch.int64)
        adm = VLLMPagedMemGPUConnectorV2.get_page_direct_views(conn, cpu, 0, 16)
        self.assertFalse(adm.complete_pages)

    def test_issue_store_gds_rejects_cuda_slot_mapping(self):
        engine = mock.Mock(spec=LMCacheEngine)
        engine.config = SimpleNamespace(
            kv_io_mode="gds", kv_gds_direct_store_max_inflight=0
        )
        engine.gpu_connector = SimpleNamespace(
            page_direct_enabled=lambda: True,
            get_page_direct_views=lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("must fail closed before views")
            ),
            initialize_kvcaches_ptr=lambda **kw: None,
        )
        engine._gds_metrics = GdsPipelineMetrics()
        engine._pd_profile = SimpleNamespace(align=64, header_region_bytes=64)
        engine._gds_remote_adapter = lambda: SimpleNamespace(
            kvb_store_from_gpu_iov=mock.Mock()
        )
        engine._maybe_prewarm_gds_pool = lambda **kw: None
        engine._try_acquire_direct_permit = lambda pending: True
        engine._gds_immediate_drop = lambda: False
        pending = SimpleNamespace(
            req_id="r1",
            keys=["k0"],
            kwargs_keepalive={
                "starts": [0],
                "ends": [8],
                "slot_mapping": _CudaSlotFake(),
            },
            error=None,
        )
        with self.assertRaises(ValueError) as ctx:
            LMCacheEngine._issue_store_gds(engine, pending)
        self.assertIn("get_page_direct_views requires CPU slot_mapping", str(ctx.exception))

    def test_submit_page_direct_reads_rejects_cuda(self):
        engine = mock.Mock(spec=LMCacheEngine)
        engine.gpu_connector = SimpleNamespace(
            get_page_direct_views=lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("must fail closed before views")
            )
        )
        engine._pd_profile = SimpleNamespace(align=64, record_extent_bytes=64)
        engine._gds_max_waiters = lambda: 1
        pending = PendingLoad(req_id="r1", chunks=[])
        pending.pd_adapter = SimpleNamespace(kvb_read_to_gpu_submit=lambda *a, **k: None)
        pending.pd_hit = object()
        pending.pd_prefix = 1
        pending.pd_spans = [(0, 8)]
        with self.assertRaises(ValueError) as ctx:
            LMCacheEngine._submit_page_direct_reads(
                engine, pending, None, None, slot_mapping=_CudaSlotFake()
            )
        self.assertIn("get_page_direct_views requires CPU slot_mapping", str(ctx.exception))

    def test_submit_page_direct_reads_passes_cpu_mapping(self):
        seen = []
        engine = mock.Mock(spec=LMCacheEngine)
        engine.gpu_connector = SimpleNamespace(
            get_page_direct_views=lambda slot, start, end: seen.append(slot)
            or SimpleNamespace(complete_pages=True, page_views=[], reason=""),
            acquire_page_direct_staging=lambda *a, **k: (1, 0, 64),
        )
        engine._pd_profile = SimpleNamespace(
            align=64, record_extent_bytes=64
        )
        engine._gds_max_waiters = lambda: 1
        cpu = torch.zeros(8, dtype=torch.int64)
        pending = PendingLoad(req_id="r1", chunks=[])
        pending.pd_adapter = SimpleNamespace(
            kvb_read_to_gpu_submit=lambda *a, **k: SimpleNamespace()
        )
        pending.pd_hit = object()
        pending.pd_prefix = 1
        pending.pd_spans = [(0, 8)]
        with mock.patch(
            "lmcache.v1.cache_engine.profile_as_sdk_dict",
            return_value={},
        ):
            LMCacheEngine._submit_page_direct_reads(
                engine, pending, None, None, slot_mapping=cpu
            )
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], cpu)
        self.assertEqual(seen[0].device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
