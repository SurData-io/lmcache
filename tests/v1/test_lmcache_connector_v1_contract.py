# SPDX-License-Identifier: Apache-2.0
"""Stock vLLM 0.20.1 connector wrapper contract (no overlay APIs)."""

from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock


def _install_vllm_stubs() -> None:
    if "vllm.config" in sys.modules:
        return

    def _mod(name: str, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    _mod("vllm")
    _mod("vllm.config", VllmConfig=object)
    base = _mod("vllm.distributed")
    kv = _mod("vllm.distributed.kv_transfer")
    conn = _mod("vllm.distributed.kv_transfer.kv_connector")
    v1 = _mod("vllm.distributed.kv_transfer.kv_connector.v1")

    class KVConnectorMetadata:
        pass

    class KVConnectorRole:
        SCHEDULER = SimpleNamespace(name="SCHEDULER")
        WORKER = SimpleNamespace(name="WORKER")

    class KVConnectorBase_V1:
        def __init__(self, vllm_config=None, role=None, kv_cache_config=None):
            self._vllm_config = vllm_config
            self._role = role

    v1base = _mod(
        "vllm.distributed.kv_transfer.kv_connector.v1.base",
        KVConnectorBase_V1=KVConnectorBase_V1,
        KVConnectorMetadata=KVConnectorMetadata,
        KVConnectorRole=KVConnectorRole,
    )
    for parent, name, child in (
        (sys.modules["vllm"], "distributed", base),
        (base, "kv_transfer", kv),
        (kv, "kv_connector", conn),
        (conn, "v1", v1),
        (v1, "base", v1base),
    ):
        setattr(parent, name, child)

    _mod("vllm.distributed.parallel_state", get_pp_group=lambda: None)
    _mod("vllm.sampling_params", SamplingParams=object)
    _mod("vllm.v1")
    _mod("vllm.v1.core")
    _mod("vllm.v1.core.sched")
    _mod("vllm.v1.core.sched.output", SchedulerOutput=object)
    _mod("vllm.v1.request", RequestStatus=SimpleNamespace(FINISHED_ABORTED="ABORTED"))
    _mod("vllm.version", __version__="0.20.1-stub")
    _mod("vllm.logger", init_logger=lambda name: mock.Mock())
    _mod("vllm.v1.outputs")


_install_vllm_stubs()

try:
    from lmcache.integration.vllm.lmcache_connector_v1 import (
        LMCacheConnectorV1Dynamic,
    )
    from lmcache.integration.vllm.vllm_v1_adapter import (
        LoadSpec,
        LMCacheConnectorMetadata,
        LMCacheConnectorV1Impl,
    )
except Exception:  # pragma: no cover
    LMCacheConnectorV1Dynamic = None  # type: ignore
    LoadSpec = None  # type: ignore
    LMCacheConnectorMetadata = None  # type: ignore
    LMCacheConnectorV1Impl = None  # type: ignore


@unittest.skipUnless(LMCacheConnectorV1Dynamic is not None, "vllm/torch connector deps missing")
class TestConnectorWrapperContract(unittest.TestCase):
    def test_get_num_new_matched_tokens_is_flat_tuple(self):
        wrapper = LMCacheConnectorV1Dynamic.__new__(LMCacheConnectorV1Dynamic)
        wrapper._lmcache_engine = SimpleNamespace(
            get_num_new_matched_tokens=lambda request, n: (2048, True)
        )
        out = LMCacheConnectorV1Dynamic.get_num_new_matched_tokens(
            wrapper, SimpleNamespace(), 0
        )
        self.assertEqual(out, (2048, True))
        self.assertIsInstance(out[0], int)
        self.assertIsInstance(out[1], bool)

    def test_get_num_new_matched_tokens_none_wait(self):
        wrapper = LMCacheConnectorV1Dynamic.__new__(LMCacheConnectorV1Dynamic)
        wrapper._lmcache_engine = SimpleNamespace(
            get_num_new_matched_tokens=lambda request, n: (None, True)
        )
        out = LMCacheConnectorV1Dynamic.get_num_new_matched_tokens(
            wrapper, SimpleNamespace(), 0
        )
        self.assertEqual(out, (None, True))

    def test_update_state_after_alloc_forwards_blocks(self):
        seen = {}

        def update(request, blocks, num_external_tokens=0):
            seen["blocks"] = blocks
            seen["n"] = num_external_tokens

        wrapper = LMCacheConnectorV1Dynamic.__new__(LMCacheConnectorV1Dynamic)
        wrapper._lmcache_engine = SimpleNamespace(update_state_after_alloc=update)
        blocks = object()
        LMCacheConnectorV1Dynamic.update_state_after_alloc(
            wrapper, SimpleNamespace(), blocks, 2048
        )
        self.assertIs(seen["blocks"], blocks)
        self.assertEqual(seen["n"], 2048)

    def test_handle_preemptions_forwarded(self):
        called = []
        wrapper = LMCacheConnectorV1Dynamic.__new__(LMCacheConnectorV1Dynamic)
        wrapper._lmcache_engine = SimpleNamespace(
            handle_preemptions=lambda meta: called.append(meta)
        )
        meta = object()
        LMCacheConnectorV1Dynamic.handle_preemptions(wrapper, meta)
        self.assertEqual(called, [meta])

    def test_overlay_methods_absent(self):
        for name in (
            "lookup_after_hbm_alloc",
            "start_dram_preload",
            "bind_gpu_block_pool",
            "prefetch_admission_ok",
            "take_xfer_records",
            "cancel_async_load",
        ):
            self.assertFalse(hasattr(LMCacheConnectorV1Impl, name), name)
        self.assertTrue(hasattr(LMCacheConnectorV1Impl, "handle_preemptions"))


@unittest.skipUnless(LMCacheConnectorV1Impl is not None, "vllm/torch connector deps missing")
class TestGdsLookupNoCap(unittest.TestCase):
    def _adapter(self, hits, io_mode="gds"):
        obj = SimpleNamespace(
            lookup_client=SimpleNamespace(
                lookup_cache=lambda lookup_id: hits,
            ),
            kv_role="kv_both",
            config=SimpleNamespace(
                min_retrieve_tokens=0,
                kv_io_mode=io_mode,
                extra_config={"kv.io_mode": io_mode},
            ),
            load_specs={},
            _max_tokens_per_load=256 if io_mode == "gds" else 0,
            _lmcache_chunk_size=256,
            _hbm_cancelled_req_ids=set(),
            skip_last_n_tokens=0,
            _requests_priority={},
        )
        obj.kv_io_mode = lambda: io_mode
        obj._hbm_prefetch_enabled = lambda: False
        obj._poll_or_start_lookup = lambda request: hits
        obj._gds_lookup_defer_started = {}
        obj._gds_lookup_defer_counts = {}
        obj._gds_lookup_fallback_ids = set()
        obj._gds_lookup_fallback_count = 0
        obj._gds_lookup_late_hit_count = 0
        obj.config.gds_lookup_deferral_timeout_ms = 50
        obj._gds_lookup_should_fallback = (
            lambda req_id: LMCacheConnectorV1Impl._gds_lookup_should_fallback(obj, req_id)
        )
        obj._gds_lookup_note_late_hit = (
            lambda req_id: LMCacheConnectorV1Impl._gds_lookup_note_late_hit(obj, req_id)
        )
        return obj

    def test_gds_full_prefix_not_capped(self):
        adapter = self._adapter(2048)
        req = SimpleNamespace(
            request_id="r1",
            num_tokens=2048,
            num_computed_tokens=0,
        )
        need, async_load = LMCacheConnectorV1Impl.get_num_new_matched_tokens(
            adapter, req, 0
        )
        self.assertEqual(need, 2047)
        self.assertTrue(async_load)
        self.assertEqual(adapter.load_specs["r1"].lmcache_cached_tokens, 2048)

    def test_lookup_pending_returns_none(self):
        adapter = self._adapter(None)
        req = SimpleNamespace(
            request_id="r2",
            num_tokens=2048,
            num_computed_tokens=0,
        )
        need, async_load = LMCacheConnectorV1Impl.get_num_new_matched_tokens(
            adapter, req, 0
        )
        self.assertIsNone(need)
        self.assertTrue(async_load)
        self.assertNotIn("r2", adapter.load_specs)

    def test_lookup_deferral_timeout_schedules_miss(self):
        adapter = self._adapter(None)
        adapter._gds_lookup_defer_started["r2"] = __import__("time").monotonic() - 1.0
        req = SimpleNamespace(
            request_id="r2",
            num_tokens=2048,
            num_computed_tokens=0,
        )
        need, async_load = LMCacheConnectorV1Impl.get_num_new_matched_tokens(
            adapter, req, 0
        )
        self.assertEqual(need, 0)
        self.assertFalse(async_load)
        self.assertEqual(adapter._gds_lookup_fallback_count, 1)
        self.assertNotIn("r2", adapter.load_specs)

    def test_lookup_late_hit_counted_after_fallback(self):
        adapter = self._adapter(2048)
        adapter._gds_lookup_fallback_ids.add("r1")
        req = SimpleNamespace(
            request_id="r1",
            num_tokens=2048,
            num_computed_tokens=0,
        )
        LMCacheConnectorV1Impl.get_num_new_matched_tokens(adapter, req, 0)
        self.assertEqual(adapter._gds_lookup_late_hit_count, 1)
        self.assertNotIn("r1", adapter._gds_lookup_fallback_ids)

    def test_release_load_spec_abort_only_on_abort_reason(self):
        adapter = SimpleNamespace(
            load_specs={"r": LoadSpec(0, 2048, True)},
            _load_spec_abort=0,
            _load_spec_finish=0,
        )
        adapter._inc_restore_stat = lambda name, n=1: LMCacheConnectorV1Impl._inc_restore_stat(
            adapter, name, n
        )
        adapter._log_load_spec = lambda *a, **k: None
        LMCacheConnectorV1Impl._release_load_spec(adapter, "r", "finish")
        self.assertEqual(adapter._load_spec_abort, 0)
        adapter.load_specs["s"] = LoadSpec(0, 2048, True)
        LMCacheConnectorV1Impl._release_load_spec(adapter, "s", "abort")
        self.assertEqual(adapter._load_spec_abort, 1)


@unittest.skipUnless(LMCacheConnectorV1Impl is not None, "vllm/torch connector deps missing")
class TestGdsSlotMappingAdapter(unittest.TestCase):
    def _save_adapter(self, io_mode: str):
        captured = {}
        pending = SimpleNamespace(
            keys=[1],
            tot_kv_size=8,
            gpu_block_count=0,
            error=None,
            req_id="r1",
        )

        def issue_store_async(*_a, **kwargs):
            captured["kwargs"] = kwargs
            return pending

        obj = SimpleNamespace(
            lmcache_engine=SimpleNamespace(
                issue_store_async=issue_store_async,
                abandon_store=lambda *_a, **_k: None,
            ),
            kv_io_mode=lambda: io_mode,
            _gpu_block_count_upper_bound=lambda _slot: 1,
            _async_store_over_watermark=lambda *_a, **_k: False,
            _async_store_drops=0,
            _async_t_issue_us=0,
            _pending_bytes=0,
            _pending_saves={},
            _delayed_free_blocks=0,
            _gds_slot_mapping_cuda_fallback=0,
            _async_put_failure=0,
        )
        obj._slot_mapping_for_gds_page_direct = (
            lambda slot: LMCacheConnectorV1Impl._slot_mapping_for_gds_page_direct(
                obj, slot
            )
        )
        obj._copy_slot_mapping_to_device = mock.Mock(side_effect=lambda slot: slot)
        obj._account_pending_add = lambda _pending: None
        obj.captured = captured
        return obj

    def test_gds_async_save_passes_cpu_mapping(self):
        import torch

        adapter = self._save_adapter("gds")
        cpu = torch.arange(8, dtype=torch.int64)
        req = SimpleNamespace(
            req_id="r1", disagg_spec=None, request_configs=None, generation=0
        )
        ok = LMCacheConnectorV1Impl._try_issue_async_save(
            adapter,
            req,
            list(range(8)),
            torch.ones(8, dtype=torch.bool),
            cpu,
            0,
            [],
        )
        self.assertTrue(ok)
        slot = adapter.captured["kwargs"]["slot_mapping"]
        self.assertIs(slot, cpu)
        self.assertEqual(slot.device.type, "cpu")
        adapter._copy_slot_mapping_to_device.assert_not_called()

    def test_dram_async_save_copies_to_device(self):
        import torch

        adapter = self._save_adapter("dram")
        cpu = torch.arange(8, dtype=torch.int64)
        req = SimpleNamespace(
            req_id="r1", disagg_spec=None, request_configs=None, generation=0
        )
        ok = LMCacheConnectorV1Impl._try_issue_async_save(
            adapter,
            req,
            list(range(8)),
            torch.ones(8, dtype=torch.bool),
            cpu,
            0,
            [],
        )
        self.assertTrue(ok)
        adapter._copy_slot_mapping_to_device.assert_called_once_with(cpu)

    def test_gds_helper_returns_cpu_identity(self):
        import torch

        adapter = self._save_adapter("gds")
        cpu = torch.arange(4, dtype=torch.int64)
        out = LMCacheConnectorV1Impl._slot_mapping_for_gds_page_direct(adapter, cpu)
        self.assertIs(out, cpu)

    def _load_adapter(self, io_mode: str, *, layerwise: bool = False):
        import torch

        captured = {}
        cpu = torch.arange(8, dtype=torch.int64)
        req = SimpleNamespace(
            load_spec=LoadSpec(0, 8, True),
            token_ids=list(range(8)),
            slot_mapping=cpu,
            request_configs=None,
            req_id="r1",
            allocated_block_ids=(),
        )
        engine = SimpleNamespace(
            gpu_connector=None,
            retrieve=mock.Mock(
                return_value=torch.ones(8, dtype=torch.bool)
            ),
            retrieve_layer=mock.Mock(return_value=iter([None, None])),
            _last_cancelled_load=None,
        )
        obj = SimpleNamespace(
            kv_caches={"a": torch.zeros(1)},
            current_layer=0,
            lmcache_engine=engine,
            _parent=SimpleNamespace(
                _get_connector_metadata=lambda: LMCacheConnectorMetadata(
                    requests=[req]
                )
            ),
            _async_store_enabled=lambda: False,
            _stats_monitor=SimpleNamespace(
                update_interval_vllm_hit_tokens=lambda *_a, **_k: None,
                update_interval_prompt_tokens=lambda *_a, **_k: None,
            ),
            _lmcache_chunk_size=256,
            use_layerwise=layerwise,
            enable_blending=False,
            layerwise_retrievers=[],
            kv_io_mode=lambda: io_mode,
            _hbm_prefetch_enabled=lambda: False,
            _gds_slot_mapping_cuda_fallback=0,
            _last_compute_load_sync_ms=0.0,
        )
        obj._copy_slot_mapping_to_device = mock.Mock(side_effect=lambda slot: slot)
        obj._slot_mapping_for_gds_page_direct = (
            lambda slot: LMCacheConnectorV1Impl._slot_mapping_for_gds_page_direct(
                obj, slot
            )
        )

        def start_or_progress(request, tokens, mask, kwargs, generation, block_ids):
            captured["async_kwargs"] = kwargs

        obj._start_or_progress_async_load = start_or_progress
        obj.captured = captured
        obj.cpu_slots = cpu
        obj.engine = engine
        return obj

    def test_start_load_kv_gds_async_keeps_cpu_mapping(self):
        adapter = self._load_adapter("gds")
        LMCacheConnectorV1Impl.start_load_kv(
            adapter, SimpleNamespace(attn_metadata=None)
        )
        slot = adapter.captured["async_kwargs"]["slot_mapping"]
        self.assertEqual(slot.device.type, "cpu")
        adapter._copy_slot_mapping_to_device.assert_not_called()

    def test_start_load_kv_dram_async_copies_to_device(self):
        adapter = self._load_adapter("dram")
        LMCacheConnectorV1Impl.start_load_kv(
            adapter, SimpleNamespace(attn_metadata=None)
        )
        adapter._copy_slot_mapping_to_device.assert_called()
        self.assertIn("async_kwargs", adapter.captured)

    def test_start_load_kv_layerwise_copies_to_device(self):
        adapter = self._load_adapter("gds", layerwise=True)
        LMCacheConnectorV1Impl.start_load_kv(
            adapter, SimpleNamespace(attn_metadata=None)
        )
        adapter._copy_slot_mapping_to_device.assert_called()
        adapter.engine.retrieve_layer.assert_called_once()
        slot = adapter.engine.retrieve_layer.call_args.kwargs["slot_mapping"]
        self.assertEqual(list(slot), list(adapter.cpu_slots[:8]))
        self.assertNotIn("async_kwargs", adapter.captured)

    def test_start_load_kv_non_prefetch_retrieve_uses_gpu_mapping(self):
        adapter = self._load_adapter("gds")
        adapter.kv_io_mode = lambda: "none"
        LMCacheConnectorV1Impl.start_load_kv(
            adapter, SimpleNamespace(attn_metadata=None)
        )
        adapter._copy_slot_mapping_to_device.assert_called()
        adapter.engine.retrieve.assert_called_once()
        slot = adapter.engine.retrieve.call_args.kwargs["slot_mapping"]
        self.assertEqual(list(slot), list(adapter.cpu_slots[:8]))

    def _async_load_obj(self, io_mode: str):
        engine = SimpleNamespace(
            retrieve=mock.Mock(),
            lookup_unpin=mock.Mock(),
            abandon_load=mock.Mock(),
            _last_pending_load=None,
        )
        obj = SimpleNamespace(
            lmcache_engine=engine,
            kv_io_mode=lambda: io_mode,
            _pending_loads={},
            _own_stale_event=0,
            _own_duplicate_generation=0,
            _own_block_mismatch=0,
            _cancelled_recving=set(),
            _prefetch_cancelled=0,
            _prefetch_issue_count=0,
            _prefetch_inflight=0,
            _gds_retrieve_submit_missing=0,
        )
        obj._pending_key = (
            lambda req_id, generation=0: LMCacheConnectorV1Impl._pending_key(
                obj, req_id, generation
            )
        )
        obj._load_req_id = lambda key: LMCacheConnectorV1Impl._load_req_id(obj, key)
        obj._load_generation = (
            lambda key, pending=None: LMCacheConnectorV1Impl._load_generation(
                obj, key, pending
            )
        )
        return obj, engine

    def test_gds_load_without_submit_fails_closed(self):
        import torch

        obj, engine = self._async_load_obj("gds")
        req = SimpleNamespace(req_id="r1")
        LMCacheConnectorV1Impl._start_or_progress_async_load(
            obj,
            req,
            torch.arange(8),
            torch.ones(8, dtype=torch.bool),
            {"slot_mapping": torch.arange(8)},
            0,
            (),
        )
        engine.retrieve.assert_not_called()
        self.assertEqual(obj._gds_retrieve_submit_missing, 1)
        self.assertIn("r1", obj._cancelled_recving)
        engine.lookup_unpin.assert_called_once_with("r1")

    def test_dram_load_without_submit_falls_back_to_retrieve(self):
        import torch

        obj, engine = self._async_load_obj("dram")
        req = SimpleNamespace(req_id="r1")
        LMCacheConnectorV1Impl._start_or_progress_async_load(
            obj,
            req,
            torch.arange(8),
            torch.ones(8, dtype=torch.bool),
            {"slot_mapping": torch.arange(8)},
            0,
            (),
        )
        engine.retrieve.assert_called_once()
        self.assertEqual(obj._gds_retrieve_submit_missing, 0)


if __name__ == "__main__":
    unittest.main()
