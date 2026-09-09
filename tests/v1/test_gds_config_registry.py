# SPDX-License-Identifier: Apache-2.0
"""Registered GDS / SurData knobs must load from YAML extra_config."""

from __future__ import annotations

import unittest

try:
    import yaml  # noqa: F401
    from lmcache.v1.config import LMCacheEngineConfig
except ImportError:
    yaml = None
    LMCacheEngineConfig = None  # type: ignore


@unittest.skipUnless(LMCacheEngineConfig is not None, "lmcache config deps missing")
class TestGdsConfigRegistry(unittest.TestCase):
    def test_registered_gds_fields(self):
        cfg = LMCacheEngineConfig.from_legacy(backend="cpu")
        for name in (
            "kv_io_mode",
            "kv_gds_inflight",
            "kv_gds_staging_bytes",
            "kv_gds_get_workers",
            "kv_gds_get_queue_depth",
            "kv_gds_watchdog_timeout_ms",
            "hbm_prefetch_max_inflight",
            "max_tokens_per_load",
            "kv_gds_retrieve_slots",
            "kv_gds_store_slots",
            "kv_gds_dram_fallback_bytes",
            "kv_gds_staging_arena_bytes",
            "kv_gds_staging_max_waiters",
            "kv_gds_staging_wait_timeout_ms",
            "kv_gds_store_overload_policy",
            "kv_gds_direct_store_max_inflight",
            "async_store_max_delayed_free_ratio",
            "gds_lookup_deferral_timeout_ms",
            "gds_skip_exists_empty_ns",
            "gds_empty_ns_stamp",
        ):
            self.assertTrue(hasattr(cfg, name), name)
        self.assertFalse(hasattr(cfg, "kv_gds_dram_fallback_inflight"))
        self.assertEqual(cfg.kv_gds_store_overload_policy, "immediate_drop")
        self.assertEqual(cfg.async_store_max_delayed_free_ratio, 0.25)
        self.assertEqual(cfg.gds_lookup_deferral_timeout_ms, 50)
        self.assertFalse(cfg.gds_skip_exists_empty_ns)
        self.assertEqual(cfg.gds_empty_ns_stamp, "")

    def test_extra_config_overrides(self):
        cfg = LMCacheEngineConfig.from_legacy(backend="cpu")
        cfg.extra_config = {
            "kv.io_mode": "gds",
            "kv.gds_inflight": 32,
            "kv.gds_staging_bytes": 671088640,
            "kv.gds_get_workers": 8,
            "kv.gds_get_queue_depth": 32,
            "kv.gds_watchdog_timeout_ms": 0,
        }
        cfg.max_local_cpu_size = 1.0
        cfg.validate()
        self.assertEqual(cfg.kv_io_mode, "gds")
        self.assertEqual(cfg.kv_gds_inflight, 32)
        self.assertEqual(cfg.kv_gds_get_workers, 8)
        self.assertEqual(cfg.kv_gds_watchdog_timeout_ms, 0)
        self.assertFalse(cfg.local_cpu)

    def test_gds_caps_dram_l1_cpu_size(self):
        cfg = LMCacheEngineConfig.from_legacy(backend="cpu")
        cfg.extra_config = {"kv.io_mode": "gds"}
        cfg.max_local_cpu_size = 20.0
        cfg.validate()
        self.assertEqual(cfg.kv_io_mode, "gds")
        self.assertFalse(cfg.local_cpu)
        self.assertLessEqual(cfg.max_local_cpu_size, 3.5)
        self.assertGreaterEqual(cfg.max_local_cpu_size, 3.0)
        self.assertGreater(cfg.max_local_cpu_size, 0.0)
        self.assertEqual(int(cfg.extra_config.get("surdata.l1_budget_enable", 1)), 0)

    def test_gds_fallback_extra_overrides_scratch_cap(self):
        cfg = LMCacheEngineConfig.from_legacy(backend="cpu")
        cfg.extra_config = {
            "kv.io_mode": "gds",
            "kv.gds_dram_fallback_bytes": 536870912,
            "kv.gds_retrieve_slots": 8,
            "kv.gds_store_slots": 8,
            "kv.gds_direct_store_max_inflight": 4,
            "kv.gds_store_overload_policy": "immediate_d2h_fallback",
        }
        cfg.max_local_cpu_size = 20.0
        cfg.validate()
        self.assertEqual(cfg.kv_gds_dram_fallback_bytes, 536870912)
        self.assertEqual(cfg.kv_gds_direct_store_max_inflight, 4)
        self.assertEqual(cfg.kv_gds_store_overload_policy, "immediate_d2h_fallback")
        self.assertEqual(cfg.kv_gds_retrieve_slots, 8)
        self.assertEqual(cfg.kv_gds_store_slots, 8)
        self.assertFalse(cfg.local_cpu)
        self.assertLessEqual(cfg.max_local_cpu_size, 1.0)
        self.assertGreater(cfg.max_local_cpu_size, 0.75)

    def test_gds_forces_no_prefetch_and_no_token_cap(self):
        cfg = LMCacheEngineConfig.from_legacy(backend="cpu")
        cfg.extra_config = {"kv.io_mode": "gds"}
        cfg.hbm_prefetch_enable = True
        cfg.max_tokens_per_load = 256
        cfg.max_local_cpu_size = 1.0
        cfg.validate()
        self.assertFalse(cfg.hbm_prefetch_enable)
        self.assertEqual(cfg.max_tokens_per_load, 0)

    def test_d2h_rejects_only_on_bytes_or_alloc(self):
        cfg = LMCacheEngineConfig.from_legacy(backend="cpu")
        cfg.extra_config = {
            "kv.io_mode": "gds",
            "kv.gds_dram_fallback_bytes": 0,
        }
        cfg.max_local_cpu_size = 1.0
        cfg.validate()
        self.assertEqual(cfg.kv_gds_dram_fallback_bytes, 0)
        self.assertEqual(cfg.kv_gds_direct_store_max_inflight, 0)

    def test_unknown_overload_policy_rejected(self):
        cfg = LMCacheEngineConfig.from_legacy(backend="cpu")
        cfg.extra_config = {
            "kv.io_mode": "gds",
            "kv.gds_store_overload_policy": "drop_silent",
        }
        cfg.max_local_cpu_size = 1.0
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_skip_exists_extra_config_default_off(self):
        cfg = LMCacheEngineConfig.from_legacy(backend="cpu")
        cfg.extra_config = {
            "kv.io_mode": "gds",
            "kv.gds_skip_exists_empty_ns": True,
            "kv.gds_empty_ns_stamp": "stamp-1",
            "kv.gds_lookup_deferral_timeout_ms": 80,
        }
        cfg.max_local_cpu_size = 1.0
        cfg.validate()
        self.assertTrue(cfg.gds_skip_exists_empty_ns)
        self.assertEqual(cfg.gds_empty_ns_stamp, "stamp-1")
        self.assertEqual(cfg.gds_lookup_deferral_timeout_ms, 80)


if __name__ == "__main__":
    unittest.main()
