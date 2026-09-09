# SPDX-License-Identifier: Apache-2.0
"""CUDA ops backend: bulk-bind the compiled ``lmcache.c_ops`` extension.

:class:`CudaDeviceOps` calls :meth:`bind_native` in :meth:`ensure_native`
to layer the compiled CUDA extension on top of the torch baseline.  If the
extension is missing, a warning is logged and the instance stays on the
torch fallback (soft-fail, same as XPU).
"""

# Future
from __future__ import annotations

# Standard
import glob
import os
from importlib.machinery import ExtensionFileLoader
from importlib.util import module_from_spec, spec_from_loader
from typing import ClassVar

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.device_ops import DeviceOps

logger = init_logger(__name__)


def _load_c_ops_extension():
    """Load ``c_ops*.so`` without ``import lmcache.c_ops``.

    ``lmcache.__init__`` calls ``ensure_native`` while the package is still
    initializing, so a submodule import would fail with a circular
    ImportError and leave CUDA ops on the torch baseline.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    matches = sorted(glob.glob(os.path.join(root, "c_ops*.so")))
    if not matches:
        raise ImportError("c_ops*.so missing under %s" % root)
    so_path = matches[0]
    loader = ExtensionFileLoader("c_ops", so_path)
    spec = spec_from_loader("c_ops", loader)
    if spec is None or spec.loader is None:
        raise ImportError("cannot make import spec for %s" % so_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CudaDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "cuda"

    def ensure_native(self) -> None:
        if self._native_bound:
            return
        self._native_bound = True  # set early to prevent repeated attempts
        try:
            native = _load_c_ops_extension()
        except Exception as exc:
            logger.warning(
                "lmcache.c_ops compiled extension not found; "
                "CudaDeviceOps stays on the torch baseline for all ops. (%s)",
                exc,
            )
            return
        self.bind_native(native)
