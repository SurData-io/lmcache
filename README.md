# SurData LMCache

Apache-2.0 fork of [LMCache](https://github.com/LMCache/LMCache) **v0.5.3**.
This repository publishes the **engine** (`lmcache/`) and the **vLLM adapter**
(`lmcache/integration/vllm/`).

It is **not** the upstream [LMCache/LMCache](https://github.com/LMCache/LMCache)
project. Install the official package from PyPI when you want unmodified LMCache.

## What is included

- LMCache KV-cache engine (CPU / disk / remote backends, GDS page-direct path)
- vLLM v1 connector (`LMCacheConnectorV1`)
- Native C++/CUDA extensions from the v0.5.3 tree
- Tests for the GDS retrieve/store state machine that do not require a private SDK

## What is not included

The SurData storage **SDK**, kernel module, and meta services stay closed-source.
This tree does not ship `src/surdata_lmcache`, `libsurdata_sdk`, or
`csrc/storage_backends/surdata/`.

A remote connector plugin can still be loaded at runtime if you have a separate
build of that SDK.

## Install from source

```bash
# CUDA extensions need torch first
pip install torch
pip install -e . --no-build-isolation

# Source-only (no native extensions)
NO_NATIVE_EXT=1 pip install -e .
```

See upstream docs for serving recipes: <https://docs.lmcache.ai/>

## License

Apache License 2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE), and
[NOTICE.surdata](NOTICE.surdata).
