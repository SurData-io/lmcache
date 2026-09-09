// SPDX-License-Identifier: Apache-2.0
#include "mem_kernels.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace {

__global__ void residual_copy_kernel(uint8_t* dst, const uint8_t* src, size_t n) {
  size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;
  for (; i < n; i += stride) {
    dst[i] = src[i];
  }
}

}  // namespace

void residual_copy_bytes(uintptr_t dst_ptr, uintptr_t src_ptr, size_t nbytes,
                         uintptr_t stream_ptr) {
  if (nbytes == 0) {
    return;
  }
  TORCH_CHECK(dst_ptr != 0 && src_ptr != 0,
              "residual_copy_bytes requires non-null pointers");
  cudaStream_t stream = stream_ptr
                            ? reinterpret_cast<cudaStream_t>(stream_ptr)
                            : at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  int blocks = static_cast<int>((nbytes + threads - 1) / threads);
  if (blocks < 1) {
    blocks = 1;
  }
  if (blocks > 4096) {
    blocks = 4096;
  }
  residual_copy_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<uint8_t*>(dst_ptr),
      reinterpret_cast<const uint8_t*>(src_ptr), nbytes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
