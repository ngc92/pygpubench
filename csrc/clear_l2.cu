// Copyright (c) 2026 Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <stdexcept>
#include <cuda/cmath>

#include "utils.h"

__global__ void write_cache_kernel(uint4* dummy_memory, const int size) {
    const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= size) return;
    dummy_memory[i] = make_uint4(0, 0, 0, 0);
}

__global__ void discard_cache_kernel(uint4* dummy_memory, const int size) {
    const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= size) return;

    asm volatile(
           "discard.global.L2 [%0], 128;"
           :
           :"l"(dummy_memory + i)
           :"memory");
}

void clear_cache(void* dummy_memory, int size, bool discard, cudaStream_t stream) {
    // make sure there's no sneaky cache persistency setting that defeats our spam-writing
    CUDA_CHECK(cudaCtxResetPersistingL2Cache());

    int nelem = size / sizeof(uint4);
    // write a large amount of memory to ensure all cache lines are cleared
    int threads = 256;
    int blocks = cuda::ceil_div(nelem, threads);
    write_cache_kernel<<<blocks, threads, 0, stream>>>(static_cast<uint4*>(dummy_memory), nelem);
    CUDA_CHECK(cudaGetLastError());
    if (discard) {
        discard_cache_kernel<<<blocks, threads, 0, stream>>>(static_cast<uint4*>(dummy_memory), nelem);
        CUDA_CHECK(cudaGetLastError());
    }
}
