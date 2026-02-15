// Copyright (c) 2026 Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <stdexcept>
#include <cuda/cmath>

#include "utils.h"

__global__ void clear_cache_kernel(uint4* dummy_memory, const int size) {
    const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= size) return;

    // write a L2 cache line, then immediately discard it so
    // it is not written back to main memory.
    asm volatile(
           "st.weak.global.wt.v4.u32 [%0], {0, 0, 0, 0};"
           "discard.global.L2 [%0], 128;"
           :
           :"l"(dummy_memory + i)
           :"memory");
}

void clear_cache(void* dummy_memory, int size, cudaStream_t stream) {
    // make sure there's no sneaky cache persistency setting that defeats our spam-writing
    CUDA_CHECK(cudaCtxResetPersistingL2Cache());

    int nelem = size / sizeof(uint4);
    // write a large amount of memory to ensure all cache lines are cleared
    int threads = 256;
    int blocks = cuda::ceil_div(nelem, threads);
    clear_cache_kernel<<<blocks, threads, 0, stream>>>(static_cast<uint4*>(dummy_memory), nelem);
    CUDA_CHECK(cudaGetLastError());
}
