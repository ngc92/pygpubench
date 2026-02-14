// Copyright (c) 2026 Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <stdexcept>
#include <cuda/cmath>

#include "utils.h"

__global__ void clear_cache(uint4* dummy_memory, const int size) {
    const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= size) return;

    // do some random useless arithmetic to keep the GPU cores churning
    unsigned val = (i * i) % 53145u + blockIdx.x;
    for (int j = 0; j < 10; j++) {
        val = (val * 735326u + 1345143u) % 34622u;
    }
    uint4 res;
    res.x = val;
    res.y = val * val;
    res.z = val + val % 256;
    res.w = val ^ (val << 1);
    dummy_memory[i] = res;
}

void clear_cache(void* dummy_memory, int size, cudaStream_t stream) {
    // make sure there's no sneaky cache persistency setting that defeats our spam-writing
    CUDA_CHECK(cudaCtxResetPersistingL2Cache());

    int nelem = size / sizeof(uint4);
    // write a large amount of memory to ensure all cache lines are cleared
    int threads = 256;
    int blocks = cuda::ceil_div(nelem, threads);
    clear_cache<<<blocks, threads, 0, stream>>>(static_cast<uint4*>(dummy_memory), nelem);
    CUDA_CHECK(cudaGetLastError());
}
