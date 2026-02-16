// Copyright (c) 2026 Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <stdexcept>
#include <vector_types.h>
#include <cuda/atomic>
#include <cuda/cmath>

#include "utils.h"

// bitwise exact match
__global__ void check_exact_match_kernel(unsigned* result, const uint4* expected, const uint4* received, std::size_t size) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    uint4 a = expected[idx];
    cuda::atomic_ref<unsigned, cuda::thread_scope_device> res(*result);

#if __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif

    uint4 b = received[idx];
    if (a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w) {
        ++res;
    }
}

template<typename Float>
__global__ void check_approx_match_kernel(unsigned* result, const Float* expected, const Float* received, float r_tol, float a_tol, std::size_t size) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    cuda::atomic_ref<unsigned, cuda::thread_scope_device> res(*result);
    float a = static_cast<float>(expected[idx]);

    // Nan is expected is wildcard for arbitrary results
    if (isnan(a))
        return;

#if __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif

    float b = static_cast<float>(received[idx]);
    if (!isfinite(a)) {
        if (a != b) {
            ++res;
            return;
        }
    }
    if (!isfinite(b)) {
        ++res;
        return;
    }

    // both are zero
    if (a == 0 && b == 0) return;

    // ok, a and b are finite, and at least one is not zero
    float diff = fabsf(a - b);
    if (diff > r_tol * fabsf(a) && diff > a_tol) {
        ++res;
    }
}


void check_exact_match_launcher(unsigned* result, const std::byte* expected, const std::byte* received, std::size_t nbytes, cudaStream_t stream) {
    if (nbytes % sizeof(uint4) != 0) {
        throw std::runtime_error("Expected number of bytes to be divisible by 16");
    }
    if (reinterpret_cast<std::uintptr_t>(expected) % 16 != 0) {
        throw std::runtime_error("Expected pointer must be aligned to 16");
    }
    if (reinterpret_cast<std::uintptr_t>(received) % 16 != 0) {
        throw std::runtime_error("Received pointer must be aligned to 16");
    }

    int threads = 256;
    int blocks = cuda::ceil_div(nbytes / 16, threads);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config;
    config.attrs = attribute;
    config.numAttrs = 1;
    config.blockDim = dim3(threads, 1, 1);
    config.gridDim = dim3(blocks, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    CUDA_CHECK(cudaLaunchKernelEx(&config, check_exact_match_kernel, result, reinterpret_cast<const uint4*>(expected), reinterpret_cast<const uint4*>(received), nbytes / sizeof(uint4)));
}

template<typename Float>
void check_check_approx_match_launcher_tpl(unsigned* result, const Float* expected, const Float* received, float r_tol, float a_tol, std::size_t size, cudaStream_t stream) {
    if ( !(a_tol >= 0) ) throw std::runtime_error("Absolute tolerance must be non-negative");
    if ( !(r_tol >= 0) ) throw std::runtime_error("Relative tolerance must be non-negative");
    int threads = 256;
    int blocks = cuda::ceil_div(size, threads);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config;
    config.attrs = attribute;
    config.numAttrs = 1;
    config.blockDim = dim3(threads, 1, 1);
    config.gridDim = dim3(blocks, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    CUDA_CHECK(cudaLaunchKernelEx(&config, check_approx_match_kernel<Float>, result, expected, received, r_tol, a_tol, size));
}

void check_check_approx_match_launcher(unsigned* result, const float* expected, const float* received, float r_tol, float a_tol, std::size_t size, cudaStream_t stream) {
    check_check_approx_match_launcher_tpl<float>(result, expected, received, r_tol, a_tol, size, stream);
}

void check_check_approx_match_launcher(unsigned* result, const nv_bfloat16* expected, const nv_bfloat16* received, float r_tol, float a_tol, std::size_t size, cudaStream_t stream) {
    check_check_approx_match_launcher_tpl<nv_bfloat16>(result, expected, received, r_tol, a_tol, size, stream);
}

void check_check_approx_match_launcher(unsigned* result, const half* expected, const half* received, float r_tol, float a_tol, std::size_t size, cudaStream_t stream) {
    check_check_approx_match_launcher_tpl<half>(result, expected, received, r_tol, a_tol, size, stream);
}

__global__ void canaries_kernel(uint4* data, int size, unsigned seed) {
    unsigned tg = threadIdx.x / 8;
    unsigned idx = (blockIdx.x * blockDim.x + threadIdx.x) / 8;
    unsigned rtg = tg ^ (seed & 0x1f); // "random" index  [0, 31]
    unsigned lb = seed >> 5u;
    // we pick 1 out of every 256 cache lines
    unsigned cache_line = idx * 256 + (rtg * 48 + lb & 0x7);
    unsigned addr = cache_line * 128 / sizeof(int4) + threadIdx.x % 8;
    if (addr < size) {
        uint4 load = data[addr];
        // self-inverse transformation
        load.x = load.x ^ seed;
        load.y = ~load.y;
        load.z = load.z ^ seed;
        load.w = ~load.w;
        data[addr] = load;
    }
}

void canaries(void* data, size_t size, unsigned seed, cudaStream_t stream) {
    int num_sectors = size / (128 * 256);
    int block_size  = 256;
    int num_blocks  = cuda::ceil_div(num_sectors, block_size / 8);
    canaries_kernel<<<num_blocks, 256, 0, stream>>>(reinterpret_cast<uint4*>(data), size / sizeof(int4), seed);
}