// Copyright (c) 2026 Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef PYGPUBENCH_SRC_MANAGER_H
#define PYGPUBENCH_SRC_MANAGER_H

#include <functional>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>
#include <optional>
#include <nanobind/nanobind.h>
#include "nanobind/ndarray.h"

namespace nb = nanobind;

using nb_cuda_array = nb::ndarray<nb::c_contig, nb::device::cuda>;

class BenchmarkManager {
public:
    BenchmarkManager(std::string result_file, bool unlink);
    ~BenchmarkManager();
    std::pair<std::vector<nb::tuple>, std::vector<nb::tuple>> setup_benchmark(const nb::callable& generate_test_case, const nb::tuple& args, int repeats);
    void do_bench_py(const nb::callable& kernel_generator, const std::vector<nb::tuple>& args, const std::vector<nb::tuple>& expected, cudaStream_t stream);
private:
    double mWarmupSeconds = 1.0;
    double mBenchmarkSeconds = 1.0;

    std::vector<cudaEvent_t> mStartEvents;
    std::vector<cudaEvent_t> mEndEvents;

    std::chrono::high_resolution_clock::time_point mCPUStart;

    int* mDeviceDummyMemory;
    int mL2CacheSize;
    unsigned* mDeviceErrorCounter;
    bool mNVTXEnabled;

    std::ofstream mOutputFile;

    struct Expected {
        enum EMode {
            ExactMatch,
            ApproxMatch
        } Mode;
        nb_cuda_array Value;
        float ATol;
        float RTol;
    };

    struct ShadowArgument {
        nb_cuda_array Original;
        void* Shadow = nullptr;
        unsigned Seed = -1;
        ShadowArgument(nb_cuda_array original, void* shadow, unsigned seed);
        ~ShadowArgument();
        ShadowArgument(ShadowArgument&& other) noexcept;
        ShadowArgument& operator=(ShadowArgument&& other) noexcept;
    };

    using ShadowArgumentList = std::vector<std::optional<ShadowArgument>>;

    static ShadowArgumentList make_shadow_args(const nb::tuple& args, cudaStream_t stream);

    void nvtx_push(const char* name);
    void nvtx_pop();

    void validate_result(Expected& expected, const nb_cuda_array& result, cudaStream_t stream);
};

#endif //PYGPUBENCH_SRC_MANAGER_H
