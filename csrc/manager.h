// Copyright (c) 2026 Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef PYGPUBENCH_SRC_MANAGER_H
#define PYGPUBENCH_SRC_MANAGER_H

#include <functional>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

class BenchmarkManager {
public:
    BenchmarkManager(std::string result_file);
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

    std::ofstream mOutputFile;
};

#endif //PYGPUBENCH_SRC_MANAGER_H
