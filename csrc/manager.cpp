// Copyright (c) 2026 Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "manager.h"
#include "utils.h"
#include "check.h"
#include <chrono>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "nanobind/ndarray.h"
using nb_cuda_array = nb::ndarray<nb::c_contig, nb::device::cuda>;

void clear_cache(void* dummy_memory, int size, cudaStream_t stream);

void check_check_approx_match_dispatch(unsigned* result, const nb_cuda_array& expected, const nb_cuda_array& received, float r_tol, float a_tol, std::size_t n_bytes, cudaStream_t stream) {
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};
    nb::dlpack::dtype fp16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Float), 16, 1};
    nb::dlpack::dtype fp32_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Float), 32, 1};
    if (expected.dtype() == bf16_dt) {
        check_check_approx_match_launcher(result, static_cast<const nv_bfloat16*>(expected.data()), static_cast<const nv_bfloat16*>(received.data()), r_tol, a_tol, n_bytes / 2, stream);
    } else if (expected.dtype() == fp16_dt) {
        check_check_approx_match_launcher(result, static_cast<const half*>(expected.data()), static_cast<const half*>(received.data()), r_tol, a_tol, n_bytes / 2, stream);
    } else if (expected.dtype() == fp32_dt) {
        check_check_approx_match_launcher(result, static_cast<const float*>(expected.data()), static_cast<const float*>(received.data()), r_tol, a_tol, n_bytes / 4, stream);
    } else {
        throw std::runtime_error("Unsupported dtype for check_approx_match");
    }
}

BenchmarkManager::BenchmarkManager(std::string result_file) {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&mL2CacheSize, cudaDevAttrL2CacheSize, device));
    CUDA_CHECK(cudaMalloc(&mDeviceDummyMemory, 2 * mL2CacheSize));
    CUDA_CHECK(cudaMalloc(&mDeviceErrorCounter, sizeof(unsigned)));
    mOutputFile.open(result_file);
    //std::remove(result_file.c_str());
}

BenchmarkManager::~BenchmarkManager() {
    cudaFree(mDeviceDummyMemory);
    cudaFree(mDeviceErrorCounter);
    for (auto& event : mStartEvents) cudaEventDestroy(event);
    for (auto& event : mEndEvents) cudaEventDestroy(event);
}

std::pair<std::vector<nb::tuple>, std::vector<nb::tuple>> BenchmarkManager::setup_benchmark(const nb::callable& generate_test_case, const nb::tuple& args, int repeats) {
    // generate one more input to handle warmup
    std::vector<nb::tuple> kernel_args(repeats + 1);
    std::vector<nb::tuple> expected(repeats + 1);
    for (int i = 0; i < repeats + 1; i++) {
        auto gen = nb::cast<nb::tuple>(generate_test_case(args));
        kernel_args[i] = nb::cast<nb::tuple>(gen[0]);
        expected[i] = nb::cast<nb::tuple>(gen[1]);
    }
    return std::make_pair(std::move(kernel_args), std::move(expected));
}

void BenchmarkManager::do_bench_py(const nb::callable& kernel_generator, const std::vector<nb::tuple>& args, const std::vector<nb::tuple>& expected, cudaStream_t stream) {
    if (args.size() < 5) {
        throw std::runtime_error("Not enough test cases to run benchmark");
    }
    if (expected.size() != args.size()) {
        throw std::runtime_error("Expected results and test case list do not have the same length");
    }
    int calls = args.size() - 1;

    // extract relevant infos from args and expected
    // by convention, the first arg is the output tensor.
    // TODO handle multiple outputs
    std::vector<nb_cuda_array> outputs(args.size());
    for (int i = 0; i < args.size(); i++) {
        outputs.at(i) = nb::cast<nb_cuda_array>(args.at(i)[0]);
    }
    struct Expected {
        enum EMode {
            ExactMatch,
            ApproxMatch
        } Mode;
        nb_cuda_array Value;
        float ATol;
        float RTol;
    };
    std::vector<Expected> expected_outputs(args.size());
    for (int i = 0; i < args.size(); i++) {
        const nb::tuple& expected_tuple = expected.at(i);
        nb_cuda_array expected_array = nb::cast<nb_cuda_array>(expected_tuple[0]);
        if (expected.at(i).size() == 1) {
            expected_outputs.at(i) = {Expected::ExactMatch, expected_array, 0.f, 0.f};
        } else {
            float rtol = nb::cast<float>(expected_tuple[1]);
            float atol = nb::cast<float>(expected_tuple[2]);
            expected_outputs.at(i) = {Expected::ApproxMatch, expected_array, atol, rtol};
        }
    }

    // at this point, we call user code (`kernel_generator` is supposed to be the first place the user file is imported)
    // after this, we cannot trust python anymore
    nb::callable kernel = nb::cast<nb::callable>(kernel_generator());

    // ok, first run for compilations etc
    nvtxRangePush("warmup");
    CUDA_CHECK(cudaDeviceSynchronize());
    kernel(args.at(0));
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    // now, run a few more times for warmup; in total aim for 1 second of warmup runs
    std::chrono::high_resolution_clock::time_point cpu_start = std::chrono::high_resolution_clock::now();
    int warmup_run_count = 0;
    double time_estimate;
    nvtxRangePush("timing");
    while (true) {
        // note: we are assuming here that calling the kernel multiple times for the same input is a safe operation
        // this is only potentially problematic for in-place kernels;
        CUDA_CHECK(cudaDeviceSynchronize());
        clear_cache(mDeviceDummyMemory, 2 * mL2CacheSize, stream);
        kernel(args.at(0));
        CUDA_CHECK(cudaDeviceSynchronize());
        std::chrono::high_resolution_clock::time_point cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = cpu_end - cpu_start;
        ++warmup_run_count;
        if (elapsed_seconds.count() > mWarmupSeconds) {
            time_estimate = elapsed_seconds.count() / warmup_run_count;
            break;
        }
    }
    nvtxRangePop();

    // note: this is a very conservative estimate. Timing above was measured with syncs between every kernel.
    const int actual_calls = std::clamp(static_cast<int>(std::ceil(mBenchmarkSeconds / time_estimate)), 1, calls);

    if (actual_calls < 3) {
        throw std::runtime_error("The initial speed test indicated that running times are too slow to generate meaningful benchmark numbers: " + std::to_string(time_estimate));
    }

    mStartEvents.resize(actual_calls);
    mEndEvents.resize(actual_calls);
    for (int i = 0; i < actual_calls; i++) {
        CUDA_CHECK(cudaEventCreate(&mStartEvents.at(i), cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreate(&mEndEvents.at(i), cudaEventDisableTiming));
    }

    CUDA_CHECK(cudaMemsetAsync(mDeviceErrorCounter, 0, sizeof(unsigned), stream));

    // dry run
    nvtxRangePush("dry-run");
    clear_cache(mDeviceDummyMemory, 2 * mL2CacheSize, stream);
    for (int i = 0; i < actual_calls; i++) {
        CUDA_CHECK(cudaEventRecord(mStartEvents.at(i), stream));
        CUDA_CHECK(cudaEventRecord(mEndEvents.at(i), stream));
    }
    nvtxRangePop();
    CUDA_CHECK(cudaDeviceSynchronize());

    nvtxRangePush("benchmark");
    // now do the real runs
    for (int i = 0; i < actual_calls; i++) {
        nvtxRangePush("cc");
        clear_cache(mDeviceDummyMemory, 2 * mL2CacheSize, stream);
        nvtxRangePop();
        CUDA_CHECK(cudaEventRecord(mStartEvents.at(i), stream));
        nvtxRangePush("kernel");
        kernel(args.at(i + 1));
        nvtxRangePop();
        CUDA_CHECK(cudaEventRecord(mEndEvents.at(i), stream));
        // immediately after the kernel, launch the checking code; if there is some unsynced work done on another stream,
        // this increases the chance of detection.
        if (expected_outputs.at(i + 1).Mode == Expected::ExactMatch) {
            check_exact_match_launcher(
                mDeviceErrorCounter,
                static_cast<std::byte*>(expected_outputs.at(i + 1).Value.data()),
                static_cast<std::byte*>(outputs.at(i + 1).data()),
                outputs.at(i + 1).nbytes(), stream);
        } else {
            check_check_approx_match_dispatch(
                mDeviceErrorCounter,
                expected_outputs.at(i + 1).Value, outputs.at(i + 1),
                expected_outputs.at(i + 1).RTol, expected_outputs.at(i + 1).ATol, outputs.at(i + 1).nbytes(), stream);
        }
    }
    nvtxRangePop();

    cudaEventSynchronize(mEndEvents.back());
    int error_count;
    CUDA_CHECK(cudaMemcpy(&error_count, mDeviceErrorCounter, sizeof(unsigned), cudaMemcpyDeviceToHost));
    if (error_count > 0) {
        //throw std::runtime_error("Detected " + std::to_string(error_count) + " errors in the benchmark");
    }

    // extract run times and write to output file
    for (int i = 0; i < actual_calls; i++) {
        float duration;
        //CUDA_CHECK(cudaEventElapsedTime(&duration, mStartEvents.at(i), mEndEvents.at(i)));
        //mOutputFile << (duration * 1000) << "\n";
    }
    mOutputFile.flush();

    // cleanup events
    for (auto& event : mStartEvents) CUDA_CHECK(cudaEventDestroy(event));
    for (auto& event : mEndEvents) CUDA_CHECK(cudaEventDestroy(event));
    mStartEvents.clear();
    mEndEvents.clear();
}
