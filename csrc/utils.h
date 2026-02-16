// Copyright (c) 2026 Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

class cuda_error : public std::runtime_error {
public:
    cuda_error(const cudaError_t err, const std::string& arg) :
            std::runtime_error(arg), code(err) {};

    cudaError_t code;
};

inline void cuda_throw_on_error(cudaError_t status, const char* statement, const char* file, int line) {
    if (status != cudaSuccess) {
        std::string msg = std::string("Cuda Error in ") + file + ":" + std::to_string(line) + " (" + std::string(statement) + "): " + cudaGetErrorName(status) + ": ";
        msg += cudaGetErrorString(status);
        throw cuda_error(status, msg);
    }
}

#define CUDA_CHECK(status) cuda_throw_on_error(status, #status, __FILE__, __LINE__)
