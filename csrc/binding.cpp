// Copyright (c) 2026 Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <utility>
#include "manager.h"

namespace nb = nanobind;


void do_bench(std::string target_file, const nb::object& kernel_generator, const nb::object& test_generator, const nb::tuple& test_args, int repeats, std::uintptr_t stream, bool unlink) {
    BenchmarkManager mgr(std::move(target_file), unlink);
    auto [args, expected] = mgr.setup_benchmark(nb::cast<nb::callable>(test_generator), test_args, repeats);
    mgr.do_bench_py(nb::cast<nb::callable>(kernel_generator), args, expected, reinterpret_cast<cudaStream_t>(stream));
}


NB_MODULE(_pygpubench, m) {
    m.def("do_bench", do_bench);
}
