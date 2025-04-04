//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#ifdef DLAF_WITH_GPU

#include <stdio.h>
#include <whip.hpp>

#ifdef DLAF_ASSERT_HEAVY_ENABLE

#define DLAF_GPU_ASSERT_HEAVY(expr)                              \
  ::dlaf::gpu::internal::gpuAssert(expr, [] __device__() {       \
    printf("GPU assertion failed: %s:%d\n", __FILE__, __LINE__); \
  })

#else
#define DLAF_GPU_ASSERT_HEAVY(expr)
#endif

namespace dlaf::gpu::internal {

template <class PrintFunc>
__device__ void gpuAssert(bool expr, PrintFunc&& print_func) {
  if (!expr) {
    print_func();
#ifdef DLAF_WITH_CUDA
    __trap();
#else
    abort();
#endif
  }
}

}

#endif
