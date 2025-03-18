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

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <pika/runtime.hpp>

#include <dlaf/tune.h>

namespace dlaf::eigensolver::internal {

inline size_t get_red2band_panel_worker_minwork() noexcept {
  return 1;
}

inline size_t get_red2band_panel_num_workers() noexcept {
  // Note: precautionarily we leave at least 1 thread "free" to do other stuff (if possible)
  const std::size_t available_workers = pika::resource::get_thread_pool("default").get_os_thread_count();
  const std::size_t min_workers = 1;
  const auto max_workers = std::max(min_workers, available_workers - 1);

  const std::size_t nworkers = getTuneParameters().red2band_panel_num_threads;
  return std::clamp(nworkers, min_workers, max_workers);
}

}
