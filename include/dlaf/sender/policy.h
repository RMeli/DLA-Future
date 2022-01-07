//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2020-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <hpx/local/execution.hpp>

#include <type_traits>
#include <utility>

namespace dlaf {
namespace internal {
/// A policy class for use as a tag for dispatching algorithms to a particular
/// backend.
template <Backend B>
class Policy {
private:
  const hpx::threads::thread_priority priority_ = hpx::threads::thread_priority::normal;

public:
  Policy() = default;
  explicit Policy(hpx::threads::thread_priority priority) : priority_(priority) {}
  Policy(Policy&&) = default;
  Policy(Policy const&) = default;
  Policy& operator=(Policy&&) = default;
  Policy& operator=(Policy const&) = default;

  hpx::threads::thread_priority priority() const noexcept {
    return priority_;
  }
};
}
}