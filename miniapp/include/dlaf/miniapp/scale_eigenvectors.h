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

#include <blas.hh>

#include <pika/runtime.hpp>

#include <dlaf/blas/scal.h>
#include <dlaf/common/index2d.h>
#include <dlaf/common/range2d.h>
#include <dlaf/common/single_threaded_blas.h>
#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/types.h>

namespace dlaf::miniapp {
using common::Ordering;
using matrix::Tile;
using matrix::internal::MatrixRef;

template <typename T>
void scaleTile(const Tile<const BaseType<T>, Device::CPU>& lambda, const Tile<T, Device::CPU>& tile) {
  common::internal::SingleThreadedBlasScope single;
  for (SizeType j = 0; j < tile.size().cols(); ++j) {
    blas::scal(tile.size().rows(), lambda({j, 0}), tile.ptr({0, j}), 1);
  }
}

template <typename T>
void scaleEigenvectors(MatrixRef<const BaseType<T>, Device::CPU>& evalues,
                       MatrixRef<const T, Device::CPU>& evectors, Matrix<T, Device::CPU>& result) {
  using pika::execution::thread_priority;
  matrix::internal::copy(evectors, result);

  const auto& dist = result.distribution();

  for (const auto& ij : iterate_range2d(dist.localNrTiles())) {
    SizeType j = dist.template globalTileFromLocalTile<Coord::Col>(ij.col());
    pika::execution::experimental::start_detached(
        dlaf::internal::whenAllLift(evalues.read(GlobalTileIndex{j, 0}), result.readwrite(ij)) |
        dlaf::internal::transform(dlaf::internal::Policy<Backend::MC>(thread_priority::normal),
                                  scaleTile<T>));
  }
}

template <typename T>
void scaleEigenvectors(Matrix<const BaseType<T>, Device::CPU>& evalues,
                       Matrix<const T, Device::CPU>& evectors, Matrix<T, Device::CPU>& result) {
  MatrixRef<const BaseType<T>, Device::CPU> evalues_ref(evalues);
  MatrixRef<const T, Device::CPU> evectors_ref(evectors);

  scaleEigenvectors<T>(evalues_ref, evectors_ref, result);
}
}
