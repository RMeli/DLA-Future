//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <blas.hh>
#include <blas/util.hh>
#include <mpi.h>

#include <pika/init.hpp>

#include <dlaf/factorization/cholesky.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf_c/desc.h>
#include <dlaf_c/grid.h>
#include <dlaf_c/utils.h>

#include "../blacs.h"
#include "../grid.h"
#include "../utils.h"

template <typename T>
int cholesky(int dlaf_context, char uplo, T* a, DLAF_descriptor dlaf_desca) {
  using MatrixMirror = dlaf::matrix::MatrixMirror<T, dlaf::Device::Default, dlaf::Device::CPU>;

  DLAF_ASSERT(dlaf_desca.i == 0, dlaf_desca.i);
  DLAF_ASSERT(dlaf_desca.j == 0, dlaf_desca.j);

  pika::resume();

  auto communicator_grid = dlaf_grids.at(dlaf_context);

  auto [distribution, layout] = distribution_and_layout(dlaf_desca, communicator_grid);

  dlaf::matrix::Matrix<T, dlaf::Device::CPU> matrix_host(std::move(distribution), layout, a);

  {
    MatrixMirror matrix(matrix_host);

    dlaf::factorization::cholesky<dlaf::Backend::Default, dlaf::Device::Default, T>(
        communicator_grid, blas::char2uplo(uplo), matrix.get());
  }  // Destroy mirror

  matrix_host.waitLocalTiles();

  pika::suspend();

  return 0;
}

#ifdef DLAF_WITH_SCALAPACK

template <typename T>
void pxpotrf(char uplo, int n, T* a, int ia, int ja, int* desca, int& info) {
  DLAF_ASSERT(desca[0] == 1, desca[0]);
  DLAF_ASSERT(ia == 1, ia);
  DLAF_ASSERT(ja == 1, ja);

  auto dlaf_desca = make_dlaf_descriptor(n, n, ia, ja, desca);

  auto _info = cholesky(desca[1], uplo, a, dlaf_desca);
  info = _info;
}

#endif
