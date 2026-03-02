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
#include <mpi.h>

#include <pika/init.hpp>

#include <dlaf/blas/enum_parse.h>
#include <dlaf/common/assert.h>
#include <dlaf/eigensolver/eigensolver.h>
#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/create_matrix.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/types.h>
#include <dlaf_c/desc.h>
#include <dlaf_c/grid.h>

#include "../utils.h"

template <typename T>
int hermitian_eigensolver(const int dlaf_context, const char uplo, T* a,
                          const DLAF_descriptor dlaf_desca, dlaf::BaseType<T>* w, T* z,
                          const DLAF_descriptor dlaf_descz, SizeType eigenvalues_index_begin,
                          SizeType eigenvalues_index_end) {
  using MatrixHost = dlaf::matrix::Matrix<T, dlaf::Device::CPU>;
  using MatrixDevice = dlaf::matrix::Matrix<T, dlaf::Device::Default>;
  using MatrixBaseMirror =
      dlaf::matrix::MatrixMirror<dlaf::BaseType<T>, dlaf::Device::Default, dlaf::Device::CPU>;

  DLAF_ASSERT(dlaf_desca.i == 0, dlaf_desca.i);
  DLAF_ASSERT(dlaf_desca.j == 0, dlaf_desca.j);
  DLAF_ASSERT(dlaf_descz.i == 0, dlaf_descz.i);
  DLAF_ASSERT(dlaf_descz.j == 0, dlaf_descz.j);

  pika::resume();

  auto& communicator_grid = grid_from_context(dlaf_context);

  auto layout_a = make_layout(dlaf_desca, communicator_grid);
  auto layout_z = make_layout(dlaf_descz, communicator_grid);

  MatrixHost matrix_host(layout_a, a);
  MatrixHost eigenvectors_host(layout_z, z);
  auto eigenvalues_host = dlaf::matrix::create_matrix_from_col_major<dlaf::Device::CPU>(
      {dlaf_descz.m, 1}, {dlaf_descz.mb, 1}, std::max(dlaf_descz.m, 1), w);

  const auto& dist_host = matrix_host.distribution();
  const dlaf::GlobalElementSize matrix_size = matrix_host.size();

  const auto opt_device_block_size = get_internal_block_size();
  const bool needs_redistribution =
      opt_device_block_size.has_value() && !matrix_size.isEmpty() &&
      (dist_host.block_size().rows() != *opt_device_block_size ||
       dist_host.block_size().cols() != *opt_device_block_size);

  if (needs_redistribution) {
    const SizeType device_block_size = *opt_device_block_size;
    const dlaf::TileElementSize tile_size(std::min(device_block_size, matrix_size.rows()),
                                          std::min(device_block_size, matrix_size.cols()));

    dlaf::matrix::Distribution dist_device(matrix_size, tile_size, dist_host.grid_size(),
                                           dist_host.rank_index(), dist_host.source_rank_index());
    MatrixDevice matrix_device(dist_device);
    MatrixDevice eigenvectors_device(dist_device);

    dlaf::matrix::copy(matrix_host, matrix_device, communicator_grid);

    // Create eigenvalues on device with matching block size (local, non-distributed)
    using MatrixDeviceBase = dlaf::matrix::Matrix<dlaf::BaseType<T>, dlaf::Device::Default>;
    const dlaf::GlobalElementSize eigenvalues_size(matrix_size.rows(), 1);
    const dlaf::TileElementSize eigenvalues_tile_size(tile_size.rows(), 1);
    MatrixDeviceBase eigenvalues_device(eigenvalues_size, eigenvalues_tile_size);

    dlaf::hermitian_eigensolver<dlaf::Backend::Default, dlaf::Device::Default, T>(
        communicator_grid, dlaf::internal::char2uplo(uplo), matrix_device, eigenvalues_device,
        eigenvectors_device, eigenvalues_index_begin, eigenvalues_index_end);

    dlaf::matrix::copy(eigenvectors_device, eigenvectors_host, communicator_grid);
    dlaf::matrix::copy(eigenvalues_device, eigenvalues_host);

    eigenvalues_host.waitLocalTiles();
    eigenvectors_host.waitLocalTiles();
  }
  else { // No redistribution needed, use MatrixMirror to avoid extra copy
    {
      using MatrixMirror = dlaf::matrix::MatrixMirror<T, dlaf::Device::Default, dlaf::Device::CPU>;
      
      MatrixBaseMirror eigenvalues(eigenvalues_host);
      MatrixMirror matrix(matrix_host);
      MatrixMirror eigenvectors(eigenvectors_host);

      dlaf::hermitian_eigensolver<dlaf::Backend::Default, dlaf::Device::Default, T>(
          communicator_grid, dlaf::internal::char2uplo(uplo), matrix.get(), eigenvalues.get(),
          eigenvectors.get(), eigenvalues_index_begin, eigenvalues_index_end);
    }

    eigenvalues_host.waitLocalTiles();
    eigenvectors_host.waitLocalTiles();
  }

  pika::suspend();
  return 0;
}

#ifdef DLAF_WITH_SCALAPACK

template <typename T>
void pxheevd(const char uplo, const int m, T* a, const int ia, const int ja, const int desca[9],
             dlaf::BaseType<T>* w, T* z, const int iz, int jz, const int descz[9],
             const SizeType eigenvalues_index_begin, const SizeType eigenvalues_index_end, int& info) {
  DLAF_ASSERT(desca[0] == 1, desca[0]);
  DLAF_ASSERT(descz[0] == 1, descz[0]);
  DLAF_ASSERT(desca[1] == descz[1], desca[1], descz[1]);
  DLAF_ASSERT(ia == 1, ia);
  DLAF_ASSERT(ja == 1, ja);
  DLAF_ASSERT(iz == 1, iz);
  DLAF_ASSERT(iz == 1, iz);
  DLAF_ASSERT(m > 0 ? eigenvalues_index_begin >= 1 : eigenvalues_index_begin == 1, m,
              eigenvalues_index_begin);
  DLAF_ASSERT(m > 0 ? eigenvalues_index_end <= m : eigenvalues_index_end == 0, m, eigenvalues_index_end);
  DLAF_ASSERT(m > 0 ? eigenvalues_index_begin <= eigenvalues_index_end : true, m,
              eigenvalues_index_begin, eigenvalues_index_end);

  auto dlaf_desca = make_dlaf_descriptor(m, m, ia, ja, desca);
  auto dlaf_descz = make_dlaf_descriptor(m, m, iz, jz, descz);

  auto _info = hermitian_eigensolver(desca[1], uplo, a, dlaf_desca, w, z, dlaf_descz,
                                     eigenvalues_index_begin - 1, eigenvalues_index_end);
  info = _info;
}

#endif
