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

/// @file

#include <blas.hh>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/factorization/cholesky/api.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

namespace dlaf {

/// Cholesky factorization which computes the factorization of an Hermitian positive
/// definite matrix A.
///
/// The factorization has the form A=LL^H (uplo = Lower) or A = U^H U (uplo = Upper),
/// where L is a lower and U is an upper triangular matrix.
/// @param uplo specifies if the elements of the Hermitian matrix to be referenced are the elements in
/// the lower or upper triangular part,
///
/// @param mat_a on entry it contains the triangular matrix A, on exit the matrix elements
/// are overwritten with the elements of the Cholesky factor. Only the tiles of the matrix
/// which contain the upper or the lower triangular part (depending on the value of uplo),
/// are accessed and modified.
///
/// @pre @p mat_a is not distributed
/// @pre @p mat_a has size (N x N)
/// @pre @p mat_a has block size (NB x NB)
/// @pre @p mat_a has tile size (NB x NB)
template <Backend backend, Device device, class T>
void cholesky_factorization(blas::Uplo uplo, Matrix<T, device>& mat_a) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_block_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_a), mat_a);
  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);

  if (uplo == blas::Uplo::Lower)
    factorization::internal::Cholesky<backend, device, T>::call_L(mat_a);
  else
    factorization::internal::Cholesky<backend, device, T>::call_U(mat_a);
}

/// Cholesky factorization which computes the factorization of an Hermitian positive
/// definite matrix A.
///
/// The factorization has the form A=LL^H (uplo = Lower) or A = U^H U (uplo = Upper),
/// where L is a lower and U is an upper triangular matrix.
/// @param grid is the communicator grid on which the matrix A has been distributed,
/// @param uplo specifies if the elements of the Hermitian matrix to be referenced are the elements in
/// the lower or upper triangular part,
/// @param mat_a on entry it contains the triangular matrix A, on exit the matrix elements
/// are overwritten with the elements of the Cholesky factor. Only the tiles of the matrix
/// which contain the upper or the lower triangular part (depending on the value of uplo),
/// are accessed and modified.
///
/// @pre @p mat_a is distributed according to @p grid
/// @pre @p mat_a has size (N x N)
/// @pre @p mat_a has block size (NB x NB)
/// @pre @p mat_a has tile size (NB x NB)
template <Backend backend, Device device, class T>
void cholesky_factorization(comm::CommunicatorGrid& grid, blas::Uplo uplo, Matrix<T, device>& mat_a) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_block_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_a), mat_a);
  DLAF_ASSERT(matrix::equal_process_grid(mat_a, grid), mat_a, grid);

  // Method only for Lower triangular matrix
  if (uplo == blas::Uplo::Lower)
    factorization::internal::Cholesky<backend, device, T>::call_L(grid, mat_a);
  else
    factorization::internal::Cholesky<backend, device, T>::call_U(grid, mat_a);
}

}
