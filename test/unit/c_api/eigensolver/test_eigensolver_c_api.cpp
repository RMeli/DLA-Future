//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <functional>
#include <tuple>

#include <pika/init.hpp>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/eigensolver/eigensolver.h>
#include <dlaf/eigensolver/eigensolver/api.h>
#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/tune.h>
#include <dlaf/types.h>
#include <dlaf_c/grid.h>
#include <dlaf_c/init.h>

#include "test_eigensolver_c_api_wrapper.h"

#include <gtest/gtest.h>

#include <dlaf_test/blacs.h>
#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/eigensolver/test_eigensolver_correctness.h>
#include <dlaf_test/matrix/matrix_local.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/matrix/util_matrix_local.h>
#include <dlaf_test/util_types.h>

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksCAPIEnvironment);

template <typename Type>
class EigensolverTest : public TestWithCommGrids {};

template <class T>
using EigensolverTestMC = EigensolverTest<T>;

TYPED_TEST_SUITE(EigensolverTestMC, RealMatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
using EigensolverTestGPU = EigensolverTest<T>;

TYPED_TEST_SUITE(EigensolverTestGPU, RealMatrixElementTypes);
#endif

const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower});

const std::vector<std::tuple<SizeType, SizeType, SizeType>> sizes = {
    // {m, mb, eigensolver_min_band}
    {34, 13, 100},
    {32, 5, 100},  // m > mb
    {34, 8, 3},
    {32, 6, 3}  // m > mb, sub-band
};

enum class API { dlaf, scalapack };

template <class T, Backend B, Device D, API api>
void testEigensolver(const blas::Uplo uplo, const SizeType m, const SizeType mb, CommunicatorGrid grid) {
  const char* pika_argv[] = {"test_eigensolver_c_api", "--pika:print-bind", nullptr};
  const char* dlaf_argv[] = {"test_eigensolver_c_api", "--pika:print-bind", nullptr};
  dlaf_initialize(2, pika_argv, 1, dlaf_argv);

  char grid_order = grid_ordering(MPI_COMM_WORLD, grid.size().rows(), grid.size().cols(),
                                  grid.rank().row(), grid.rank().col());

  int dlaf_context = -1;
  if constexpr (api == API::dlaf) {
    // Create DLAF grid directly
    dlaf_context = dlaf_create_grid(MPI_COMM_WORLD, grid.size().rows(), grid.size().cols(), grid_order);
  }
  else if constexpr (api == API::scalapack) {
    // Create BLACS grid
    Cblacs_get(0, 0, &dlaf_context);  // Default system context
    Cblacs_gridinit(&dlaf_context, &grid_order, grid.size().rows(), grid.size().cols());

    // Create DLAF grid from BLACS context
    dlaf_create_grid_from_blacs(dlaf_context);
  }

  // In normal use the runtime is resumed by the C API call
  // The pika runtime is suspended by dlaf_initialize
  // Here we need to resume it manually to build the matrices with DLA-Future
  pika::resume();

  const LocalElementSize size(m, m);
  const TileElementSize block_size(mb, mb);

  Matrix<const T, Device::CPU> reference = [&]() {
    auto reference = [&]() -> auto{
      return Matrix<T, Device::CPU>(GlobalElementSize(m, m), block_size, grid);
    }
    ();
    matrix::util::set_random_hermitian(reference);
    return reference;
  }();

  Matrix<T, Device::CPU> mat_a_h(reference.distribution());
  copy(reference, mat_a_h);
  mat_a_h.waitLocalTiles();

  eigensolver::EigensolverResult<T, D> ret = [&]() {
    const SizeType size = mat_a_h.size().rows();
    Matrix<BaseType<T>, D> eigenvalues(LocalElementSize(size, 1),
                                       TileElementSize(mat_a_h.blockSize().rows(), 1));
    Matrix<T, D> eigenvectors(GlobalElementSize(size, size), mat_a_h.blockSize(), grid);

    eigenvalues.waitLocalTiles();
    eigenvectors.waitLocalTiles();

    char dlaf_uplo = uplo == blas::Uplo::Upper ? 'U' : 'L';

    // Get top left local tiles
    int lld_a, lld_eigenvectors;
    T *local_a_ptr, *local_eigenvectors_ptr, *eigenvalues_ptr;
    {
      auto toplefttile_a =
          pika::this_thread::experimental::sync_wait(mat_a_h.readwrite(LocalTileIndex(0, 0)));
      auto toplefttile_eigenvalues =
          pika::this_thread::experimental::sync_wait(eigenvalues.readwrite(LocalTileIndex(0, 0)));
      auto toplefttile_eigenvectors =
          pika::this_thread::experimental::sync_wait(eigenvectors.readwrite(LocalTileIndex(0, 0)));

      lld_a = static_cast<int>(toplefttile_a.ld());
      lld_eigenvectors = static_cast<int>(toplefttile_eigenvectors.ld());

      local_a_ptr = toplefttile_a.ptr();
      local_eigenvectors_ptr = toplefttile_eigenvectors.ptr();
      eigenvalues_ptr = toplefttile_eigenvalues.ptr();

    }  // Destroy tiles (avoid spurious dependencies)

    // Suspend pika to ensure it is resumed by the C API
    pika::suspend();

    if constexpr (api == API::dlaf) {
      DLAF_descriptor dlaf_desc_a = {(int) m, (int) m, (int) mb, (int) mb, 0, 0, 1, 1, lld_a};
      DLAF_descriptor dlaf_desc_eigenvectors = {(int) m, (int) m, (int) mb, (int) mb,        0,
                                                0,       1,       1,        lld_eigenvectors};

      if constexpr (std::is_same_v<T, double>) {
        C_dlaf_eigensolver_d(dlaf_context, dlaf_uplo, local_a_ptr, dlaf_desc_a, eigenvalues_ptr,
                             local_eigenvectors_ptr, dlaf_desc_eigenvectors);
      }
      else {
        C_dlaf_eigensolver_s(dlaf_context, dlaf_uplo, local_a_ptr, dlaf_desc_a, eigenvalues_ptr,
                             local_eigenvectors_ptr, dlaf_desc_eigenvectors);
      }
    }
    else if constexpr (api == API::scalapack) {
      int desc_a[] = {1, dlaf_context, (int) m, (int) m, (int) mb, (int) mb, 0, 0, lld_a};
      int desc_z[] = {1, dlaf_context, (int) m, (int) m, (int) mb, (int) mb, 0, 0, lld_eigenvectors};
      int info = -1;
      if constexpr (std::is_same_v<T, double>) {
        C_dlaf_pdsyevd(dlaf_uplo, (int) m, local_a_ptr, desc_a, eigenvalues_ptr, local_eigenvectors_ptr,
                       desc_z, &info);
      }
      else {
        C_dlaf_pssyevd(dlaf_uplo, (int) m, local_a_ptr, desc_a, eigenvalues_ptr, local_eigenvectors_ptr,
                       desc_z, &info);
      }
    }

    return eigensolver::EigensolverResult<T, D>{std::move(eigenvalues), std::move(eigenvectors)};
  }();

  if (mat_a_h.size().isEmpty())
    return;

  // Resume pika runtime suspended by C API for correctness checks
  pika::resume();

  testEigensolverCorrectness(uplo, reference, ret.eigenvalues, ret.eigenvectors, grid);

  // Suspend pika to make sure dla_finalize resumes it
  pika::suspend();

  dlaf_free_grid(dlaf_context);
  dlaf_finalize();

  if constexpr (api == API::scalapack) {
    Cblacs_gridexit(dlaf_context);
  }
}

TYPED_TEST(EigensolverTestMC, CorrectnessDistributedDLAF) {
  for (const comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        // getTuneParameters().eigensolver_min_band = b_min;
        testEigensolver<TypeParam, Backend::MC, Device::CPU, API::dlaf>(uplo, m, mb, grid);
      }
    }
  }
}

TYPED_TEST(EigensolverTestMC, CorrectnessDistributedScalapack) {
  for (const comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        testEigensolver<TypeParam, Backend::MC, Device::CPU, API::scalapack>(uplo, m, mb, grid);
      }
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(EigensolverTestGPU, CorrectnessDistributedDLAF) {
  for (const comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        // getTuneParameters().eigensolver_min_band = b_min;
        testEigensolver<TypeParam, Backend::GPU, Device::GPU, API::dlaf>(uplo, m, mb, grid);
      }
    }
  }
}

TYPED_TEST(EigensolverTestGPU, CorrectnessDistributedScalapack) {
  for (const comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        testEigensolver<TypeParam, Backend::GPU, Device::GPU, API::scalapack>(uplo, m, mb, grid);
      }
    }
  }
}
#endif
