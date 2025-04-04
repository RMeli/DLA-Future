//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <vector>

#include <dlaf/common/single_threaded_blas.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/eigensolver/tridiag_solver/rot.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>

#include <gtest/gtest.h>

#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/matrix/matrix_local.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/matrix/util_matrix_local.h>
#include <dlaf_test/util_types.h>

using namespace dlaf;
using namespace dlaf::test;

namespace ex = pika::execution::experimental;
namespace di = dlaf::eigensolver::internal;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename T>
struct TridiagEigensolverRotTest : public TestWithCommGrids {
  static constexpr T angle = static_cast<T>(2.6);
  const T rot_c = std::cos(angle);
  const T rot_s = std::sin(angle);

  using GRot = di::GivensRotation<T>;

  struct config_t {
    SizeType m;
    SizeType mb;
    SizeType i_begin;
    SizeType i_end;
    std::vector<GRot> rots;
  };

  // Note:
  // GivenRotation indices are relative to the range [i_begin, i_end)
  const std::vector<config_t> configs{
      // range with one-sided margin
      {9, 3, 1, 3, {GRot{0, 5, rot_c, rot_s}}},
      {8, 3, 1, 3, {GRot{0, 2, rot_c, rot_s}}},  // incomplete tile
      {9, 3, 0, 2, {GRot{2, 4, rot_c, rot_s}}},
      // range fully in-bound
      {12, 3, 1, 3, {GRot{0, 5, rot_c, rot_s}}},
      {11, 3, 1, 3, {GRot{0, 5, rot_c, rot_s}}},  // incomplete tile
      // full-range, multiple rotations
      {9, 3, 0, 3, {GRot{0, 8, rot_c, rot_s}, GRot{0, 2, rot_c, rot_s}}},
      {8, 3, 0, 3, {GRot{0, 7, rot_c, rot_s}, GRot{0, 2, rot_c, rot_s}}},  // incomplete tile
      // range fully in-bound, independent rotations from same tiles
      {12, 3, 1, 3, {GRot{0, 5, rot_c, rot_s}, GRot{1, 2, rot_c, rot_s}}},
      // range fully in-bound, non-independent rotations, between same pair of tiles
      {12, 3, 1, 3, {GRot{0, 5, rot_c, rot_s}, GRot{0, 4, rot_c, rot_s}}},
      {12, 3, 1, 3, {GRot{0, 5, rot_c, rot_s}, GRot{1, 5, rot_c, rot_s}}},
  };
};

template <typename T>
using TridiagEigensolverRotMCTest = TridiagEigensolverRotTest<T>;

TYPED_TEST_SUITE(TridiagEigensolverRotMCTest, RealMatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <typename T>
using TridiagEigensolverRotGPUTest = TridiagEigensolverRotTest<T>;

TYPED_TEST_SUITE(TridiagEigensolverRotGPUTest, RealMatrixElementTypes);
#endif

template <class T, Device D>
void testApplyGivenRotations(comm::CommunicatorGrid& grid, const SizeType m, const SizeType mb,
                             const SizeType idx_begin, const SizeType idx_end,
                             std::vector<di::GivensRotation<T>> rots) {
  using dlaf::eigensolver::internal::applyGivensRotationsToMatrixColumns;

  constexpr comm::IndexT_MPI tag = 0;
  matrix::Distribution dist({m, m}, {mb, mb}, grid.size(), grid.rank(), {0, 0});

  matrix::Matrix<T, Device::CPU> mat_h(dist);
  matrix::util::set_random(mat_h);

  matrix::test::MatrixLocal<T> mat_loc = matrix::test::allGather<T>(blas::Uplo::General, mat_h, grid);

  {
    matrix::MatrixMirror<T, D, Device::CPU> mat(mat_h);
    auto comm_row_chain = grid.row_communicator_pipeline();
    applyGivensRotationsToMatrixColumns(comm_row_chain, tag, idx_begin, idx_end, ex::just(rots),
                                        mat.get());
  }

  // Apply Given Rotations
  const SizeType n = std::min((idx_end) *mb, m) - idx_begin * mb;
  const GlobalElementSize offset(idx_begin * mb, idx_begin * mb);

  {
    dlaf::common::internal::SingleThreadedBlasScope single;

    for (auto rot : rots) {
      T* x = mat_loc.ptr(GlobalElementIndex{0, rot.i} + offset);
      T* y = mat_loc.ptr(GlobalElementIndex{0, rot.j} + offset);
      blas::rot(n, x, 1, y, 1, rot.c, rot.s);
    }
  }

  auto result = [&dist = mat_h.distribution(), &mat_local = mat_loc](const GlobalElementIndex& element) {
    const auto tile_index = dist.globalTileIndex(element);
    const auto tile_element = dist.tileElementIndex(element);
    return mat_local.tile_read(tile_index)(tile_element);
  };

  CHECK_MATRIX_NEAR(result, mat_h, m * TypeUtilities<T>::error, m * TypeUtilities<T>::error);
}

TYPED_TEST(TridiagEigensolverRotMCTest, ApplyGivenRotations) {
  for (auto& grid : this->commGrids()) {
    for (const auto& [m, mb, idx_begin, idx_end, rots] : this->configs) {
      testApplyGivenRotations<TypeParam, Device::CPU>(grid, m, mb, idx_begin, idx_end, rots);
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(TridiagEigensolverRotGPUTest, ApplyGivenRotations) {
  for (auto& grid : this->commGrids()) {
    for (const auto& [m, mb, idx_begin, idx_end, rots] : this->configs) {
      testApplyGivenRotations<TypeParam, Device::GPU>(grid, m, mb, idx_begin, idx_end, rots);
    }
  }
}
#endif
