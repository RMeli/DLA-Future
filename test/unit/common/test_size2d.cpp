//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <array>
#include <sstream>

#include <dlaf/common/index2d.h>

#include <gtest/gtest.h>

template <typename IndexType>
using Size2D = dlaf::common::Size2D<IndexType, struct TAG_TEST>;

template <typename IndexType>
using Index2D = dlaf::common::Index2D<IndexType, struct TAG_TEST>;

template <typename IndexType>
class Size2DTest : public ::testing::Test {};

using IndexTypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t>;
TYPED_TEST_SUITE(Size2DTest, IndexTypes);

TYPED_TEST(Size2DTest, ConstructorFromParams) {
  TypeParam row = 5;
  TypeParam col = 3;
  Size2D<TypeParam> size(row, col);

  EXPECT_EQ(row, size.rows());
  EXPECT_EQ(col, size.cols());

  EXPECT_TRUE(size.isValid());
}

TYPED_TEST(Size2DTest, ConstructorFromArray) {
  std::array<TypeParam, 2> coords{5, 3};
  Size2D<TypeParam> size(coords);

  EXPECT_EQ(coords[0], size.rows());
  EXPECT_EQ(coords[1], size.cols());

  EXPECT_TRUE(size.isValid());
}

TYPED_TEST(Size2DTest, Comparison) {
  TypeParam row = 5;
  TypeParam col = 3;
  Size2D<TypeParam> size1(row, col);

  std::array<TypeParam, 2> coords{row, col};
  Size2D<TypeParam> size2(coords);

  EXPECT_TRUE(size1 == size2);
  EXPECT_FALSE(size1 != size2);

  Size2D<TypeParam> size3(row + 1, col);
  Size2D<TypeParam> size4(row, col - 1);
  Size2D<TypeParam> size5(row + 4, col - 2);

  EXPECT_TRUE(size1 != size3);
  EXPECT_TRUE(size1 != size4);
  EXPECT_TRUE(size1 != size5);
  EXPECT_FALSE(size1 == size3);
  EXPECT_FALSE(size1 == size4);
  EXPECT_FALSE(size1 == size5);
}

TYPED_TEST(Size2DTest, Transpose) {
  const Size2D<TypeParam> size_original(7, 13);
  Size2D<TypeParam> size = size_original;

  // tranpose it (with member function)
  size.transpose();
  EXPECT_EQ(Size2D<TypeParam>(13, 7), size);

  // get its tranpose, without changing it (with free function)
  const auto size_transposed = transposed(size);
  // check that tranpose is self-inverse
  EXPECT_EQ(size_original, size_transposed);
  // check the source has not been changed
  EXPECT_EQ(Size2D<TypeParam>(13, 7), size);
}

TYPED_TEST(Size2DTest, Print) {
  Size2D<TypeParam> size1(7, 13);
  std::array<TypeParam, 2> coords{9, 6};
  Size2D<TypeParam> size2(coords);

  std::stringstream s;
  s << size1;
  EXPECT_EQ("(7, 13)", s.str());

  s.str("");
  s << size2;
  EXPECT_EQ("(9, 6)", s.str());
}

TYPED_TEST(Size2DTest, Arithmetic) {
  using size2d_t = Size2D<TypeParam>;
  using index2d_t = Index2D<TypeParam>;
  ASSERT_TRUE(index2d_t(6, 5) - size2d_t(3, 4) == index2d_t(3, 1));
  ASSERT_TRUE(index2d_t(6, 5) + size2d_t(3, 4) == index2d_t(9, 9));
  ASSERT_TRUE(size2d_t(6, 5) - size2d_t(3, 4) == size2d_t(3, 1));
  ASSERT_TRUE(size2d_t(6, 5) + size2d_t(3, 4) == size2d_t(9, 9));
  ASSERT_TRUE(index2d_t(6, 5) - index2d_t(3, 4) == size2d_t(3, 1));
}
