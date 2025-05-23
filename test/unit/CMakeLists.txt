#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

DLAF_addTest(
  test_types
  SOURCES test_types.cpp
  LIBRARIES dlaf.core
  USE_MAIN PLAIN
)

if(DLAF_WITH_GPU)
  DLAF_addTest(
    test_util_cuda
    SOURCES test_util_cuda.cu
    LIBRARIES dlaf.core
    USE_MAIN PLAIN
  )
endif()

DLAF_addTest(
  test_util_math
  SOURCES test_util_math.cpp
  LIBRARIES dlaf.core
  USE_MAIN PLAIN
)

DLAF_addTest(
  test_blas_tile
  SOURCES test_blas_tile.cpp
  LIBRARIES dlaf.core
  USE_MAIN PIKA
)

DLAF_addTest(
  test_blas_tile_extensions
  SOURCES test_blas_tile_extensions.cpp
  LIBRARIES dlaf.core
  USE_MAIN PIKA
)

DLAF_addTest(
  test_lapack_tile
  SOURCES test_lapack_tile.cpp
  LIBRARIES dlaf.core
  USE_MAIN PIKA
)

DLAF_addTest(
  test_lapack_tile_extensions
  SOURCES test_lapack_tile_extensions.cpp
  LIBRARIES dlaf.core
  USE_MAIN PIKA
)

# Generic libraries
add_subdirectory(common)
add_subdirectory(communication)
add_subdirectory(init)
add_subdirectory(lapack)
add_subdirectory(matrix)
add_subdirectory(memory)
add_subdirectory(sender)

# Operations
add_subdirectory(auxiliary)
add_subdirectory(eigensolver)
add_subdirectory(factorization)
add_subdirectory(inverse)
add_subdirectory(multiplication)
add_subdirectory(permutations)
add_subdirectory(solver)

# C API
add_subdirectory(c_api/eigensolver)
add_subdirectory(c_api/factorization)
add_subdirectory(c_api/inverse)
add_subdirectory(c_api/grid)
