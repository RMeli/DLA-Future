#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

cmake_minimum_required(VERSION 3.22)
project(DLAF-miniapps)

list(PREPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(DLAF REQUIRED)
find_package(pika REQUIRED)

include(DLAF_AddMiniapp)
include(DLAF_AddTargetWarnings)

add_library(DLAF_miniapp INTERFACE)
target_link_libraries(DLAF_miniapp INTERFACE DLAF::DLAF pika::pika)
target_include_directories(DLAF_miniapp INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

DLAF_addMiniapp(miniapp_cholesky SOURCES miniapp_cholesky.cpp)
DLAF_addMiniapp(miniapp_gen_to_std SOURCES miniapp_gen_to_std.cpp)
DLAF_addMiniapp(miniapp_reduction_to_band SOURCES miniapp_reduction_to_band.cpp)
DLAF_addMiniapp(miniapp_band_to_tridiag SOURCES miniapp_band_to_tridiag.cpp)
DLAF_addMiniapp(miniapp_tridiag_solver SOURCES miniapp_tridiag_solver.cpp)
DLAF_addMiniapp(miniapp_bt_band_to_tridiag SOURCES miniapp_bt_band_to_tridiag.cpp)
DLAF_addMiniapp(miniapp_bt_reduction_to_band SOURCES miniapp_bt_reduction_to_band.cpp)
DLAF_addMiniapp(miniapp_triangular_solver SOURCES miniapp_triangular_solver.cpp)
DLAF_addMiniapp(miniapp_triangular_multiplication SOURCES miniapp_triangular_multiplication.cpp)
DLAF_addMiniapp(miniapp_eigensolver SOURCES miniapp_eigensolver.cpp)
DLAF_addMiniapp(miniapp_gen_eigensolver SOURCES miniapp_gen_eigensolver.cpp)
DLAF_addMiniapp(miniapp_communication SOURCES miniapp_communication.cpp)
DLAF_addMiniapp(miniapp_triangular_inverse SOURCES miniapp_triangular_inverse.cpp)
DLAF_addMiniapp(miniapp_inverse_from_cholesky_factor SOURCES miniapp_inverse_from_cholesky_factor.cpp)

if(DLAF_BUILD_TESTING)
  set(miniapp_test_args
      USE_MAIN
      MPIPIKA
      MPIRANKS
      6
      ARGUMENTS
      --grid-rows=3
      --grid-cols=2
      --check=all
      CATEGORY
      MINIAPP
  )
  DLAF_addTargetTest(miniapp_cholesky ${miniapp_test_args})
  DLAF_addTargetTest(miniapp_gen_to_std ${miniapp_test_args})
  DLAF_addTargetTest(miniapp_reduction_to_band ${miniapp_test_args})
  DLAF_addTargetTest(miniapp_band_to_tridiag ${miniapp_test_args})
  DLAF_addTargetTest(miniapp_tridiag_solver ${miniapp_test_args})
  DLAF_addTargetTest(miniapp_bt_band_to_tridiag ${miniapp_test_args})
  DLAF_addTargetTest(miniapp_bt_reduction_to_band ${miniapp_test_args})
  DLAF_addTargetTest(miniapp_triangular_solver ${miniapp_test_args})
  DLAF_addTargetTest(miniapp_triangular_multiplication ${miniapp_test_args})
  DLAF_addTargetTest(miniapp_eigensolver ${miniapp_test_args})
  DLAF_addTargetTest(miniapp_gen_eigensolver ${miniapp_test_args})
  DLAF_addTargetTest(miniapp_communication ${miniapp_test_args})
  DLAF_addTargetTest(miniapp_triangular_inverse ${miniapp_test_args})
  DLAF_addTargetTest(miniapp_inverse_from_cholesky_factor ${miniapp_test_args})
endif()

add_subdirectory(kernel)
