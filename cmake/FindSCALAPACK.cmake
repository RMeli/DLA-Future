#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

include(FindPackageHandleStandardArgs)
find_package(PkgConfig REQUIRED)

pkg_search_module(_SCALAPACK scalapack)

find_library(
  DLAF_SCALAPACK_LIBRARY NAMES scalapack
  HINTS ${_SCALAPACK_LIBRARY_DIRS}
        ENV
        SCALAPACKROOT
        SCALAPACK_ROOT
        SCALAPACK_PREFIX
        SCALAPACK_DIR
        SCALAPACKDIR
        /usr
  PATH_SUFFIXES lib
)

find_package_handle_standard_args(SCALAPACK DEFAULT_MSG DLAF_SCALAPACK_LIBRARY)

mark_as_advanced(DLAF_SCALAPACK_LIBRARY)
mark_as_advanced(DLAF_SCALAPACK_INCLUDE_DIR)

if(NOT TARGET DLAF::SCALAPACK)
  add_library(DLAF::SCALAPACK INTERFACE IMPORTED GLOBAL)
endif()

target_link_libraries(DLAF::SCALAPACK INTERFACE "${DLAF_SCALAPACK_LIBRARY}")
target_include_directories(DLAF::SCALAPACK INTERFACE "${DLAF_SCALAPACK_INCLUDE_DIR}")
