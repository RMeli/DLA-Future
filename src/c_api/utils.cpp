//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf_c/desc.h>
#include <dlaf_c/utils.h>

#include "grid.h"
#include "utils.h"

struct DLAF_descriptor make_dlaf_descriptor(const int m, const int n, const int i, const int j,
                                            const int desc[9]) noexcept {
  DLAF_ASSERT(i == 1, i);
  DLAF_ASSERT(j == 1, j);

  struct DLAF_descriptor dlaf_desc = {m, n, desc[4], desc[5], desc[6], desc[7], i - 1, j - 1, desc[8]};

  return dlaf_desc;
}

dlaf::matrix::ColMajorLayout make_layout(const struct DLAF_descriptor dlaf_desc,
                                         dlaf::comm::CommunicatorGrid& grid) {
  dlaf::GlobalElementSize matrix_size(dlaf_desc.m, dlaf_desc.n);
  dlaf::TileElementSize block_size(dlaf_desc.mb, dlaf_desc.nb);

  dlaf::comm::Index2D src_rank_index(dlaf_desc.isrc, dlaf_desc.jsrc);

  dlaf::matrix::Distribution distribution(matrix_size, block_size, grid.size(), grid.rank(),
                                          src_rank_index);

  dlaf::matrix::ColMajorLayout layout{std::move(distribution), dlaf_desc.ld};

  return layout;
}

dlaf::common::Ordering char2order(const char order) {
  return order == 'C' or order == 'c' ? dlaf::common::Ordering::ColumnMajor
                                      : dlaf::common::Ordering::RowMajor;
}

dlaf::comm::CommunicatorGrid& grid_from_context(int dlaf_context) {
  try {
    return dlaf_grids.at(dlaf_context);
  }
  catch (const std::out_of_range& e) {
    std::stringstream ss;
    ss << "[ERROR] No DLA-Future grid for context " << dlaf_context << ". ";
    ss << "Did you forget to call dlaf_create_grid() or dlaf_create_grid_from_blacs()?\n";

    std::cerr << ss.str() << std::flush;

    std::terminate();
  }
}

std::optional<SizeType> get_internal_block_size() {
  if (const char* env_value = std::getenv("DLAF_INTERNAL_BLOCK_SIZE")) {
    try {
      long value = std::stol(env_value);
      if (value > 0) {
        return static_cast<SizeType>(value);
      }
      else {
        std::cerr << "[WARNING] DLAF_INTERNAL_BLOCK_SIZE must be positive, ignoring.\n";
      }
    }
    catch (const std::exception& e) {
      std::cerr << "[WARNING] Invalid value for DLAF_INTERNAL_BLOCK_SIZE: '" << env_value
                << "', ignoring.\n";
    }
  }

  return std::nullopt;
}

std::optional<dlaf::matrix::Distribution> get_device_distribution(
    const dlaf::matrix::Distribution& dist_host) {
  const auto opt_block_size = get_internal_block_size();
  
  const dlaf::GlobalElementSize matrix_size = dist_host.size();

  if (!opt_block_size.has_value() || matrix_size.isEmpty()){
    return std::nullopt;
  }

  const SizeType device_block_size = *opt_block_size;

  if (dist_host.block_size().rows() == device_block_size &&
      dist_host.block_size().cols() == device_block_size){
    return std::nullopt;
}

  const dlaf::TileElementSize tile_size(std::min(device_block_size, matrix_size.rows()),
                                        std::min(device_block_size, matrix_size.cols()));

  return dlaf::matrix::Distribution(matrix_size, tile_size, dist_host.grid_size(),
                                    dist_host.rank_index(), dist_host.source_rank_index());
}
