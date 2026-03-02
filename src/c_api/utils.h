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

#include <optional>
#include <tuple>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/matrix/col_major_layout.h>
#include <dlaf/matrix/distribution.h>

dlaf::matrix::ColMajorLayout make_layout(const struct DLAF_descriptor dlaf_desc,
                                         dlaf::comm::CommunicatorGrid& grid);

dlaf::common::Ordering char2order(const char order);

dlaf::comm::CommunicatorGrid& grid_from_context(int dlaf_context);

/// Returns the block size to use for device matrices in DLA-Future C API solvers.
/// Reads from DLAF_INTERNAL_BLOCK_SIZE environment variable, returns std::nullopt if not set.
std::optional<SizeType> get_internal_block_size();

/// Returns the device matrix distribution if redistribution is needed, std::nullopt otherwise.
///
/// Redistribution is needed when DLAF_INTERNAL_BLOCK_SIZE is set and differs from the host
/// block size. The returned distribution uses the same process grid and source rank as @p dist_host,
/// but with the block size from DLAF_INTERNAL_BLOCK_SIZE.
std::optional<dlaf::matrix::Distribution> get_device_distribution(
    const dlaf::matrix::Distribution& dist_host);

struct PikaScope{
    PikaScope() {
        pika::resume();
    }
    ~PikaScope() {
        pika::stop();
    }
};