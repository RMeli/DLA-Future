//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/layout_info.h>

#include <mpi.h>


namespace dlaf::interface::blacs {

struct DlafSetup {
  dlaf::matrix::Distribution distribution;
  dlaf::matrix::LayoutInfo layout_info;
  dlaf::comm::CommunicatorGrid communicator_grid;
};

DlafSetup from_desc(int* desc);

}
