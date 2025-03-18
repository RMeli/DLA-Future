//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <complex>

#include <dlaf/eigensolver/bt_band_to_tridiag/impl.h>

namespace dlaf::eigensolver::internal {

DLAF_EIGENSOLVER_BT_BAND_TO_TRIDIAGONAL_ETI(, Backend::MC, Device::CPU, float)
DLAF_EIGENSOLVER_BT_BAND_TO_TRIDIAGONAL_ETI(, Backend::MC, Device::CPU, double)
DLAF_EIGENSOLVER_BT_BAND_TO_TRIDIAGONAL_ETI(, Backend::MC, Device::CPU, std::complex<float>)
DLAF_EIGENSOLVER_BT_BAND_TO_TRIDIAGONAL_ETI(, Backend::MC, Device::CPU, std::complex<double>)
}
