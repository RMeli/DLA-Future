//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "cholesky.h"

#include <dlaf_c/factorization/cholesky.h>
#include <dlaf_c/utils.h>

int dlaf_cholesky_s(const int dlaf_context, const char uplo, float* a,
                    const DLAF_descriptor dlaf_desca) {
  return cholesky<float>(dlaf_context, uplo, a, dlaf_desca);
}

int dlaf_cholesky_d(const int dlaf_context, const char uplo, double* a,
                    const DLAF_descriptor dlaf_desca) {
  return cholesky<double>(dlaf_context, uplo, a, dlaf_desca);
}

int dlaf_cholesky_c(const int dlaf_context, const char uplo, dlaf_complex_c* a,
                    const DLAF_descriptor dlaf_desca) {
  return cholesky<std::complex<float>>(dlaf_context, uplo, a, dlaf_desca);
}

int dlaf_cholesky_z(const int dlaf_context, const char uplo, dlaf_complex_z* a,
                    const DLAF_descriptor dlaf_desca) {
  return cholesky<std::complex<double>>(dlaf_context, uplo, a, dlaf_desca);
}

#ifdef DLAF_WITH_SCALAPACK

void dlaf_pspotrf(const char uplo, const int n, float* a, const int ia, const int ja, const int desca[9],
                  int* info) {
  pxpotrf<float>(uplo, n, a, ia, ja, desca, *info);
}

void dlaf_pdpotrf(const char uplo, const int n, double* a, const int ia, const int ja,
                  const int desca[9], int* info) {
  pxpotrf<double>(uplo, n, a, ia, ja, desca, *info);
}

void dlaf_pcpotrf(const char uplo, const int n, dlaf_complex_c* a, const int ia, const int ja,
                  const int desca[9], int* info) {
  pxpotrf<std::complex<float>>(uplo, n, a, ia, ja, desca, *info);
}

void dlaf_pzpotrf(const char uplo, const int n, dlaf_complex_z* a, const int ia, const int ja,
                  const int desca[9], int* info) {
  pxpotrf<std::complex<double>>(uplo, n, a, ia, ja, desca, *info);
}

#endif
