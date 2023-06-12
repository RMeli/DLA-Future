//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "test_cholesky_c_api_wrapper.h"

#include "dlaf_c/factorization/cholesky.h"
#include "dlaf_c/grid.h"
#include "dlaf_c/init.h"
#include "dlaf_c/utils.h"

#include <gtest/gtest.h>
#include <pika/runtime.hpp>

#include <mpi.h>

#include <iostream>

// BLACS
DLAF_EXTERN_C void Cblacs_gridinit(int* ictxt, char* layout, int nprow, int npcol);
DLAF_EXTERN_C void Cblacs_gridexit(int ictxt);

// ScaLAPACK
DLAF_EXTERN_C int numroc(const int* n, const int* nb, const int* iproc, const int* isrcproc,
                         const int* nprocs);
DLAF_EXTERN_C void descinit(int* desc, const int* m, const int* n, const int* mb, const int* nb,
                            const int* irsrc, const int* icsrc, const int* ictxt, const int* lld,
                            int* info);
DLAF_EXTERN_C void pdgemr2d(int* m, int* n, double* A, int* ia, int* ja, int* desca, double* B, int* ib,
                            int* jb, int* descb, int* ictxt);

int izero = 0;
int ione = 1;

// TODO: Move?
#include "dlaf/communication/error.h"
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_grid.h>
dlaf::common::Ordering grid_order(MPI_Comm& communicator, int nprow, int npcol, int myprow, int mypcol) {
  int rank;
  DLAF_MPI_CHECK_ERROR(MPI_Comm_rank(communicator, &rank));

  std::cout << rank << ' ' << myprow << ' ' << mypcol << '\n';

  bool _row_major = false, _col_major = false;
  bool row_major, col_major;

  if (rank == myprow * npcol + mypcol) {
    _row_major = true;
  }
  else if (rank == mypcol * nprow + myprow) {
    _col_major = true;
  }

  DLAF_MPI_CHECK_ERROR(MPI_Allreduce(&_row_major, &row_major, 1, MPI_C_BOOL, MPI_LAND, communicator));
  DLAF_MPI_CHECK_ERROR(MPI_Allreduce(&_col_major, &col_major, 1, MPI_C_BOOL, MPI_LAND, communicator));

  if (row_major) {
    return dlaf::common::Ordering::RowMajor;
  }
  else if (col_major) {
    return dlaf::common::Ordering::ColumnMajor;
  }
  // TODO: Deal with gridmap-initialised grids
}

TEST(CAPIScaLAPACKTest, GridOrderColumnR) {
  int rank;
  int num_ranks;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  EXPECT_EQ(num_ranks, 6);

  int nprow = 2;  // Rows of process grid
  int npcol = 3;  // Cols of process grid

  EXPECT_EQ(nprow * npcol, num_ranks);

  char order = 'R';

  int contxt = 0;

  Cblacs_get(0, 0, &contxt);
  Cblacs_gridinit(&contxt, &order, nprow, npcol);

  int system_ctxt;
  // SGET_BLACSCONTXT == 10
  int get_blacs_contxt = 10;
  Cblacs_get(contxt, get_blacs_contxt, &system_ctxt);

  MPI_Comm communicator = Cblacs2sys_handle(system_ctxt);

  int dims[2] = {0, 0};
  int coords[2] = {-1, -1};

  Cblacs_gridinfo(contxt, &dims[0], &dims[1], &coords[0], &coords[1]);

  auto go = grid_order(communicator, dims[0], dims[1], coords[0], coords[1]);

  EXPECT_EQ(go, dlaf::common::Ordering::RowMajor);

  Cblacs_gridexit(contxt);
}

TEST(CAPIScaLAPACKTest, GridOrderColumnC) {
  int rank;
  int num_ranks;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  EXPECT_EQ(num_ranks, 6);

  int nprow = 2;  // Rows of process grid
  int npcol = 3;  // Cols of process grid

  EXPECT_EQ(nprow * npcol, num_ranks);

  char order = 'C';

  int contxt = 0;

  Cblacs_get(0, 0, &contxt);
  Cblacs_gridinit(&contxt, &order, nprow, npcol);

  int system_ctxt;
  // SGET_BLACSCONTXT == 10
  int get_blacs_contxt = 10;
  Cblacs_get(contxt, get_blacs_contxt, &system_ctxt);

  MPI_Comm communicator = Cblacs2sys_handle(system_ctxt);

  int dims[2] = {0, 0};
  int coords[2] = {-1, -1};

  Cblacs_gridinfo(contxt, &dims[0], &dims[1], &coords[0], &coords[1]);

  auto go = grid_order(communicator, dims[0], dims[1], coords[0], coords[1]);

  EXPECT_EQ(go, dlaf::common::Ordering::ColumnMajor);

  Cblacs_gridexit(contxt);
}

// TODO: Check double and float
TEST(CholeskyCAPIScaLAPACKTest, CorrectnessDistributed) {
  int rank;
  int num_ranks;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  EXPECT_EQ(num_ranks, 6);

  int n = 3;
  int m = 3;

  int nprow = 2;  // Rows of process grid
  int npcol = 3;  // Cols of process grid

  EXPECT_EQ(nprow * npcol, num_ranks);

  int nb = 1;
  int mb = 1;

  char order = 'C';
  char uplo = 'L';

  int contxt = 0;
  int contxt_global = 0;

  // Global matrix
  Cblacs_get(0, 0, &contxt);
  contxt_global = contxt;

  Cblacs_gridinit(&contxt_global, &order, 1, 1);  // Global matrix: only on rank 0
  Cblacs_gridinit(&contxt, &order, nprow, npcol);

  int myprow, mypcol;
  Cblacs_gridinfo(contxt, &nprow, &npcol, &myprow, &mypcol);

  int izero = 0;
  int m_local = numroc(&m, &mb, &myprow, &izero, &nprow);
  int n_local = numroc(&n, &nb, &mypcol, &izero, &npcol);

  // Global matrix (one copy on each rank)
  double* A;
  int descA[9] = {0, -1, 0, 0, 0, 0, 0, 0, 0};
  if (rank == 0) {
    A = new double[n * m];
    A[0] = 4.0;
    A[1] = 12.0;
    A[2] = -16.0;
    A[3] = 12.0;
    A[4] = 37.0;
    A[5] = -43.0;
    A[6] = -16.0;
    A[7] = -43.0;
    A[8] = 98.0;

    int info = -1;
    int lldA = m;
    descinit(descA, &m, &n, &m, &n, &izero, &izero, &contxt_global, &lldA, &info);
    ASSERT_EQ(info, 0);
    ASSERT_EQ(descA[0], 1);
  }

  auto a = new double[m_local * n_local];

  int desca[9];
  int info = -1;
  int llda = m_local;
  descinit(desca, &m, &n, &mb, &nb, &izero, &izero, &contxt, &llda, &info);
  ASSERT_EQ(info, 0);
  ASSERT_EQ(desca[0], 1);

  // Distribute global matrix to local matrices
  int ione = 1;
  pdgemr2d(&m, &n, A, &ione, &ione, descA, a, &ione, &ione, desca, &contxt);

  // Use EXPECT_EQ to avoid potential deadlocks!
  if (myprow == 0 && mypcol == 0) {
    EXPECT_EQ(m_local * n_local, 2);
    EXPECT_DOUBLE_EQ(a[0], 4.0);
    EXPECT_DOUBLE_EQ(a[1], -16.0);
  }
  if (myprow == 0 && mypcol == 1) {
    EXPECT_EQ(m_local * n_local, 2);
    EXPECT_DOUBLE_EQ(a[0], 12.0);
    EXPECT_DOUBLE_EQ(a[1], -43.0);
  }
  if (myprow == 0 && mypcol == 2) {
    EXPECT_EQ(m_local * n_local, 2);
    EXPECT_DOUBLE_EQ(a[0], -16.0);
    EXPECT_DOUBLE_EQ(a[1], 98.0);
  }
  if (myprow == 1 && mypcol == 0) {
    EXPECT_EQ(m_local * n_local, 1);
    EXPECT_DOUBLE_EQ(a[0], 12.0);
  }
  if (myprow == 1 && mypcol == 1) {
    EXPECT_EQ(m_local * n_local, 1);
    EXPECT_DOUBLE_EQ(a[0], 37.0);
  }
  if (myprow == 1 && mypcol == 2) {
    EXPECT_EQ(m_local * n_local, 1);
    EXPECT_DOUBLE_EQ(a[0], -43.0);
  }

  const char* argv[] = {"test_interface_", nullptr};
  dlaf_initialize(1, argv);
  dlaf_create_grid_from_blacs(contxt);

  info = -1;
  C_dlaf_pdpotrf(uplo, n, a, 1, 1, desca, &info);
  ASSERT_EQ(info, 0);

  // Gather local matrices into global one
  pdgemr2d(&m, &n, a, &ione, &ione, desca, A, &ione, &ione, descA, &contxt);

  if (rank == 0) {
    EXPECT_DOUBLE_EQ(A[0], 2.0);
    EXPECT_DOUBLE_EQ(A[1], 6.0);
    EXPECT_DOUBLE_EQ(A[2], -8.0);
    EXPECT_DOUBLE_EQ(A[3], 12.0);  // Upper: original value
    EXPECT_DOUBLE_EQ(A[4], 1.0);
    EXPECT_DOUBLE_EQ(A[5], 5.0);
    EXPECT_DOUBLE_EQ(A[6], -16.0);  // Upper: original value
    EXPECT_DOUBLE_EQ(A[7], -43.0);  // Upper: original value
    EXPECT_DOUBLE_EQ(A[8], 3.0);

    delete[] A;
  }

  dlaf_free_grid(contxt);
  dlaf_finalize();

  delete[] a;
  Cblacs_gridexit(contxt);
}

TEST(CholeskyCAPITest, CorrectnessDistributed) {
  int rank;
  int num_ranks;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  EXPECT_EQ(num_ranks, 6);

  int n = 3;
  int m = 3;

  int nprow = 2;  // Rows of process grid
  int npcol = 3;  // Cols of process grid

  EXPECT_EQ(nprow * npcol, num_ranks);

  int nb = 1;
  int mb = 1;

  char order = 'C';
  char uplo = 'L';

  int contxt = 0;
  int contxt_global = 0;

  // Global matrix
  Cblacs_get(0, 0, &contxt);
  contxt_global = contxt;

  Cblacs_gridinit(&contxt_global, &order, 1, 1);  // Global matrix: only on rank 0
  Cblacs_gridinit(&contxt, &order, nprow, npcol);

  // Get MPI_Comm
  // TODO: Re-use code from dlaf_create_grid_from_blacs?
  int system_context;
  int get_blacs_contxt = 10;
  Cblacs_get(contxt, get_blacs_contxt, &system_context);
  MPI_Comm comm = Cblacs2sys_handle(system_context);

  int myprow, mypcol;
  Cblacs_gridinfo(contxt, &nprow, &npcol, &myprow, &mypcol);

  int izero = 0;
  int m_local = numroc(&m, &mb, &myprow, &izero, &nprow);
  int n_local = numroc(&n, &nb, &mypcol, &izero, &npcol);

  // Global matrix (one copy on each rank)
  double* A;
  int descA[9] = {0, -1, 0, 0, 0, 0, 0, 0, 0};
  if (rank == 0) {
    A = new double[n * m];
    A[0] = 4.0;
    A[1] = 12.0;
    A[2] = -16.0;
    A[3] = 12.0;
    A[4] = 37.0;
    A[5] = -43.0;
    A[6] = -16.0;
    A[7] = -43.0;
    A[8] = 98.0;

    int info = -1;
    int lldA = m;
    descinit(descA, &m, &n, &m, &n, &izero, &izero, &contxt_global, &lldA, &info);
    ASSERT_EQ(info, 0);
    ASSERT_EQ(descA[0], 1);
  }

  auto a = new double[m_local * n_local];

  int desca[9];
  int info = -1;
  int llda = m_local;
  descinit(desca, &m, &n, &mb, &nb, &izero, &izero, &contxt, &llda, &info);
  ASSERT_EQ(info, 0);
  ASSERT_EQ(desca[0], 1);

  // Distribute global matrix to local matrices
  int ione = 1;
  pdgemr2d(&m, &n, A, &ione, &ione, descA, a, &ione, &ione, desca, &contxt);

  // Use EXPECT_EQ to avoid potential deadlocks!
  if (myprow == 0 && mypcol == 0) {
    EXPECT_EQ(m_local * n_local, 2);
    EXPECT_DOUBLE_EQ(a[0], 4.0);
    EXPECT_DOUBLE_EQ(a[1], -16.0);
  }
  if (myprow == 0 && mypcol == 1) {
    EXPECT_EQ(m_local * n_local, 2);
    EXPECT_DOUBLE_EQ(a[0], 12.0);
    EXPECT_DOUBLE_EQ(a[1], -43.0);
  }
  if (myprow == 0 && mypcol == 2) {
    EXPECT_EQ(m_local * n_local, 2);
    EXPECT_DOUBLE_EQ(a[0], -16.0);
    EXPECT_DOUBLE_EQ(a[1], 98.0);
  }
  if (myprow == 1 && mypcol == 0) {
    EXPECT_EQ(m_local * n_local, 1);
    EXPECT_DOUBLE_EQ(a[0], 12.0);
  }
  if (myprow == 1 && mypcol == 1) {
    EXPECT_EQ(m_local * n_local, 1);
    EXPECT_DOUBLE_EQ(a[0], 37.0);
  }
  if (myprow == 1 && mypcol == 2) {
    EXPECT_EQ(m_local * n_local, 1);
    EXPECT_DOUBLE_EQ(a[0], -43.0);
  }

  const char* argv[] = {"test_interface_", nullptr};
  dlaf_initialize(1, argv);
  // FIXME: Order 'C' insteaf of 'R'?
  int dlaf_context = dlaf_create_grid(comm, nprow, npcol, 'R');

  dlaf_cholesky_d(dlaf_context, uplo, a, {m, n, mb, nb, 0, 0, 1, 1, m_local});

  // Gather local matrices into global one
  pdgemr2d(&m, &n, a, &ione, &ione, desca, A, &ione, &ione, descA, &contxt);

  if (rank == 0) {
    EXPECT_DOUBLE_EQ(A[0], 2.0);
    EXPECT_DOUBLE_EQ(A[1], 6.0);
    EXPECT_DOUBLE_EQ(A[2], -8.0);
    EXPECT_DOUBLE_EQ(A[3], 12.0);  // Upper: original value
    EXPECT_DOUBLE_EQ(A[4], 1.0);
    EXPECT_DOUBLE_EQ(A[5], 5.0);
    EXPECT_DOUBLE_EQ(A[6], -16.0);  // Upper: original value
    EXPECT_DOUBLE_EQ(A[7], -43.0);  // Upper: original value
    EXPECT_DOUBLE_EQ(A[8], 3.0);

    delete[] A;
  }

  dlaf_free_grid(dlaf_context);
  dlaf_finalize();

  delete[] a;
  Cblacs_gridexit(contxt);
}