// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <dlaf/interface/utils.h>

#include <dlaf/init.h>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/program_options.hpp>
#include <pika/runtime.hpp>

namespace dlaf::interface::utils{

extern "C" void dlafuture_init(int argc, char** argv){
  if(!initialized){
    pika::program_options::options_description desc("");
    desc.add(dlaf::getOptionsDescription());

    // pika initialization
    pika::init_params params;
    params.rp_callback = dlaf::initResourcePartitionerHandler;
    params.desc_cmdline = desc;
    pika::start(nullptr, argc, argv, params);

    // DLA-Future initialization
    dlaf::initialize(argc, argv);
    initialized = true;

    pika::suspend();
  }
}

extern "C" void dlafuture_finalize(){
  pika::resume();
  pika::finalize();
  dlaf::finalize();
  pika::stop();

  initialized = false;
}

void dlaf_check(char uplo, int* desc, int& info) {
  if (uplo != 'U' && uplo != 'u' && uplo != 'L' && uplo != 'l') {
    info = -1;
    std::cerr << "ERROR: The UpLo parameter has a incorrect value: '" << uplo;
    std::cerr << "'. Please check the ScaLAPACK documentation.\n";
  } 

  if (desc[0] != 1) {
    info = -1;
    std::cerr << "ERROR: DLA-Future can only treat dense matrices.\n";
  }

  if (!initialized) {
    info = -1;
    std::cerr << "Error: DLA-Future must be initialized.\n";
  }
}



}
