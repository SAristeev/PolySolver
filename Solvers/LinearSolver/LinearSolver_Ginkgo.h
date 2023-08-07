#pragma once
#include "LinearSolver_IMPL.h"
#include <ginkgo\ginkgo.hpp>

namespace SPARSE {

	void check() {
		const auto exec = gko::CudaExecutor::create();
	}

}