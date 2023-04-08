#pragma once
#include "LinearSolver_IMPL.h"
#include <amgx_c.h>
#include <amgx_config.h>
#include <cuda_runtime.h>


namespace SPARSE {
	struct Settings_AMGX{
		std::string configAMGX;
		uint64_t max_iter;
		double tolerance;
	};
	class LinearSolverAMGX : public LinearSolver {
		Settings_AMGX settings;
	public:
		LinearSolverAMGX(): LinearSolver("AMGX") {}
		int SolveRightSide(SparseMatrix& A,
			SparseVector& b,
			SparseVector& x) final;
		virtual int SetSettingsFromJSON(json settings) final;
	};
	
}