#pragma once
#include "LinearSolver_IMPL.h"
//#include <vector>
#include <amgx_c.h>
#include <amgx_config.h>
#include <cuda_runtime.h>


namespace SPARSE {
	struct Settings_AMGX{
		int n_configs;
		//std::vector<std::string> configsAMGX;
		std::string configAMGX;
		int max_iter;
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