#pragma once
#include "LinearSolver_IMPL.h"
#include <amgx_c.h>
#include <amgx_config.h>
#include <cuda_runtime.h>


namespace SPARSE {
	struct Settings_AMGX{
		int n_configs;
		std::vector<std::string> configsAMGX;
		std::string configs_path;
		int max_iter;
		double tolerance;
	};

	class LinearSolverAMGX : public LinearSolver {
		Settings_AMGX settings;
	public:
		LinearSolverAMGX(): LinearSolver("AMGX") {}
		int SolveRightSide(SparseMatrix<int, double>& A,
			SparseVector<double>& b,
			SparseVector<double>& x) final;
		virtual int SetSettingsFromJSON(json settings) final;
	};
	
}

namespace SOLVER {
	class LinearSolverAMGX_32_d : public LinearSolver {

	public:
		LinearSolverAMGX_32_d() : LinearSolver("AMGX") {};
		int Solve(const SPARSE::Case<int, double>& rhs);
	};
}