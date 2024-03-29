#pragma once
#include "implementation.hpp"

class LinearSolver_cuda_cg : public LinearSolver {
public:
	int Solve(const std::vector<double>& vals,
		const std::vector<MKL_INT>& cols,
		const std::vector<MKL_INT>& rows,
		const std::vector<double>& b,
		std::vector<double>& x
	);
};
