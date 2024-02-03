#pragma once
#include "implementation.hpp"



class LinearSolverPARDISO : public LinearSolver {
	MKL_INT _iparm[64];
	void* _pt[64];
public:
	void initParams();
	int Solve(const std::vector<double>& vals,
		const std::vector<MKL_INT>& cols,
		const std::vector<MKL_INT>& rows,
		const std::vector<double>& b,
		std::vector<double>& x
	);
};
