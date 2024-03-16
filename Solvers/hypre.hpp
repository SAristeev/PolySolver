#pragma once
#include "implementation.hpp"

#include <_hypre_utilities.h>
#include <HYPRE_krylov.h>
#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>

class LinearSolver_hypre : public LinearSolver {
public:
	void initParams();
	int Solve(const std::vector<double>& vals,
		const std::vector<MKL_INT>& cols,
		const std::vector<MKL_INT>& rows,
		const std::vector<double>& b,
		std::vector<double>& x
	);
};
