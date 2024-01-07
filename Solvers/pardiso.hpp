#pragma once
#include "implementation.hpp"



class LinearSolverPARDISO : public LinearSolver {
	MKL_INT _iparm[64];
	void* _pt[64];
public:
	void initParams();
	int Solve(const SPARSE::SparseMatrix<MKL_INT, double>& A,
		const SPARSE::SparseVector<double>& b,
		SPARSE::SparseVector<double>& x
	);
};
