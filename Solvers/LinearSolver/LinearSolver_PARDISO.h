#pragma once
#include "LinearSolver_IMPL.h"
#include <mkl.h>


namespace SOLVER {

	class LinearSolverPARDISO_32_d : public LinearSolver {
		MKL_INT _iparm[64];
		void* _pt[64];
	public:
		LinearSolverPARDISO_32_d() : LinearSolver("PARDISO") {};
		void initParams();
		int Solve(const SPARSE::Case<int, double> & rhs);
	};

	class LinearSolverPARDISO_64_d : public LinearSolver {
		long long int _iparm[64];
		void* _pt[64];
	public:
		LinearSolverPARDISO_64_d() : LinearSolver("PARDISO") {};
		void initParams();
		int Solve(const SPARSE::Case<long long int, double>& rhs);
	};
}
