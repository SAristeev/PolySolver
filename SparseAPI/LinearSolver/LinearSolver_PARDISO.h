#pragma once
#include "LinearSolver_IMPL.h"
#include <mkl.h>

namespace SPARSE {

	class LinearSolverPARDISO : public LinearSolver {
		int		n;
		int		nnzA;
		int* h_RowsA = nullptr; // CPU <int>    n+1
		int* h_ColsA = nullptr; // CPU <int>    nnzA
		double* h_ValsA = nullptr; // CPU <double> nnzA 
		double* h_x = nullptr; // CPU <double> n		
		double* h_b = nullptr; // CPU <double> n
	public:
		LinearSolverPARDISO() : LinearSolver("PARDISO") {}
		virtual int SolveRightSide(SparseMatrix& A,
			SparseVector& b,
			SparseVector& x) final;
		virtual int SetSettingsFromJSON(json settings) final { return 0; };
	};
}
