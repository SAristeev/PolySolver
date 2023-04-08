#pragma once
#include "LinearSolver_IMPL.h"
#include <mkl.h>

namespace SPARSE {

	class LinearSolverPARDISO : public LinearSolver {
		MKL_INT _iparm[64];
		MKL_INT _pt[64];
	public:
		LinearSolverPARDISO() : LinearSolver("PARDISO") {}
		virtual int SolveRightSide(SparseMatrix& A,
			SparseVector& b,
			SparseVector& x) final;
		virtual int SetSettingsFromJSON(json settingsJSON) final { nrhs = settingsJSON["n_rhs"]; return 0; };
		void initParams();
	};
}
