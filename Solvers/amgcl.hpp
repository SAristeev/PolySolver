#pragma once
#include "implementation.hpp"

#include <amgcl/backend/builtin.hpp>
#include <amgcl/backend/mkl.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/coarsening/ruge_stuben.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/spai1.hpp>
#include <amgcl/relaxation/ilu0.hpp>

#include <amgcl/solver/fgmres.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/profiler.hpp>


class LinearSolverAMGCL : public LinearSolver {
public:
	int Solve(const SPARSE::SparseMatrix<MKL_INT, double>& A,
		const SPARSE::SparseVector<double>& b,
		SPARSE::SparseVector<double>& x
	);
};
