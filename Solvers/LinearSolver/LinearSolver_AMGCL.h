#pragma once
#include "LinearSolver_IMPL.h"

#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstab.hpp>

#include <amgcl/io/mm.hpp>
#include <amgcl/profiler.hpp>

namespace SOLVER {
	class LinearSolverAMGCL_32_d : public LinearSolver {

	public:
		LinearSolverAMGCL_32_d() : LinearSolver("AMGCL") {};
		int Solve(const SPARSE::Case<int, double>& rhs);
	};
}