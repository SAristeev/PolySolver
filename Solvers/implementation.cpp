#include "implementation.hpp"
#include "pardiso.hpp"
#include "amgcl.hpp"
void InitLinearSolvers(ObjectSolverFactory<LinearSolver, SolverID>& LinearFactory) {
	LinearFactory.add<LinearSolver_PARDISO>(SolverID::ePARDISO);
	LinearFactory.add<LinearSolver_AMGCL>(SolverID::eAMGCL);
}