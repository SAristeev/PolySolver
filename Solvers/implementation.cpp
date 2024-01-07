#include "implementation.hpp"
#include "pardiso.hpp"
void InitLinearSolvers(ObjectSolverFactory<LinearSolver, SolverID>& LinearFactory) {
	LinearFactory.add<LinearSolverPARDISO>(SolverID::sPARDISO);
}