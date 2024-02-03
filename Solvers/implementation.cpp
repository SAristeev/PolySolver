#include "implementation.hpp"
#include "pardiso.hpp"
#include "amgcl.hpp"
void InitLinearSolvers(ObjectSolverFactory<LinearSolver, SolverID>& LinearFactory) {
	LinearFactory.add<LinearSolverPARDISO>(SolverID::ePARDISO);
	LinearFactory.add<LinearSolverAMGCL>(SolverID::eAMGCL);
}