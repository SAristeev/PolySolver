#include "implementation.hpp"
#include "pardiso.hpp"
#include "amgcl.hpp"
#include "my_cg.hpp"
#include "my_pcg.hpp"
#include "cuda_cg_ic0.hpp"
void InitLinearSolvers(ObjectSolverFactory<LinearSolver, SolverID>& LinearFactory) {
	LinearFactory.add<LinearSolver_PARDISO>(SolverID::ePARDISO);
	LinearFactory.add<LinearSolver_AMGCL>(SolverID::eAMGCL);
	LinearFactory.add<LinearSolver_my_cg>(SolverID::eMY_CG);
	LinearFactory.add<LinearSolver_my_pcg>(SolverID::eMY_PCG);
	LinearFactory.add<LinearSolver_cuda_cg_ic0>(SolverID::eCUDA_CG_CI0);
}