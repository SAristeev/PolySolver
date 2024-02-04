#include "implementation.hpp"
#include "pardiso.hpp"
#include "amgcl.hpp"
#include "mkl_cg.hpp"
#include "mkl_pcg.hpp"
#include "cuda_cg.hpp"
#include "cuda_pcg.hpp"
void InitLinearSolvers(ObjectSolverFactory<LinearSolver, SolverID>& LinearFactory) {
	LinearFactory.add<LinearSolver_PARDISO>(SolverID::ePARDISO);
	LinearFactory.add<LinearSolver_AMGCL>(SolverID::eAMGCL);
	LinearFactory.add<LinearSolver_mkl_cg>(SolverID::eMKL_CG);
	LinearFactory.add<LinearSolver_mkl_pcg>(SolverID::eMKL_PCG);
	LinearFactory.add<LinearSolver_cuda_cg>(SolverID::eCUDA_CG);
	LinearFactory.add<LinearSolver_cuda_pcg>(SolverID::eCUDA_PCG);
}