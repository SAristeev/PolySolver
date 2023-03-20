#include "LinearSolver_IMPL.h"
namespace SPARSE {

	std::string SolverID2String(SolverID ID) {
		switch (ID)
		{
		case SPARSE::SolverID::cuSOLVERSP:
			return std::string("cuSOLVERSP");
		case SPARSE::SolverID::cuSOLVERRF:
			return std::string("cuSOLVERRF");
		case SPARSE::SolverID::cuSOLVERRF0:
			return std::string("cuSOLVERRF0");
		case SPARSE::SolverID::cuSOLVERRF_ALLGPU:
			return std::string("cuSOLVERRF_ALLGPU");
		case SPARSE::SolverID::AMGX:
			return std::string("AMGX");
		}
	}



	int LinearSolver::IsReadyToSolve() {
		return 1;
	}

	int LinearSolver::IsReadyToSolveByMatrix(SparseMatrix A,
		SparseVector b,
		SparseVector x) {
		return IsReadyToSolve();
	}
}