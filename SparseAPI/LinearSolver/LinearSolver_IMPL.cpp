#include "LinearSolver_IMPL.h"
#include <unordered_map>
namespace SPARSE {

	

	/*std::string SolverID2String(SolverID ID) {
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
	}*/

	int LinearSolver::IsReadyToSolve() {
		return 1;
	}

	int LinearSolver::IsReadyToSolveByMatrix(SparseMatrix A,
		SparseVector b,
		SparseVector x) {
		return IsReadyToSolve();
	}
	void AddLinearImplementation(std::map<LinearSolver*, SolverID>& LinearSolvers, ObjectSolverFactory<LinearSolver, SolverID> &LinearFactory, std::string solver) {
		static std::unordered_map<std::string, SolverID> const table = { {"cuSOLVER",SolverID::cuSOLVERSP}, {"AMGX",SolverID::AMGX}, {"PARDISO",SolverID::PARDISO} };
		auto it = table.find(solver);
		SolverID SID;
		if (it != table.end()) {
			SID = it->second;
		}
		LinearSolvers.insert({ LinearFactory.get(SID), SID });
	}

}