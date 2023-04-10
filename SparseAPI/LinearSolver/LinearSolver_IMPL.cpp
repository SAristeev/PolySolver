#include "LinearSolver_IMPL.h"
#include <unordered_map>
namespace SPARSE {

	int LinearSolver::IsReadyToSolve() {
		return 1;
	}

	int LinearSolver::PrepareToSolveByMatrix(SparseMatrix A,
		SparseVector b,
		SparseVector x) {

		return IsReadyToSolve();
	}
	void AddLinearImplementation(std::map<LinearSolver*, SolverID>& LinearSolvers, ObjectSolverFactory<LinearSolver, SolverID> &LinearFactory, std::string solver) {
		static std::unordered_map<std::string, SolverID> const table = { {"cuSOLVER",SolverID::cuSOLVERSP}, {"AMGX",SolverID::AMGX}, {"PARDISO",SolverID::PARDISO} };
		auto it = table.find(solver);
		if (it == table.end()) {
			throw std::exception(("Can't find solver: " + solver).c_str());
		}
		SolverID SID;
		if (it != table.end()) {
			SID = it->second;
		}
		LinearSolvers.insert({ LinearFactory.get(SID), SID });
	}

}