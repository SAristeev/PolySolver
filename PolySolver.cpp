#include "PolySolver.hpp"

void CreateSolvers(const json& settings, std::set<std::unique_ptr<LinearSolver>>& solvers) {

	ObjectSolverFactory<LinearSolver, SolverID> LinearFactory;
	InitLinearSolvers(LinearFactory);

	for (auto& solver : settings["LinearProblem"]["solvers"]) {
		if (polysolver::table.find(solver) != polysolver::table.end()) {
			solvers.emplace(LinearFactory.get(polysolver::table.at(solver)));
		}
		else {
			throw std::runtime_error("Can't find solver: " + solver);
		}
    }
}