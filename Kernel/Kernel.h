#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <map>
#include "json.hpp"
#include "../SparseAPI/SparseAPI.h"
#include "../SparseAPI/LinearSolver/LinearSolver_IMPL.h"
#include "../SparseAPI/LinearSolver/LinearSolver_AMGX.h"
#include "../SparseAPI/LinearSolver/LinearSolver_cuSOLVER.h"



//class Config {
//private:
//	std::vector<SPARSE::SolverID> solvers;
//	std::string CaseName;
//	double abstolerance = 1e-6;
//	double reltolerance = 1e-6;
//	bool needTime = false;
//	bool needPrint = false;
//
//public:
//	Config(std::initializer_list<SPARSE::SolverID> list, std::string CN) : solvers(list) { CaseName = CN; }
//	Config(SPARSE::SolverID id, std::string CN) {solvers.push_back(id); CaseName = CN;}
//	void SetAbsTol(double tol) { abstolerance = tol; }
//	void SetRelTol(double tol) { reltolerance = tol; }
//	void SetTime(bool time) { needTime = time; }
//	void SetPrint(bool print) { needPrint = print; }
//	
//	std::vector<SPARSE::SolverID>::iterator GetFirstID() { return solvers.begin(); }
//	std::vector<SPARSE::SolverID>::iterator GetLastID() { return solvers.end(); }
//
//};
namespace KERNEL {
	double second();

	using json = nlohmann::json;
	void InitLinearSolvers(SPARSE::ObjectSolverFactory<SPARSE::LinearSolver, SPARSE::SolverID> &LinearFactory);
	void AddLinearImplementation(std::map<SPARSE::LinearSolver*, SPARSE::SolverID> &LinearSolvers, std::string solver);
	//void AddEigenImplementation(SPARSE::ObjectSolverFactory<SPARSE::LinearSolver, SPARSE::SolverID>& LinearFactory);
	
	

	class ProblemCase
	{
	private:
		json config;
		std::string CaseName;
		SPARSE::SparseMatrix A;
		SPARSE::SparseVector b;
		SPARSE::SparseVector x;
		std::map<SPARSE::LinearSolver*, SPARSE::SolverID> LinearSolvers;
		SPARSE::ObjectSolverFactory<SPARSE::LinearSolver, SPARSE::SolverID> LinearFactory;
	public:
		ProblemCase(std::string CN, SPARSE::SolverID SID);
		ProblemCase(std::string CN);
		void loadconfig(std::string CN) {
			std::ifstream file(CN);
			config = json::parse(file);
			for (auto x : config["LinearProblem"]["solvers"]) {
				std::cout << x << std::endl;
			}
		}
		void start(double& time) {

			double start, stop;
			for (auto solver : LinearSolvers) {
				start = second();
				solver.first->SolveRightSide(A, b, x);
				stop = second();
				std::string SolverName = SPARSE::SolverID2String(solver.second);
				std::string destPath = "C:/WorkDirectory/Cases/X_";
				if (1) {
					x.fprint(0, destPath + SolverName + ".txt");
				}
			}
			time = stop - start;
		}

		void Check(double& absnorm1, double& absnorm2, double& absnorminf, double& relnorm1, double& relnorm2, double& relnorminf);
		void print();
	};



}