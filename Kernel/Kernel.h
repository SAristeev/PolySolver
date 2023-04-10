#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <mkl.h>
#include "json.hpp"
#include "../SparseAPI/SparseAPI.h"
#include "../SparseAPI/LinearSolver/LinearSolver_IMPL.h"
#ifdef USE_cuSOLVER
#include "../SparseAPI/LinearSolver/LinearSolver_cuSOLVER.h"
#endif // USE_cuSOLVER
#ifdef USE_AMGX
#include "../SparseAPI/LinearSolver/LinearSolver_AMGX.h"
#endif // USE_AMGX
#include "../SparseAPI/LinearSolver/LinearSolver_PARDISO.h"




namespace KERNEL {
	double second();

	using json = nlohmann::json;
	void InitLinearSolvers(SPARSE::ObjectSolverFactory<SPARSE::LinearSolver, SPARSE::SolverID> &LinearFactory);
	void AddLinearImplementation(std::map<SPARSE::LinearSolver*, SPARSE::SolverID> &LinearSolvers, std::string solver);
	//void AddEigenImplementation(SPARSE::ObjectSolverFactory<SPARSE::LinearSolver, SPARSE::SolverID>& LinearFactory);
	
	struct Settings {
		std::string caseName;
		std::string casePath;
		std::string resFileName;
		std::vector<std::string> solversName;
		std::vector<std::string> casesNames;

		int n_rhs;
		std::vector<double> time;
		int AMGX_copies = 0;

		double absnorm1, absnorm2, absnorminf;
		double relnorm1, relnorm2, relnorminf;

		bool print_answer;
		bool print_time;
		bool print_to_file;
		bool check_answer;

	};

	class ProblemCase
	{
	private:
		json config;
		SPARSE::SparseMatrix A;
		SPARSE::SparseVector b;
		SPARSE::SparseVector x;
		SPARSE::SparseVector r;
		std::map<SPARSE::LinearSolver*, SPARSE::SolverID> LinearSolvers;
		SPARSE::ObjectSolverFactory<SPARSE::LinearSolver, SPARSE::SolverID> LinearFactory;
		Settings settings;
	public:
		ProblemCase(std::string CN);
		void start();
		void Check(double& absnorm1, double& absnorm2, double& absnorminf, double& relnorm1, double& relnorm2, double& relnorminf);
	};
}