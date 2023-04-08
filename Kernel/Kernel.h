#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <mkl.h>
#include "json.hpp"
#include "../SparseAPI/SparseAPI.h"
#include "../SparseAPI/LinearSolver/LinearSolver_IMPL.h"
#include "../SparseAPI/LinearSolver/LinearSolver_AMGX.h"
#include "../SparseAPI/LinearSolver/LinearSolver_cuSOLVER.h"
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

		int n_rhs;
		std::vector<double> time;
		//double time;

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
		//ProblemCase(std::string CN, SPARSE::SolverID SID);
		ProblemCase(std::string CN);
		// TODO: create setSettings()
		// void setSettings();
		//void start(double& time);
		void start();
		//void start();
		void Check(double& absnorm1, double& absnorm2, double& absnorminf, double& relnorm1, double& relnorm2, double& relnorminf);
		void CheckMKL(double& absnorm1, double& absnorm2, double& absnorminf, double& relnorm1, double& relnorm2, double& relnorminf);
		void print();
	};
}