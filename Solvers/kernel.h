#pragma once
import Sparse;
#include <string>
#include <vector>
#include <iostream>
#include <map>
#include "json.hpp"

#include "LinearSolver/LinearSolver_IMPL.h"
#ifdef USE_cuSOLVER
#include "LinearSolver/LinearSolver_cuSOLVER.h"
#endif // USE_cuSOLVER
#ifdef POLYSOLVER_USE_AMGX
#include "LinearSolver/LinearSolver_AMGX.h"
#endif // POLYSOLVER_USE_AMGX

#ifdef POLYSOLVER_USE_AMGCL
#include "LinearSolver/LinearSolver_AMGCL.h"
#endif // POLYSOLVER_USE_AMGCL

#ifdef POLYSOLVER_USE_MKL
#include "LinearSolver/LinearSolver_PARDISO.h"
#endif // POLYSOLVER_USE_MKL




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

	template<class IT, class VT>
	requires SPARSE::Matrix<IT, VT>
	class ProblemCase
	{
	private:
		json config;
		SPARSE::SparseMatrix<IT, VT> A;
		SPARSE::SparseVector<VT> b;
		SPARSE::SparseVector<VT> x;

		std::map<SPARSE::LinearSolver*, SPARSE::SolverID> LinearSolvers;
		SPARSE::ObjectSolverFactory<SPARSE::LinearSolver, SPARSE::SolverID> LinearFactory;
		Settings settings;
	public:
		void setConfig(std::string ConfigName);
		void start();
		void Check(double& absnorm1, double& absnorm2, double& absnorminf, double& relnorm1, double& relnorm2, double& relnorminf);
	};

	template<class IT, class VT>
	requires SPARSE::Matrix<IT, VT>
	void ProblemCase<IT, VT>::setConfig(std::string ConfigName) {
		std::ifstream file(ConfigName);
		try {
			config = json::parse(file);
		}
		catch (const std::exception& ex) {
			std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
			std::cerr << ex.what() << std::endl;
			throw std::exception("Can't open file");
		}

		InitLinearSolvers(LinearFactory);
		try {
			for (auto solver : config["LinearProblem"]["solvers"]) {
				if (solver == "AMGX") {
					settings.AMGX_copies = static_cast<int>(config["LinearProblem"]["AMGX_settings"]["n_configs"]);
					for (int i = 0; i < settings.AMGX_copies; i++) {
						AddLinearImplementation(LinearSolvers, LinearFactory, solver);
						settings.solversName.push_back(solver);
					}
				}
				else {
					AddLinearImplementation(LinearSolvers, LinearFactory, solver);
					settings.solversName.push_back(solver);
				}
			}
		}
		catch (const std::exception& ex) {
			std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
			std::cerr << ex.what() << std::endl;
			throw std::exception("Invalid solver name");
		}
		try {
			settings.casePath = config["LinearProblem"]["case_path"];
		}
		catch (const std::exception& ex) {
			std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
			std::cerr << ex.what() << std::endl;
			throw std::exception("Invalid case name");
		}
		try {
			for (auto caseName : config["LinearProblem"]["cases_names"]) {
				settings.casesNames.push_back(caseName);
			}
		}
		catch (const std::exception& ex) {
			std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
			std::cerr << ex.what() << std::endl;
			throw std::exception("Invalid case name");
		}

		try {
			settings.n_rhs = static_cast<int>(config["LinearProblem"]["n_rhs"]) - 1; // zero-indexing
		}
		catch (const std::exception& ex) {
			std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
			std::cerr << ex.what() << std::endl;
			std::cerr << "Setting default setting [LinearProblem][n_rhs] = 1" << std::endl;
			settings.n_rhs = 0;
		}

		try {
			settings.print_answer = config["LinearProblem"]["print_answer"];
		}
		catch (const std::exception& ex) {
			std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
			std::cerr << ex.what() << std::endl;
			std::cerr << "Setting default setting [LinearProblem][print_answer] = false" << std::endl;
			settings.print_answer = false;
		}

		try {
			settings.print_time = config["LinearProblem"]["print_time"];
		}
		catch (const std::exception& ex)
		{
			std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
			std::cerr << ex.what() << std::endl;
			std::cerr << "Setting default setting [LinearProblem][print_time] = true" << std::endl;
			settings.print_time = true;
		}
		try {
			settings.check_answer = config["LinearProblem"]["check_answer"];
		}
		catch (const std::exception& ex) {
			std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
			std::cerr << ex.what() << std::endl;
			std::cerr << "Setting default setting [LinearProblem][check_answer] = true" << std::endl;
			settings.check_answer = true;
		}
		try {
			settings.print_to_file = config["LinearProblem"]["print_to_file"];
		}
		catch (const std::exception& ex) {
			std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
			std::cerr << ex.what() << std::endl;
			std::cerr << "Setting default setting [LinearProblem][print_to_file] = true" << std::endl;
			settings.print_to_file = true;
		}
		try {
			settings.resFileName = config["LinearProblem"]["results_file_name"];
		}
		catch (const std::exception& ex) {
			std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
			std::cerr << ex.what() << std::endl;
			std::cerr << "Setting default setting [LinearProblem][resFileName] = true" << std::endl;
			settings.resFileName = "res";
		}
	}

	template<class IT, class VT>
	requires SPARSE::Matrix<IT, VT>
	void ProblemCase<IT, VT>::start() {
		double start, stop;

		//std::ofstream resFile(settings.casePath + "/" + settings.resFileName + ".csv");
		//resFile << "CaseName" << ", " << "SolverName" << ", " << "time";
		//if (settings.check_answer) {
		//	resFile << ", ||Ax-b||L1, ||Ax-b||L2, ||Ax-b||L1, ";
		//	resFile << "||Ax-b||/||b||L1, ||Ax-b||/||b||L2, ||Ax-b||/||b||L1";
		//}
		//resFile << std::endl;

		//for (auto caseName : settings.casesNames) {
		//	int curAMGXsolver = 0;
		//	A.freadCSR(settings.casePath + "/" + caseName + "/A.txt");
		//	b.AddVector(settings.casePath + "/" + caseName + "/B.vec");
		//	for (auto solver : LinearSolvers) {
		//		if (solver.first->getName() == "AMGX") {
		//			if (caseName == *(settings.casesNames.begin())) {
		//				solver.first->SetCurConfig(curAMGXsolver);
		//				try {
		//					solver.first->AddConfigToName(config["LinearProblem"]["AMGX_settings"]["configs"][curAMGXsolver]);
		//				}
		//				catch (const std::exception& ex)
		//				{
		//					std::cerr << ex.what() << std::endl;
		//					std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
		//					throw std::exception("Invalid AMGX configname");
		//				}
		//			}
		//			curAMGXsolver++;
		//		}
		//		solver.first->SetSettingsFromJSON(config["LinearProblem"]);
		//		solver.first->PrepareToSolveByMatrix(A, b, x);

		//		start = second();
		//		solver.first->SolveRightSide(A, b, x);
		//		stop = second();

		//		settings.time.push_back(stop - start);
		//		if (settings.print_answer) {
		//			int n = 0, nrhs = 0;
		//			x.GetInfo(n, nrhs);
		//			for (int i = 0; i < nrhs; i++) {
		//				x.fprint(i, settings.casePath + "/" + settings.caseName + "/X_" + std::to_string(i) + "_" + solver.first->getName() + ".vec");
		//			}
		//		}

		//		//if (settings.print_to_file) {
		//		//	resFile << caseName << ", " << solver.first->getName() << ", " << stop - start;
		//		//	if (settings.check_answer) {
		//		//		double absnorm1, absnorm2, absnorminf;
		//		//		double relnorm1, relnorm2, relnorminf;
		//		//		//Check(absnorm1, absnorm2, absnorminf, relnorm1, relnorm2, relnorminf);
		//		//		resFile << ", " << absnorm1 << ", ";
		//		//		resFile << absnorm2 << ", ";
		//		//		resFile << absnorminf << ", ";
		//		//		resFile << relnorm1 << ", ";
		//		//		resFile << relnorm2 << ", ";
		//		//		resFile << relnorminf;
		//		//	}
		//		//	resFile << std::endl;
		//		//}
		//		//x.Clear();
		//	}
		//	//A.Clear();
		//	//b.Clear();
		//}
	}
}

namespace KERNEL_1 {
	
}
