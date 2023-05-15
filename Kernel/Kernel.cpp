#include"Kernel.h"
#include<iostream>

#include<map>
#include<exception>


namespace KERNEL {
#if defined(WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

	double second(void)
	{
		LARGE_INTEGER t;
		static double oofreq;
		static int checkedForHighResTimer;
		static BOOL hasHighResTimer;

		if (!checkedForHighResTimer) {
			hasHighResTimer = QueryPerformanceFrequency(&t);
			oofreq = 1.0 / (double)t.QuadPart;
			checkedForHighResTimer = 1;
		}
		if (hasHighResTimer) {
			QueryPerformanceCounter(&t);
			return (double)t.QuadPart * oofreq;
		}
		else {
			return (double)GetTickCount() / 1000.0;
		}
	}

#elif defined(__linux__) || defined(__QNX__)
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
	double second(void)
	{
		struct timeval tv;
		gettimeofday(&tv, NULL);
		return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
	}
#endif

	void InitLinearSolvers(SPARSE::ObjectSolverFactory<SPARSE::LinearSolver, SPARSE::SolverID>& LinearFactory) {
#ifdef USE_cuSOLVER
		LinearFactory.add<SPARSE::LinearSolvercuSOLVERSP>(SPARSE::SolverID::cuSOLVERSP);
		LinearFactory.add<SPARSE::LinearSolvercuSOLVERRF>(SPARSE::SolverID::cuSOLVERRF);
		LinearFactory.add<SPARSE::LinearSolvercuSOLVERRF_ALLGPU>(SPARSE::SolverID::cuSOLVERRF_ALLGPU);
#endif // USE_cuSOLVER
#ifdef USE_AMGX
		LinearFactory.add<SPARSE::LinearSolverAMGX>(SPARSE::SolverID::AMGX);
#endif // USE_AMGX
		LinearFactory.add<SPARSE::LinearSolverPARDISO>(SPARSE::SolverID::PARDISO);
	}
	
	ProblemCase::ProblemCase(std::string ConfigName) {
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
		try{
			for (auto solver : config["LinearProblem"]["solvers"]) {
				if (solver == "AMGX") {
					settings.AMGX_copies = static_cast<int>(config["LinearProblem"]["AMGX_settings"]["n_configs"]);
					for (int i = 0; i < settings.AMGX_copies; i++) {
						AddLinearImplementation(LinearSolvers, LinearFactory, solver);
						settings.solversName.push_back(solver);
					}
				}
				else{
					AddLinearImplementation(LinearSolvers, LinearFactory, solver);
					settings.solversName.push_back(solver);
				}				
			}
		}
		catch (const std::exception& ex){
			std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
			std::cerr << ex.what() << std::endl;
			throw std::exception("Invalid solver name");
		}
		try{
			settings.casePath = config["LinearProblem"]["case_path"];
		}
		catch (const std::exception& ex){
			std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
			std::cerr << ex.what() << std::endl;
			throw std::exception("Invalid case name");
		}
		try{
			for (auto caseName : config["LinearProblem"]["cases_names"]) {
				settings.casesNames.push_back(caseName);
			}
		}
		catch (const std::exception& ex) {
			std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
			std::cerr << ex.what() << std::endl;
			throw std::exception("Invalid case name");
		}
		
		try{
			settings.n_rhs = static_cast<int>(config["LinearProblem"]["n_rhs"]) - 1; // zero-indexing
		}
		catch (const std::exception& ex) {
			std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
			std::cerr << ex.what() << std::endl;
			std::cerr << "Setting default setting [LinearProblem][n_rhs] = 1" << std::endl;
			settings.n_rhs = 0;
		}

		try{
			settings.print_answer = config["LinearProblem"]["print_answer"];
		}
		catch(const std::exception& ex){
			std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
			std::cerr << ex.what() << std::endl;
			std::cerr << "Setting default setting [LinearProblem][print_answer] = false" << std::endl;
			settings.print_answer = false;
		}

		try{
			settings.print_time = config["LinearProblem"]["print_time"];
		}
		catch (const std::exception& ex)
		{
			std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
			std::cerr << ex.what() << std::endl;
			std::cerr << "Setting default setting [LinearProblem][print_time] = true" << std::endl;
			settings.print_time = true;
		}
		try{
			settings.check_answer = config["LinearProblem"]["check_answer"];
		}
		catch (const std::exception& ex){
			std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
			std::cerr << ex.what() << std::endl;
			std::cerr << "Setting default setting [LinearProblem][check_answer] = true" << std::endl;
			settings.check_answer = true;
		}
		try{
			settings.print_to_file = config["LinearProblem"]["print_to_file"];
		}
		catch (const std::exception& ex) {
			std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
			std::cerr << ex.what() << std::endl;
			std::cerr << "Setting default setting [LinearProblem][print_to_file] = true" << std::endl;
			settings.print_to_file = true;
		}
		try{ settings.resFileName = config["LinearProblem"]["results_file_name"]; 
		}
		catch (const std::exception& ex) {
			std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
			std::cerr << ex.what() << std::endl;
			std::cerr << "Setting default setting [LinearProblem][resFileName] = true" << std::endl;
			settings.resFileName = "res";
		}
	}


	void ProblemCase::start() {
		double start, stop;
		
		std::ofstream resFile(settings.casePath + "/" + settings.resFileName + ".csv");
		resFile  << "CaseName" << ", " << "SolverName" << ", " << "time";
		if (settings.check_answer) {
			resFile << ", ||Ax-b||L1, ||Ax-b||L2, ||Ax-b||L1, ";
			resFile << "||Ax-b||/||b||L1, ||Ax-b||/||b||L2, ||Ax-b||/||b||L1";
		}
		resFile << std::endl;

		for (auto caseName : settings.casesNames) {
			int curAMGXsolver = 0;
			A.freadCSR(settings.casePath + "/" + caseName + "/A.txt");
			b.AddData(settings.casePath + "/" + caseName + "/B.vec");
			for (auto solver : LinearSolvers) {
				if (solver.first->getName() == "AMGX") {
					if (caseName == *(settings.casesNames.begin())) {
						solver.first->SetCurConfig(curAMGXsolver);
						try {
							solver.first->AddConfigToName(config["LinearProblem"]["AMGX_settings"]["configs"][curAMGXsolver]);
						}
						catch (const std::exception& ex)
						{
							std::cerr << ex.what() << std::endl;
							std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
							throw std::exception("Invalid AMGX configname");
						}
					}
					curAMGXsolver++;
				}
				solver.first->SetSettingsFromJSON(config["LinearProblem"]);
				solver.first->PrepareToSolveByMatrix(A, b, x);

				start = second();
				solver.first->SolveRightSide(A, b, x);
				stop = second();

				settings.time.push_back(stop - start);
				if (settings.print_answer) {
					int n = 0, nrhs = 0;
					x.GetInfo(n, nrhs);
					for (int i = 0; i < nrhs; i++) {
						x.fprint(i, settings.casePath + "/" + settings.caseName + "/X_" + std::to_string(i) + "_" + solver.first->getName() + ".vec");
					}
				}

				if (settings.print_to_file) {
					resFile << caseName << ", " << solver.first->getName() << ", " << stop - start;
					if (settings.check_answer) {
						double absnorm1, absnorm2, absnorminf;
						double relnorm1, relnorm2, relnorminf;
						Check(absnorm1, absnorm2, absnorminf, relnorm1, relnorm2, relnorminf);
						resFile << ", " << absnorm1 << ", ";
						resFile << absnorm2 << ", ";
						resFile << absnorminf << ", ";
						resFile << relnorm1 << ", ";
						resFile << relnorm2 << ", ";
						resFile << relnorminf;
					}
					resFile << std::endl;
				}
				x.Clear();
			}
			A.Clear();
			b.Clear();
		}
		
	}

	void ProblemCase::Check(double& absnorm1, double& absnorm2, double& absnorminf, double& relnorm1, double& relnorm2, double& relnorminf) {
		int n, nnzA;
		int* h_RowsA = nullptr; // GPU <int>    n+1
		int* h_ColsA = nullptr; // GPU <int>    nnzA
		double* h_ValsA = nullptr; // GPU <double> nnzA 
		double* h_x = nullptr; // GPU <double> n
		double* h_b = nullptr; // GPU <double> n
		double* h_r = nullptr; // GPU <double> n

		A.GetInfo(n, nnzA);
		A.GetDataCSR(&h_ValsA, &h_RowsA, &h_ColsA);
		b.GetData(&h_b);
		x.GetData(&h_x);

		const double alpha = 1.0;
		const double beta = -1.0;
		const char trans = 'N';
		char matdesra[6];
		matdesra[0] = 'G';
		matdesra[3] = 'C';
		h_r = (double*)malloc(n * sizeof(double));
		memcpy(h_r, h_b, n * sizeof(double));
		int* pointerB = (int*)malloc(n * sizeof(int));
		int* pointerE = (int*)malloc(n * sizeof(int));
		for (size_t i = 0; i < n; i++) {
			pointerB[i] = h_RowsA[i];
			pointerE[i] = h_RowsA[i + 1];
		}

		mkl_dcsrmv(&trans, &n, &n, &alpha, matdesra, h_ValsA, h_ColsA, pointerB, pointerE, h_x, &beta, h_r);

		absnorm1 = 0, absnorm2 = 0, absnorminf = 0;
		for (int i = 0; i < n; i++) {
			//h_r[i] = _distafter[i] - _distbefore[i];
			absnorm1 += abs(h_r[i]);
			absnorm2 += h_r[i] * h_r[i];
			absnorminf = abs(h_r[i]) > absnorminf ? abs(h_r[i]) : absnorminf;
		}
		absnorm2 = sqrt(absnorm2);

		relnorm1 = 0, relnorm2 = 0, relnorminf = 0;
		for (int i = 0; i < n; i++) {
			//h_r[i] = _distafter[i] - _distbefore[i];
			relnorm1 += abs(h_b[i]);
			relnorm2 += h_b[i] * h_b[i];
			relnorminf = abs(h_b[i]) > relnorminf ? abs(h_b[i]) : relnorminf;
		}
		relnorm2 = sqrt(relnorm2);

		relnorm1 = absnorm1 / relnorm1;
		relnorm2 = absnorm2 / relnorm2;
		relnorminf = absnorminf / relnorminf;

		free(h_r);
		free(pointerE);
		free(pointerB);
	}
}
