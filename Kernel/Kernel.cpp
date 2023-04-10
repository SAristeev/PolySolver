#include"Kernel.h"
#include<iostream>

#include<map>
#include<cusparse.h>
#include<exception>


namespace KERNEL {
//#define SETTINGS_FROM_JSON(s) try{setting.s = }
#if defined(_WIN32)
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
		LinearFactory.add<SPARSE::LinearSolvercuSOLVERSP>(SPARSE::SolverID::cuSOLVERSP);
		LinearFactory.add<SPARSE::LinearSolvercuSOLVERRF>(SPARSE::SolverID::cuSOLVERRF);
		LinearFactory.add<SPARSE::LinearSolvercuSOLVERRF_ALLGPU>(SPARSE::SolverID::cuSOLVERRF_ALLGPU);
		LinearFactory.add<SPARSE::LinearSolverAMGX>(SPARSE::SolverID::AMGX);
		LinearFactory.add<SPARSE::LinearSolverPARDISO>(SPARSE::SolverID::PARDISO);
	}
	
	void setSettingsFromJSON(json config);

	
	ProblemCase::ProblemCase(std::string ConfigName) {
		//TODO: create individual settings
		std::ifstream file(ConfigName);
		try {
			config = json::parse(file);
		}
		catch (const std::exception& ex) {
			std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
			std::cerr << ex.what() << std::endl;
			throw std::exception("Cannot open file");
		}
		
		InitLinearSolvers(LinearFactory);
		for (auto solver : config["LinearProblem"]["solvers"]) {
			AddLinearImplementation(LinearSolvers, LinearFactory, solver);
			settings.solversName.push_back(solver);
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
			settings.caseName = config["LinearProblem"]["case_name"];
		}
		catch (const std::exception& ex) {
			std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
			std::cerr << ex.what() << std::endl;
			throw std::exception("Invalid case name");
		}
				
		A.freadCSR(settings.casePath + "/" + settings.caseName + "/A.txt");
		b.AddData(settings.casePath + "/" + settings.caseName + "/B.vec");

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
			settings.resFileName = "res.csv";
		}
	}


	void ProblemCase::start() {
		double start, stop;
		size_t cur = 0;
		std::ofstream resFile(settings.casePath + "/" + settings.caseName + "/" + settings.resFileName);
		resFile << "SolverName" << ", " << "time";
		if (settings.check_answer) {
			resFile << ", ||Ax-b||L1, ||Ax-b||L2, ||Ax-b||L1, ";
			resFile << "||Ax-b||/||b||L1, ||Ax-b||/||b||L2, ||Ax-b||/||b||L1";
		}
		resFile << std::endl;
		for (auto solver : LinearSolvers) {
			//TODO: what do with empty yelds in json
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
				resFile << solver.first->getName() << ", " << stop - start;
				if (settings.check_answer) {
					double absnorm1, absnorm2, absnorminf;
					double relnorm1, relnorm2, relnorminf;
					CheckMKL(absnorm1, absnorm2, absnorminf, relnorm1, relnorm2, relnorminf);
					resFile << ", " << absnorm1 << ", ";
					resFile << absnorm2 << ", ";
					resFile << absnorminf << ", ";
					resFile << relnorm1 << ", ";
					resFile << relnorm2 << ", ";
					resFile << relnorminf;
				}
				resFile << std::endl;
			}
			cur++;
			x.Clear();
		}
	}
	
	void ProblemCase::Check(double& absnorm1, double& absnorm2, double& absnorminf, double& relnorm1, double& relnorm2, double& relnorminf) {
		// TODO: bug in check
		int n, nnzA;

		int* h_RowsA = nullptr; // GPU <int>    n+1
		int* h_ColsA = nullptr; // GPU <int>    nnzA
		double* h_ValsA = nullptr; // GPU <double> nnzA 
		double* h_x = nullptr; // GPU <double> n
		double* h_b = nullptr; // GPU <double> n
		double* h_r = nullptr; // GPU <double> n


		int* d_RowsA = nullptr; // GPU <int>    n+1
		int* d_ColsA = nullptr; // GPU <int>    nnzA
		double* d_ValsA = nullptr; // GPU <double> nnzA 
		double* d_x = nullptr; // GPU <double> n
		double* d_b = nullptr; // GPU <double> n
		double* d_r = nullptr; // GPU <double> n

		cusparseHandle_t cusparseH = NULL;
		cusparseCreate(&cusparseH);

		A.GetInfo(n, nnzA);
		A.GetDataCSR(&h_ValsA, &h_RowsA, &h_ColsA);
		b.GetData(&h_b);
		x.GetData(&h_x);
		h_r = (double*)malloc(n * sizeof(double));
		gpuErrchk(cudaMalloc((void**)&d_ValsA, nnzA * sizeof(double)));
		gpuErrchk(cudaMalloc((void**)&d_ColsA, nnzA * sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&d_RowsA, (n + 1) * sizeof(int)));

		gpuErrchk(cudaMalloc((void**)&d_b, n * sizeof(double)));
		gpuErrchk(cudaMalloc((void**)&d_x, n * sizeof(double)));
		gpuErrchk(cudaMalloc((void**)&d_r, n * sizeof(double)));

		gpuErrchk(cudaMemcpy(d_ValsA, h_ValsA, nnzA * sizeof(double), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_ColsA, h_ColsA, nnzA * sizeof(int), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_RowsA, h_RowsA, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));

		gpuErrchk(cudaMemcpy(d_b, h_b, n * sizeof(double), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_x, h_x, n * sizeof(double), cudaMemcpyHostToDevice));

		cusparseSpMatDescr_t matA;
		gpuErrchk(cusparseCreateCsr(&matA, n, n, nnzA, d_RowsA, d_ColsA, d_ValsA, CUSPARSE_INDEX_32I,
			CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
		cusparseDnVecDescr_t vecx = NULL;
		gpuErrchk(cusparseCreateDnVec(&vecx, n, d_x, CUDA_R_64F));
		cusparseDnVecDescr_t vecAx = NULL;
		gpuErrchk(cusparseCreateDnVec(&vecAx, n, d_b, CUDA_R_64F));

		size_t bufferSize = 0;
		double minus_one = -1.0;
		double one = 1.0;

		gpuErrchk(cusparseSpMV_bufferSize(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, vecx,
			&one, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
		void* buffer = NULL;
		gpuErrchk(cudaMalloc(&buffer, bufferSize));

		gpuErrchk(cusparseSpMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, vecx,
			&one, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));

		gpuErrchk(cudaMemcpy(h_r, d_b, sizeof(double) * n, cudaMemcpyDeviceToHost));

		cusparseDestroy(cusparseH);
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

		if (h_r) { free(h_r); }

	}

	void ProblemCase::CheckMKL(double& absnorm1, double& absnorm2, double& absnorminf, double& relnorm1, double& relnorm2, double& relnorminf) {
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

	void ProblemCase::print() {

	}


}
