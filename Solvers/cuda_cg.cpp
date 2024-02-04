#include "cuda_cg.hpp"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = false)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

inline void gpuAssert(cusolverStatus_t code, const char* file, int line, bool abort = false)
{
	switch (code)
	{
	case CUSOLVER_STATUS_NOT_INITIALIZED:
		fprintf(stderr, "GPUassert: CUSOLVER NOT INITIALIZED %s %d\n", file, line);
		break;
	case CUSOLVER_STATUS_ALLOC_FAILED:
		fprintf(stderr, "GPUassert: CUSOLVER ALLOC FAILED %s %d\n", file, line);
		break;
	case CUSOLVER_STATUS_INVALID_VALUE:
		fprintf(stderr, "GPUassert: CUSOLVER CUSOLVER INVALID VALUE %s %d\n", file, line);
		break;
	case CUSOLVER_STATUS_ARCH_MISMATCH:
		fprintf(stderr, "GPUassert: CUSOLVER CUSOLVER ARCH MISMATCH %s %d\n", file, line);
		break;
	case CUSOLVER_STATUS_EXECUTION_FAILED:
		fprintf(stderr, "GPUassert: CUSOLVER EXECUTION FAILED %s %d\n", file, line);
		break;
	case CUSOLVER_STATUS_INTERNAL_ERROR:
		fprintf(stderr, "GPUassert: CUSOLVER INTERNAL ERROR %s %d\n", file, line);
		break;
	case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		fprintf(stderr, "GPUassert: CUSOLVER MATRIX TYPE NOT SUPPORTED %s %d\n", file, line);
		break;
	default:
		break;
	}
}



inline void gpuAssert(cublasStatus_t code, const char* file, int line, bool abort = false)
{
	if (code != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cublasGetStatusString(code), file, line); \
			if (abort) exit(code);
	}
}

inline void gpuAssert(cusparseStatus_t code, const char* file, int line, bool abort = false)
{
	switch (code) {
	case CUSPARSE_STATUS_NOT_INITIALIZED:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_NOT_INITIALIZED %s %d\n", file, line);
		break;
	case CUSPARSE_STATUS_ALLOC_FAILED:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_ALLOC_FAILED %s %d\n", file, line);
		break;
	case CUSPARSE_STATUS_INVALID_VALUE:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_INVALID_VALUE %s %d\n", file, line);
		break;
	case CUSPARSE_STATUS_ARCH_MISMATCH:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_ARCH_MISMATCH %s %d\n", file, line);
		break;
	case CUSPARSE_STATUS_MAPPING_ERROR:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_MAPPING_ERROR %s %d\n", file, line);
		break;
	case CUSPARSE_STATUS_EXECUTION_FAILED:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_EXECUTION_FAILED %s %d\n", file, line);
		break;
	case CUSPARSE_STATUS_INTERNAL_ERROR:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_INTERNAL_ERROR %s %d\n", file, line);
		break;
	case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED %s %d\n", file, line);
		break;
	}
}

int LinearSolver_cuda_cg::Solve(const std::vector<double>& vals,
    const std::vector<MKL_INT>& cols,
    const std::vector<MKL_INT>& rows,
    const std::vector<double>& b,
    std::vector<double>& x
    ) {


	// settings
	double tolerance = 1e-9;
	int max_iter = 1000;
	bool print_verbose = true;
	if (print_verbose) 
	{
		std::cout << std::endl;
		std::cout << "Conjugate Gradient (CUDA-powered): start" << std::endl;
		std::cout << "-------------------------" << std::endl;
		std::cout << "|  iter  |   Resudial   |"<< std::endl;
		std::cout << "-------------------------" << std::endl;
	}
#ifdef POLYSOLVER_USE_CUDA
	MKL_INT n = rows.size() - 1;
	MKL_INT nnz = rows[n];

	// init x0 - vector of ones
	std::fill(x.begin(), x.end(), 1.0);
	int* d_cols, * d_rows;
	double* d_vals, * d_b, * d_x;
	
	gpuErrchk(cudaMalloc((void**)&d_vals, nnz * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&d_cols, nnz * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_rows, (n + 1) * sizeof(int)));

	gpuErrchk(cudaMalloc((void**)&d_b, n * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&d_x, n * sizeof(double)));

	gpuErrchk(cudaMemcpy(d_vals, vals.data(), nnz * sizeof(double),  cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_cols, cols.data(), nnz * sizeof(int),     cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_rows, rows.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
		

	gpuErrchk(cudaMemcpy(d_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_x, x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
	// r,z
	double* d_r, * d_q, * d_p, * d_z;
	gpuErrchk(cudaMalloc((void**)&d_r, n * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&d_q, n * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&d_p, n * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&d_z, n * sizeof(double)));
	
	cusparseHandle_t cusparse_handle;
	cublasHandle_t cublas_handle;

	cusparseCreate(&cusparse_handle);
	cublasCreate(&cublas_handle);

	double minus_one_d = -1.0;
	double one_d = 1.0;
	double zero_d = 0.0;

	cusparseSpMatDescr_t matA;
	cusparseDnVecDescr_t vecX, vecR, vecQ, vecP;
	void* dBuffer = NULL;
	size_t bufferSize = 0;

	// Create sparse matrix
	gpuErrchk(cusparseCreateCsr(&matA, n, n, nnz, d_rows, d_cols, d_vals, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
	
	// Creata vector
	gpuErrchk(cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_64F));
	gpuErrchk(cusparseCreateDnVec(&vecR, n, d_r, CUDA_R_64F));		
	gpuErrchk(cusparseCreateDnVec(&vecQ, n, d_q, CUDA_R_64F));		
	gpuErrchk(cusparseCreateDnVec(&vecP, n, d_p, CUDA_R_64F));

	// allocate an external buffer if needed
	gpuErrchk(cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one_d, matA, vecX, &zero_d, vecR, CUDA_R_64F,	CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
	gpuErrchk(cudaMalloc(&dBuffer, bufferSize));

	// r0 = b - A * x
	// execute SpMV
	gpuErrchk(cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one_d, matA, vecX, &zero_d, vecR, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
	gpuErrchk(cublasDscal(cublas_handle, n, &minus_one_d, d_r, 1));
	gpuErrchk(cublasDaxpy(cublas_handle, n, &one_d, d_b, 1, d_r, 1));

	double r0_norm = 0;
	gpuErrchk(cublasDnrm2(cublas_handle, n, d_r, 1, &r0_norm));

	// internal
	int iter = 0;
	bool converged = false;
	double aplha, beta, rho = 0., rho_prev = 0.;
	double absolute_resudial = 0., relative_resudial = 0.;
	do 
	{
		// precond
		if (0) 
		{
			// place to precond
		}
		else 
		{
			gpuErrchk(cublasDcopy(cublas_handle, n, d_r, 1, d_z, 1));
		}

		rho_prev = rho;
		// \rho = r^T * z
		rho = 0;
		gpuErrchk(cublasDdot(cublas_handle, n, d_r, 1, d_z, 1, &rho));

		if (iter == 0) 
		{
			// q0 = r0
			gpuErrchk(cublasDcopy(cublas_handle, n, d_z, 1, d_p, 1));
		}
		else 
		{
			// beta = rho / rho_prev
			beta = rho / rho_prev;
			//9: p = z + \beta p
			gpuErrchk(cublasDaxpy(cublas_handle, n, &beta, d_p, 1, d_z, 1));
			gpuErrchk(cublasDcopy(cublas_handle, n, d_z, 1, d_p, 1));
		}


		// q = A * p
		// free buffer
		gpuErrchk(cudaFree(dBuffer));
		// allocate an external buffer if needed
		gpuErrchk(cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one_d, matA, vecP, &zero_d, vecQ, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
		gpuErrchk(cudaMalloc(&dBuffer, bufferSize));

		// execute SpMV
		gpuErrchk(cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one_d, matA, vecP, &zero_d, vecQ, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

		//  dot product
		double pq = 0;
		gpuErrchk(cublasDdot(cublas_handle, n, d_p, 1, d_q, 1, &pq));

		// alpha = rho / (p^T * q)
		aplha = rho / pq;

		// x = x + \alpha p
		gpuErrchk(cublasDaxpy(cublas_handle, n, &aplha, d_p, 1, d_x, 1));

		// r = r - \alpha q
		double minus_alpha = -aplha;
		gpuErrchk(cublasDaxpy(cublas_handle, n, &minus_alpha, d_q, 1, d_r, 1));


		//check for convergence
		gpuErrchk(cublasDnrm2(cublas_handle, n, d_r, 1, &absolute_resudial));
		relative_resudial = absolute_resudial / r0_norm;

		// output
		if (print_verbose) 
		{
			std::cout << std::format("|{:^8}| {:>12e} |", iter, relative_resudial) << std::endl;
		}
		

		converged = relative_resudial < tolerance;
		iter++;
		if (std::isnan(absolute_resudial) || std::isnan(relative_resudial)) {
			std::cout << "Nan detected" << std::endl;
			break;
		}
	} while (!converged && iter < max_iter);

	gpuErrchk(cudaMemcpy(x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(dBuffer));
	gpuErrchk(cudaFree(d_vals));
	gpuErrchk(cudaFree(d_rows));
	gpuErrchk(cudaFree(d_cols));

	gpuErrchk(cudaFree(d_b));
	gpuErrchk(cudaFree(d_x));

	gpuErrchk(cudaFree(d_r));
	gpuErrchk(cudaFree(d_q));
	gpuErrchk(cudaFree(d_p));
	gpuErrchk(cudaFree(d_z));	
	
	// output
	if (print_verbose)
	{
		std::cout << "-------------------------" << std::endl;
		std::cout << "Conjugate Gradient (CUDA-powered). Total: " << std::endl;
		std::cout << "Iteration number: " << iter << std::endl;
		std::cout << "Absolute resudial:" << absolute_resudial << std::endl;
		std::cout << "Relative resudial:" << relative_resudial << std::endl;
		std::cout << std::endl;
	}
	
#else
	// output
	if (print_verbose)
	{
		std::cout << "-------------------------" << std::endl;
		std::cout << "Conjugate Gradient (CUDA-powered). CUDA was disabled " << std::endl;
		std::cout << std::endl;
	}
#endif // POLYSOLVER_USE_CUDA

	
	return 0;
}
