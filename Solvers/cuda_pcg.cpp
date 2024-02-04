#include "cuda_pcg.hpp"
#include "cuda_helper.hpp"

int LinearSolver_cuda_pcg::Solve(const std::vector<double>& vals,
	const std::vector<MKL_INT>& cols,
	const std::vector<MKL_INT>& rows,
	const std::vector<double>& b,
	std::vector<double>& x
) {

	// settings
	double tolerance = 1e-9;
	int max_iter = 10000;
	bool print_verbose = true;
	if (print_verbose)
	{
		std::cout << std::endl;
		std::cout << "Preconditioned Conjugate Gradient (CUDA-powered): start" << std::endl;
		std::cout << "-------------------------" << std::endl;
		std::cout << "|  iter  |   Resudial   |" << std::endl;
		std::cout << "-------------------------" << std::endl;
	}

#ifdef POLYSOLVER_USE_CUDA
	MKL_INT n = rows.size() - 1;
	MKL_INT nnz = rows[n];

	// init x0 - vector of ones
	std::fill(x.begin(), x.end(), 1.0);
	int* d_cols, * d_rows;
	double* d_vals, *d_valsL, * d_b, * d_x;

	gpuErrchk(cudaMalloc((void**)&d_vals, nnz * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&d_valsL, nnz * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&d_cols, nnz * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_rows, (n + 1) * sizeof(int)));

	gpuErrchk(cudaMalloc((void**)&d_b, n * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&d_x, n * sizeof(double)));

	gpuErrchk(cudaMemcpy(d_vals, vals.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_cols, cols.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_rows, rows.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(d_valsL, d_vals, nnz * sizeof(double), cudaMemcpyDeviceToDevice));

	gpuErrchk(cudaMemcpy(d_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_x, x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
	// r,z
	double* d_r, * d_q, * d_p, * d_z, * d_t;
	gpuErrchk(cudaMalloc((void**)&d_r, n * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&d_q, n * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&d_p, n * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&d_z, n * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&d_t, n * sizeof(double)));

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

	
	
	
	cusparseMatDescr_t descr_A = 0;
	cusparseMatDescr_t descr_L = 0;
	bsric02Info_t info_A = 0;
	bsrsv2Info_t  info_L = 0;
	bsrsv2Info_t  info_Lt = 0;
	int pbufferSize_A;
	int pBufferSize_L;
	int pBufferSize_Lt;
	int pBufferSize;
	void* pBuffer = 0;
	int structural_zero;
	int numerical_zero;
	const double alpha = 1.;

	const cusparseSolvePolicy_t policy_A = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
	const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
	const cusparseSolvePolicy_t policy_Lt = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
	const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
	const cusparseOperation_t trans_Lt = CUSPARSE_OPERATION_TRANSPOSE;

	// step 1: create a descriptor which contains
	// - matrix A is base-0
	// - matrix L is base-0
	// - matrix L is lower triangular
	// - matrix L has non-unit diagonal
	gpuErrchk(cusparseCreateMatDescr(&descr_A));
	gpuErrchk(cusparseSetMatIndexBase(descr_A, CUSPARSE_INDEX_BASE_ZERO));
	gpuErrchk(cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_GENERAL));

	gpuErrchk(cusparseCreateMatDescr(&descr_L));
	gpuErrchk(cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO));
	gpuErrchk(cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL));
	gpuErrchk(cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER));
	gpuErrchk(cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT));

	// step 2: create a empty info structure
	// we need one info for bsric02 and two info's for bsrsv2
	gpuErrchk(cusparseCreateBsric02Info(&info_A));
	gpuErrchk(cusparseCreateBsrsv2Info(&info_L));
	gpuErrchk(cusparseCreateBsrsv2Info(&info_Lt));

	// step 3: query how much memory used in bsric02 and bsrsv2, and allocate the buffer
	gpuErrchk(cusparseDbsric02_bufferSize(cusparse_handle, CUSPARSE_DIRECTION_COLUMN, n, nnz, descr_A, d_vals, d_rows, d_cols, 1, info_A, &pbufferSize_A));
	gpuErrchk(cusparseDbsrsv2_bufferSize(cusparse_handle, CUSPARSE_DIRECTION_COLUMN, trans_L, n, nnz, descr_L, d_vals, d_rows, d_cols, 1, info_L, &pBufferSize_L));
	gpuErrchk(cusparseDbsrsv2_bufferSize(cusparse_handle, CUSPARSE_DIRECTION_COLUMN, trans_Lt, n, nnz, descr_L, d_vals, d_rows, d_cols, 1, info_Lt, &pBufferSize_Lt));

	pBufferSize = std::max(pbufferSize_A, std::max(pBufferSize_L, pBufferSize_Lt));

	// pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
	gpuErrchk(cudaMalloc((void**)&pBuffer, pBufferSize));

	// step 4: perform analysis of incomplete Cholesky on M
	//         perform analysis of triangular solve on L
	//         perform analysis of triangular solve on L'
	// The lower triangular part of M has the same sparsity pattern as L, so
	// we can do analysis of bsric02 and bsrsv2 simultaneously.

	gpuErrchk(cusparseDbsric02_analysis(cusparse_handle, CUSPARSE_DIRECTION_COLUMN, n, nnz, descr_A, d_vals, d_rows, d_cols, 1, info_A, policy_A, pBuffer));
	cusparseStatus_t status = cusparseXbsric02_zeroPivot(cusparse_handle, info_A, &structural_zero);
	if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
		printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
	}

	gpuErrchk(cusparseDbsrsv2_analysis(cusparse_handle, CUSPARSE_DIRECTION_COLUMN, trans_L, n, nnz, descr_L, d_vals, d_rows, d_cols, 1, info_L, policy_L, pBuffer));
	gpuErrchk(cusparseDbsrsv2_analysis(cusparse_handle, CUSPARSE_DIRECTION_COLUMN, trans_Lt, n, nnz, descr_L, d_vals, d_rows, d_cols, 1, info_Lt, policy_Lt, pBuffer));

	// step 5: M = L * L'
	gpuErrchk(cusparseDbsric02(cusparse_handle, CUSPARSE_DIRECTION_COLUMN, n, nnz, descr_A, d_valsL, d_rows, d_cols, 1, info_A, policy_A, pBuffer));
	status = cusparseXbsric02_zeroPivot(cusparse_handle, info_A, &numerical_zero);
	if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
		printf("L(%d,%d) is not positive definite\n", numerical_zero, numerical_zero);
	}

	
	// Create sparse matrix
	gpuErrchk(cusparseCreateCsr(&matA, n, n, nnz, d_rows, d_cols, d_vals, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

	// Creata vector
	gpuErrchk(cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_64F));
	gpuErrchk(cusparseCreateDnVec(&vecR, n, d_r, CUDA_R_64F));
	gpuErrchk(cusparseCreateDnVec(&vecQ, n, d_q, CUDA_R_64F));
	gpuErrchk(cusparseCreateDnVec(&vecP, n, d_p, CUDA_R_64F));

	// allocate an external buffer if needed
	gpuErrchk(cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one_d, matA, vecX, &zero_d, vecR, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
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
		// M z = r
		if (1)
		{
			gpuErrchk(cusparseDbsrsv2_solve(cusparse_handle, CUSPARSE_DIRECTION_COLUMN, trans_L, n, nnz, &alpha, descr_L, d_valsL, d_rows, d_cols, 1, info_L, d_r, d_t, policy_L, pBuffer));
			gpuErrchk(cusparseDbsrsv2_solve(cusparse_handle, CUSPARSE_DIRECTION_COLUMN, trans_Lt, n, nnz, &alpha, descr_L, d_valsL, d_rows, d_cols, 1, info_Lt, d_t, d_z, policy_Lt, pBuffer));

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
	
	// step 6: free resources
	cudaFree(pBuffer);
	cusparseDestroyMatDescr(descr_A);
	cusparseDestroyMatDescr(descr_L);
	cusparseDestroyBsric02Info(info_A);
	cusparseDestroyBsrsv2Info(info_L);
	cusparseDestroyBsrsv2Info(info_Lt);
	
	gpuErrchk(cusparseDestroy(cusparse_handle));
	gpuErrchk(cublasDestroy(cublas_handle));

	// output
	if (print_verbose)
	{
		std::cout << "-------------------------" << std::endl;
		std::cout << "Preconditioned Conjugate Gradient (CUDA-powered). Total: " << std::endl;
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
		std::cout << "Preconditioned Conjugate Gradient (CUDA-powered). CUDA was disabled " << std::endl;
		std::cout << std::endl;
	}
#endif // POLYSOLVER_USE_CUDA


	return 0;
}
