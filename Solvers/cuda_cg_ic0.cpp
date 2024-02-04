#include "cuda_cg_ic0.hpp"

int LinearSolver_cuda_cg_ic0::Solve(const std::vector<double>& vals,
    const std::vector<MKL_INT>& cols,
    const std::vector<MKL_INT>& rows,
    const std::vector<double>& b,
    std::vector<double>& x
    ) {
	std::cout << "CUDA: CG + IC0: start" << std::endl;

#ifdef POLYSOLVER_USE_CUDA
	MKL_INT n = rows.size() - 1;
	MKL_INT nnz = rows[n];

	// init x0
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
	double* d_r, * d_q, * d_p, * d_z, * d_t, *d_f;
	gpuErrchk(cudaMalloc((void**)&d_r, n * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&d_q, n * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&d_p, n * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&d_z, n * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&d_t, n * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&d_f, n * sizeof(double)));
	
	std::vector<double> r(n);
	std::fill(r.begin(), r.end(), 0.0);
	gpuErrchk(cudaMemcpy(d_r, r.data(), n * sizeof(double), cudaMemcpyHostToDevice));
	/*std::vector<double> r(n);
	std::vector<double> q(n);
	std::vector<double> p(n);
	std::vector<double> z(n);
	std::vector<double> t(n);
	const char trans = 'n';
	mkl_cspblas_dcsrgemv(&trans, &n, vals.data(), rows.data(), cols.data(), x.data(), r.data());
	*/

	cusparseHandle_t cusparse_handle;
	cublasHandle_t cublas_handle;

	cusparseCreate(&cusparse_handle);
	cublasCreate(&cublas_handle);
	// Matrix description
	cusparseMatDescr_t descr_A = 0;
	cusparseCreateMatDescr(&descr_A);
	cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr_A, CUSPARSE_INDEX_BASE_ZERO);

	double minus_one_d = -1.0;
	double one_d = 1.0;
	double zero_d = 0.0;

	
	cusparseSpMatDescr_t matA;
	cusparseDnVecDescr_t vecX, vecR, vecQ, vecP;
	void* dBuffer = NULL;
	size_t               bufferSize = 0;

		// Create sparse matrix A in CSR format
	gpuErrchk(cusparseCreateCsr(&matA, n, n, nnz, d_rows, d_cols, d_vals, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
		// Create dense vector X
	gpuErrchk(cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_64F));
		// Create dense vector R
	gpuErrchk(cusparseCreateDnVec(&vecR, n, d_r, CUDA_R_64F));
		// Create dense vector Q
	gpuErrchk(cusparseCreateDnVec(&vecQ, n, d_q, CUDA_R_64F));
		// Create dense vector P
	gpuErrchk(cusparseCreateDnVec(&vecP, n, d_p, CUDA_R_64F));

		// allocate an external buffer if needed
	gpuErrchk(cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&one_d, matA, vecX, &zero_d, vecR, CUDA_R_64F,
		CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
	gpuErrchk(cudaMalloc(&dBuffer, bufferSize));

		// execute SpMV
	gpuErrchk(cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&one_d, matA, vecX, &zero_d, vecR, CUDA_R_64F,
		CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

	
	//gpuErrchk(cusparseDbsrmv(cusparse_handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one_d, descr_A, d_vals, d_rows, d_cols, 2, d_x, &zero_d, d_r));

	CUBLAS_CHECK(cublasDscal(cublas_handle, n, &minus_one_d, d_r, 1));
	CUBLAS_CHECK(cublasDaxpy(cublas_handle, n, &one_d, d_b, 1, d_r, 1));
	double b_norm = 0;
	CUBLAS_CHECK(cublasDnrm2(cublas_handle, n, d_r, 1, &b_norm));
	

	// settings
	double tolerance = 1e-9;
	int max_iter = 1000;

	// internal
	bool is_convergenced = false;
	double aplha, beta, rho = 0., rho_prev;
	int iter = 0;
	double cur_res = 0; double cur_rel = 0;
	do {
		// precond
		if (0) {
			const char upper = 'u';
			const char lower = 'l';
			const char trans = 'n';
			const char diag = 'n';
			//mkl_cspblas_dcsrtrsv(&lower, &trans, &diag, &n, vals.data(), rows.data(), cols.data(), r.data(), t.data());
			//mkl_cspblas_dcsrtrsv(&lower, &trans, &diag, &n, vals.data(), rows.data(), cols.data(), t.data(), z.data());
		}
		else {
			CUBLAS_CHECK(cublasDcopy(cublas_handle, n, d_r, 1, d_z, 1));
			//std::copy(r.begin(), r.end(), z.begin());
		}
		rho_prev = rho;

		// \rho = r^T * z
		rho = 0;
		CUBLAS_CHECK(cublasDdot(cublas_handle, n, d_r, 1, d_z, 1, &rho));

		if (iter == 0) {
			// q0 = r0
			CUBLAS_CHECK(cublasDcopy(cublas_handle, n, d_z, 1, d_p, 1));
		}
		else {
			// beta = rho / rho_prev
			beta = rho / rho_prev;
			//9: p = z + \beta p
			CUBLAS_CHECK(cublasDaxpy(cublas_handle, n, &beta, d_p, 1, d_z, 1));
			CUBLAS_CHECK(cublasDcopy(cublas_handle, n, d_z, 1, d_p, 1));
		}

		gpuErrchk(cudaFree(dBuffer));
		// allocate an external buffer if needed
		gpuErrchk(cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			&one_d, matA, vecP, &zero_d, vecQ, CUDA_R_64F,
			CUSPARSE_SPMV_CSR_ALG1, &bufferSize));
		gpuErrchk(cudaMalloc(&dBuffer, bufferSize));

		// execute SpMV
		gpuErrchk(cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			&one_d, matA, vecP, &zero_d, vecQ, CUDA_R_64F,
			CUSPARSE_SPMV_CSR_ALG1, dBuffer));


		// q = A * p
		//gpuErrchk(cusparseDbsrmv(cusparse_handle, CUSPARSE_DIRECTION_COLUMN, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one_d, descr_A, d_vals, d_rows, d_cols, 1, d_p, &zero_d, d_q));
		//mkl_cspblas_dcsrgemv(&trans, &n, vals.data(), rows.data(), cols.data`(), p.data(), q.data());


		//  dot product
		double pq = 0;
		CUBLAS_CHECK(cublasDdot(cublas_handle, n, d_p, 1, d_q, 1, &pq));

		// alpha = rho / (p^T * q)
		aplha = rho / pq;

		//13: x = x + \alpha p
		CUBLAS_CHECK(cublasDaxpy(cublas_handle, n, &aplha, d_p, 1, d_x, 1));
		//14: r = r - \alpha q
		double minus_alpha = -aplha;
		CUBLAS_CHECK(cublasDaxpy(cublas_handle, n, &minus_alpha, d_q, 1, d_r, 1));


		std::vector<double> h_x(n);
		gpuErrchk(cudaMemcpy(h_x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
		//for (auto num : h_x) { std::cout << std::setprecision(20) << num << std::endl; } std::cout << std::endl;


		double cur_res = 0;
		//check for convergence
		CUBLAS_CHECK(cublasDnrm2(cublas_handle, n, d_r, 1, &cur_res));
		cur_rel = cur_res / b_norm;
		std::cout << cur_rel << std::endl;

		is_convergenced = cur_rel < tolerance;
		iter++;
		if (std::isnan(cur_rel)) {
			std::cout << "Nan detected" << std::endl;
			break;
		}
	} while (!is_convergenced && iter < max_iter);
	gpuErrchk(cudaMemcpy(x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(dBuffer));
	cudaFree(d_vals);
	cudaFree(d_rows);
	cudaFree(d_cols);

	cudaFree(d_b);
	cudaFree(d_x);


	
	cudaFree(d_r);
	cudaFree(d_q);
	cudaFree(d_p);
	cudaFree(d_z);
	cudaFree(d_t);
	cudaFree(d_f);
	std::cout << "	iter_number: " << iter << std::endl;
	std::cout << "	resudial:    " << cur_rel << std::endl;
	std::cout << "CUDA: CG + IC0: end" << std::endl;
#else
	std::cout << "CUDA: disabled end" << std::endl;
#endif // POLYSOLVER_USE_CUDA

	
	return 0;
}
