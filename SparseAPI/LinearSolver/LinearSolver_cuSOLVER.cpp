#include "LinearSolver_cuSOLVER.h"
namespace SPARSE {
	int LinearSolvercuSOLVERSP::SolveRightSide(SparseMatrix &A,
		SparseVector &b,
		SparseVector &x){

		A.GetInfo(n, nnz);
		A.GetDataCSR(&h_Vals, &h_Rows, &h_Cols);
		b.GetData(&h_b);
		int nb = 0, nrhs = 0;
		b.GetInfo(nb, nrhs);
		if (nb != n) {
			return -1;
		}

		gpuErrchk(cudaMalloc((void**)&d_Vals, nnz * sizeof(double)));
		gpuErrchk(cudaMalloc((void**)&d_Cols, nnz * sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&d_Rows, (n + 1) * sizeof(int)));

		gpuErrchk(cudaMalloc((void**)&d_b, n * sizeof(double)));
		gpuErrchk(cudaMalloc((void**)&d_x, n * sizeof(double)));

		gpuErrchk(cudaMemcpy(d_Vals, h_Vals, nnz * sizeof(double),  cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_Cols, h_Cols, nnz * sizeof(int),     cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_Rows, h_Rows, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));

		gpuErrchk(cudaMemcpy(d_b, h_b, n * sizeof(double), cudaMemcpyHostToDevice));
		
		cusolverSpHandle_t handle = NULL;
		cusolverSpCreate(&handle);
		cusparseMatDescr_t descr_A = 0;
		cusparseCreateMatDescr(&descr_A);

		cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descr_A, CUSPARSE_INDEX_BASE_ZERO);

		h_x = (double*)malloc(n * sizeof(double));

		int singularity = 0;	
		for (int i = 0; i < nrhs; i++) {

			gpuErrchk(cudaMemcpy(d_b, h_b + i * n, n * sizeof(double), cudaMemcpyHostToDevice));

			gpuErrchk(cusolverSpDcsrlsvqr(handle, n, nnz, descr_A, d_Vals, d_Rows, d_Cols, d_b, 1e-12, 0, d_x, &singularity));

			gpuErrchk(cudaMemcpy(h_x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
			x.AddData(h_x, n);
		}

		gpuErrchk(cudaFree(d_Vals)); d_Vals = nullptr;
		gpuErrchk(cudaFree(d_Cols)); d_Cols = nullptr;
		gpuErrchk(cudaFree(d_Rows)); d_Rows = nullptr;
		gpuErrchk(cudaFree(d_x)); d_x = nullptr;
		gpuErrchk(cudaFree(d_b)); d_b = nullptr;

		

		//free(h_x); h_x = nullptr;
		return 0;
	}

	void LinearSolvercuSOLVERRF::freeGPUMemory() 
	{
		if (d_RowsA) { gpuErrchk(cudaFree(d_RowsA)); d_RowsA = nullptr; }
		if (d_ColsA) { gpuErrchk(cudaFree(d_ColsA)); d_ColsA = nullptr; }
		if (d_ValsA) { gpuErrchk(cudaFree(d_ValsA)); d_ValsA = nullptr; }

		if (d_RowsL) { gpuErrchk(cudaFree(d_RowsL)); d_RowsL = nullptr; }
		if (d_ColsL) { gpuErrchk(cudaFree(d_ColsL)); d_ColsL = nullptr; }
		if (d_ValsL) { gpuErrchk(cudaFree(d_ValsL)); d_ValsL = nullptr; }

		if (d_RowsU) { gpuErrchk(cudaFree(d_RowsU)); d_RowsU = nullptr; }
		if (d_ColsU) { gpuErrchk(cudaFree(d_ColsU)); d_ColsU = nullptr; }
		if (d_ValsU) { gpuErrchk(cudaFree(d_ValsU)); d_ValsU = nullptr; }

		if (d_x) { gpuErrchk(cudaFree(d_x)); d_x = nullptr; }
		if (d_y) { gpuErrchk(cudaFree(d_x)); d_x = nullptr; }
		if (d_b) { gpuErrchk(cudaFree(d_b)); d_b = nullptr; }


	}
	 int LinearSolvercuSOLVERRF::SolveRightSide(SparseMatrix& A, SparseVector& b, SparseVector& x) 
	 {
		A.GetInfo(n, nnzA);
		A.GetDataCSR(&h_ValsA, &h_RowsA, &h_ColsA);
		b.GetData(&h_b);
		int nb = 0, nrhs = 0;
		b.GetInfo(nb, nrhs);
		if (nb != n  || nrhs <=0) 
		{
			return -1;
		}
		// cuda data
		size_t freeMem  = 0;
		size_t needMem  = 0;
		size_t totalMem = 0;
		
		// cuSolverSP handle - always need
		cusolverSpHandle_t cusolverSpH = NULL;
		cusolverSpCreate(&cusolverSpH);

		// Matrix description
		cusparseMatDescr_t descr_A = 0;
		cusparseCreateMatDescr(&descr_A);

		cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descr_A, CUSPARSE_INDEX_BASE_ZERO);




		double tol = 1e-12;
		int singularity = 0; // if < 0 => matrix is singular 
		
		if (nrhs == 1) 
		{
			gpuErrchk(cudaMalloc((void**)&d_ValsA, nnzA    * sizeof(double)));
			gpuErrchk(cudaMalloc((void**)&d_ColsA, nnzA    * sizeof(int)   ));
			gpuErrchk(cudaMalloc((void**)&d_RowsA, (n + 1) * sizeof(int)   ));

			gpuErrchk(cudaMalloc((void**)&d_b, n    * sizeof(double)));
			gpuErrchk(cudaMalloc((void**)&d_x, nnzA * sizeof(int)   ));

			gpuErrchk(cudaMemcpy(d_ValsA, h_ValsA, nnzA    * sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(d_ColsA, h_ColsA, nnzA    * sizeof(int)   , cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(d_RowsA, h_RowsA, (n + 1) * sizeof(int)   , cudaMemcpyHostToDevice));

			gpuErrchk(cudaMemcpy(d_b, h_b, n * sizeof(double), cudaMemcpyHostToDevice));
			
			h_x = (double*)malloc(n * sizeof(double));
			
			gpuErrchk(cudaMemcpy(d_b, h_b, n * sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cusolverSpDcsrlsvqr(cusolverSpH, n, nnzA, descr_A, d_ValsA, d_RowsA, d_ColsA, d_b, tol, 0, d_x, &singularity));
			gpuErrchk(cudaMemcpy(h_x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

			x.AddData(h_x, n);
		}
		else
		{
			// cuSparse handle 
			// need only rhs > 1
			cusparseHandle_t cusparseH = NULL;
			cusparseCreate(&cusparseH);

			// cuSolverSp data for input LU decomposition to device solve
			csrluInfoHost_t info = NULL;
			void* buffer_cpu = NULL;
			size_t size_internal = 0;
			size_t size_lu = 0;
			const double pivot_threshold = 0.0;

			// cuSparse data to device solve triangular matrix
			cusparseMatDescr_t descr_M = 0, descr_L = 0, descr_U = 0;
			int pBufferSize_L, pBufferSize_U, pBufferSize;
			void* pBuffer = 0, * pBuffer_L = 0, * pBuffer_U = 0;
			double alpha = 1.0;
			bsrsv2Info_t info_L = 0, info_U = 0;

			const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
			const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
			const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
			const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
			const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;


			// first rhs - only CPU
			gpuErrchk(cusolverSpCreateCsrluInfoHost(&info));
			gpuErrchk(cusolverSpXcsrluAnalysisHost(cusolverSpH, n, nnzA, descr_A, h_RowsA, h_ColsA, info));
			gpuErrchk(cusolverSpDcsrluBufferInfoHost(cusolverSpH, n, nnzA, descr_A, h_ValsA, h_RowsA, h_ColsA, info, &size_internal, &size_lu));
			buffer_cpu = (void*)malloc(sizeof(char) * size_lu);
			gpuErrchk(cusolverSpDcsrluFactorHost(cusolverSpH, n, nnzA, descr_A, h_ValsA, h_RowsA, h_ColsA, info, pivot_threshold, buffer_cpu));
			gpuErrchk(cusolverSpDcsrluZeroPivotHost(cusolverSpH, info, tol, &singularity));

			h_x = (double*)malloc(n * sizeof(double));
			gpuErrchk(cusolverSpDcsrluSolveHost(cusolverSpH, n, h_b, h_x, info, buffer_cpu));
			x.AddData(h_x, n);


			// assume P*A*Q^T = L*U
			// load info about P,Q,L,U

			gpuErrchk(cusolverSpXcsrluNnzHost(cusolverSpH, &nnzL, &nnzU, info));

			gpuErrchk(cudaMemGetInfo(&freeMem, &totalMem));

			needMem += (n + 1) * sizeof(int) + nnzL * sizeof(int) + nnzL * sizeof(double);
			needMem += (n + 1) * sizeof(int) + nnzU * sizeof(int) + nnzU * sizeof(double);
			needMem += 3 * n * sizeof(double);

			if (needMem > freeMem)
			{
				free(buffer_cpu); buffer_cpu = nullptr;
				free(h_x); h_x = nullptr;
				fprintf(stderr, "Not enough memory : need %d free %d", needMem, freeMem);
				return -1;
			}

			h_P     =    (int*)malloc(sizeof(int)    * n);
			h_Q     =    (int*)malloc(sizeof(int)    * n);

			h_RowsL =    (int*)malloc(sizeof(int)    * (n + 1));
			h_ColsL =    (int*)malloc(sizeof(int)    * nnzL   );
			h_ValsL = (double*)malloc(sizeof(double) * nnzL   );

			h_RowsU =    (int*)malloc(sizeof(int)    * (n + 1));
			h_ColsU =    (int*)malloc(sizeof(int)    * nnzU   );
			h_ValsU = (double*)malloc(sizeof(double) * nnzU   );

			

			gpuErrchk(cusolverSpDcsrluExtractHost(cusolverSpH, h_P, h_Q, descr_A, h_ValsL, h_RowsL, h_ColsL, descr_A, h_ValsU, h_RowsU, h_ColsU, info, buffer_cpu));
			
			SparseVectorInt P;
			P.AddData(h_P, n);
			P.fprint(n,"C:/MatlabRepository/GPU/P.txt");
			SparseVectorInt Q;
			Q.AddData(h_Q, n);
			Q.fprint(n, "C:/MatlabRepository/GPU/Q.txt");
			free(buffer_cpu); buffer_cpu = nullptr;



			gpuErrchk(cudaMalloc((void**)&d_RowsL, (n + 1) * sizeof(int))   );
			gpuErrchk(cudaMalloc((void**)&d_ColsL, nnzL    * sizeof(int))   );
			gpuErrchk(cudaMalloc((void**)&d_ValsL, nnzL    * sizeof(double)));

			gpuErrchk(cudaMalloc((void**)&d_RowsU, (n + 1) * sizeof(int))   );
			gpuErrchk(cudaMalloc((void**)&d_ColsU, nnzU    * sizeof(int))   );
			gpuErrchk(cudaMalloc((void**)&d_ValsU, nnzU    * sizeof(double)));

			gpuErrchk(cudaMalloc((void**)&d_b    , n       * sizeof(double)));
			gpuErrchk(cudaMalloc((void**)&d_y    , n       * sizeof(double)));
			gpuErrchk(cudaMalloc((void**)&d_x    , n       * sizeof(double)));

			gpuErrchk(cudaMemcpy(d_RowsL, h_RowsL, (n + 1) * sizeof(int)   , cudaMemcpyHostToDevice)); free(h_RowsL); h_RowsL = nullptr;
			gpuErrchk(cudaMemcpy(d_ColsL, h_ColsL, nnzL    * sizeof(int)   , cudaMemcpyHostToDevice)); free(h_ColsL); h_ColsL = nullptr;
			gpuErrchk(cudaMemcpy(d_ValsL, h_ValsL, nnzL    * sizeof(double), cudaMemcpyHostToDevice)); free(h_ValsL); h_ValsL = nullptr;

			gpuErrchk(cudaMemcpy(d_RowsU, h_RowsU, (n + 1) * sizeof(int)   , cudaMemcpyHostToDevice)); free(h_RowsU); h_RowsU = nullptr;
			gpuErrchk(cudaMemcpy(d_ColsU, h_ColsU, nnzU    * sizeof(int)   , cudaMemcpyHostToDevice)); free(h_ColsU); h_ColsU = nullptr;
			gpuErrchk(cudaMemcpy(d_ValsU, h_ValsU, nnzU    * sizeof(double), cudaMemcpyHostToDevice)); free(h_ValsU); h_ValsU = nullptr;

			


			// start triangular solve
			gpuErrchk(cusparseCreateBsrsv2Info(&info_L));
			gpuErrchk(cusparseCreateBsrsv2Info(&info_U));

			gpuErrchk(cusparseCreateMatDescr (&descr_L));
			gpuErrchk(cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO));
			gpuErrchk(cusparseSetMatType     (descr_L, CUSPARSE_MATRIX_TYPE_GENERAL));
			gpuErrchk(cusparseSetMatFillMode (descr_L, CUSPARSE_FILL_MODE_LOWER));
			gpuErrchk(cusparseSetMatDiagType (descr_L, CUSPARSE_DIAG_TYPE_UNIT));

			gpuErrchk(cusparseCreateMatDescr (&descr_U));
			gpuErrchk(cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO));
			gpuErrchk(cusparseSetMatType     (descr_L, CUSPARSE_MATRIX_TYPE_GENERAL));
			gpuErrchk(cusparseSetMatFillMode (descr_U, CUSPARSE_FILL_MODE_UPPER));
			gpuErrchk(cusparseSetMatDiagType (descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT));



			gpuErrchk(cusparseDbsrsv2_bufferSize(cusparseH, CUSPARSE_DIRECTION_ROW, trans_L, n, nnzL, descr_L, d_ValsL, d_RowsL, d_ColsL, 1, info_L, &pBufferSize_L));
			gpuErrchk(cusparseDbsrsv2_bufferSize(cusparseH, CUSPARSE_DIRECTION_ROW, trans_U, n, nnzU, descr_U, d_ValsU, d_RowsU, d_ColsU, 1, info_U, &pBufferSize_U));

			pBufferSize = std::max(pBufferSize_L, pBufferSize_U);

			gpuErrchk(cudaMalloc((void**)&pBuffer_L, pBufferSize_L));
			gpuErrchk(cudaMalloc((void**)&pBuffer_U, pBufferSize_U));

			gpuErrchk(cusparseDbsrsv2_analysis(cusparseH, CUSPARSE_DIRECTION_ROW, trans_L, n, nnzL, descr_L, d_ValsL, d_RowsL, d_ColsL, 1, info_L, policy_L, pBuffer_L));
			gpuErrchk(cusparseDbsrsv2_analysis(cusparseH, CUSPARSE_DIRECTION_ROW, trans_U, n, nnzU, descr_U, d_ValsU, d_RowsU, d_ColsU, 1, info_U, policy_U, pBuffer_U));

			for (int i = 1; i < nrhs; i++) 
			{
				gpuErrchk(cudaMemcpy(d_b, h_b + i * n, n * sizeof(double), cudaMemcpyHostToDevice));
				gpuErrchk(cusparseDbsrsv2_solve(cusparseH, CUSPARSE_DIRECTION_ROW, trans_L, n, nnzL, &alpha, descr_L, d_ValsL, d_RowsL, d_ColsL, 1, info_L, d_b, d_y, policy_L, pBuffer_L));
				gpuErrchk(cusparseDbsrsv2_solve(cusparseH, CUSPARSE_DIRECTION_ROW, trans_U, n, nnzU, &alpha, descr_U, d_ValsU, d_RowsU, d_ColsU, 1, info_U, d_y, d_x, policy_U, pBuffer_U));
				gpuErrchk(cudaMemcpy(h_x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
				x.AddData(h_x, n);
			}
			gpuErrchk(cudaFree(pBuffer_L)); pBuffer_L = nullptr;
			gpuErrchk(cudaFree(pBuffer_U)); pBuffer_U = nullptr;
		}

		free(h_x); h_x = nullptr;
		freeGPUMemory();
		return 0;
	 }


	 void LinearSolvercuSOLVERRF0::freeGPUMemory()
	 {
		 if (d_RowsA) { gpuErrchk(cudaFree(d_RowsA)); d_RowsA = nullptr; }
		 if (d_ColsA) { gpuErrchk(cudaFree(d_ColsA)); d_ColsA = nullptr; }
		 if (d_ValsA) { gpuErrchk(cudaFree(d_ValsA)); d_ValsA = nullptr; }

		 if (d_RowsL) { gpuErrchk(cudaFree(d_RowsL)); d_RowsL = nullptr; }
		 if (d_ColsL) { gpuErrchk(cudaFree(d_ColsL)); d_ColsL = nullptr; }
		 if (d_ValsL) { gpuErrchk(cudaFree(d_ValsL)); d_ValsL = nullptr; }

		 if (d_RowsU) { gpuErrchk(cudaFree(d_RowsU)); d_RowsU = nullptr; }
		 if (d_ColsU) { gpuErrchk(cudaFree(d_ColsU)); d_ColsU = nullptr; }
		 if (d_ValsU) { gpuErrchk(cudaFree(d_ValsU)); d_ValsU = nullptr; }

		 if (d_x) { gpuErrchk(cudaFree(d_x)); d_x = nullptr; }
		 if (d_y) { gpuErrchk(cudaFree(d_x)); d_x = nullptr; }
		 if (d_b) { gpuErrchk(cudaFree(d_b)); d_b = nullptr; }


	 }
	 int LinearSolvercuSOLVERRF0::SolveRightSide(SparseMatrix& A, SparseVector& b, SparseVector& x)
	 {
		 A.GetInfo(n, nnzA);
		 A.GetDataCSR(&h_ValsA, &h_RowsA, &h_ColsA);
		 b.GetData(&h_b);
		 int nb = 0, nrhs = 0;
		 b.GetInfo(nb, nrhs);
		 if (nb != n || nrhs <= 0)
		 {
			 return -1;
		 }
		 // cuda data
		 size_t freeMem = 0;
		 size_t needMem = 0;
		 size_t totalMem = 0;

		 // cuSolverSP handle - always need
		 cusolverSpHandle_t cusolverSpH = NULL;
		 cusolverSpCreate(&cusolverSpH);

		 // Matrix description
		 cusparseMatDescr_t descr_A = 0;
		 cusparseCreateMatDescr(&descr_A);

		 cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_GENERAL);
		 cusparseSetMatIndexBase(descr_A, CUSPARSE_INDEX_BASE_ZERO);




		 double tol = 1e-12;
		 int singularity = 0; // if < 0 => matrix is singular 

		 if (nrhs == 1)
		 {
			 gpuErrchk(cudaMalloc((void**)&d_ValsA, nnzA * sizeof(double)));
			 gpuErrchk(cudaMalloc((void**)&d_ColsA, nnzA * sizeof(int)));
			 gpuErrchk(cudaMalloc((void**)&d_RowsA, (n + 1) * sizeof(int)));

			 gpuErrchk(cudaMalloc((void**)&d_b, n * sizeof(double)));
			 gpuErrchk(cudaMalloc((void**)&d_x, nnzA * sizeof(int)));

			 gpuErrchk(cudaMemcpy(d_ValsA, h_ValsA, nnzA * sizeof(double), cudaMemcpyHostToDevice));
			 gpuErrchk(cudaMemcpy(d_ColsA, h_ColsA, nnzA * sizeof(int), cudaMemcpyHostToDevice));
			 gpuErrchk(cudaMemcpy(d_RowsA, h_RowsA, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));

			 gpuErrchk(cudaMemcpy(d_b, h_b, n * sizeof(double), cudaMemcpyHostToDevice));

			 h_x = (double*)malloc(n * sizeof(double));

			 gpuErrchk(cudaMemcpy(d_b, h_b, n * sizeof(double), cudaMemcpyHostToDevice));
			 gpuErrchk(cusolverSpDcsrlsvqr(cusolverSpH, n, nnzA, descr_A, d_ValsA, d_RowsA, d_ColsA, d_b, tol, 0, d_x, &singularity));
			 gpuErrchk(cudaMemcpy(h_x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

			 x.AddData(h_x, n);
		 }
		 else
		 {
			 // cuSparse handle 
			 // need only rhs > 1
			 cusparseHandle_t cusparseH = NULL;
			 cusparseCreate(&cusparseH);

			 // cuSolverSp data for input LU decomposition to device solve
			 csrluInfoHost_t info = NULL;
			 void* buffer_cpu = NULL;
			 size_t size_internal = 0;
			 size_t size_lu = 0;
			 const double pivot_threshold = 0.0;

			 // cuSparse data to device solve triangular matrix
			 cusparseMatDescr_t descr_M = 0, descr_L = 0, descr_U = 0;
			 int pBufferSize_L, pBufferSize_U, pBufferSize;
			 void* pBuffer = 0, * pBuffer_L = 0, * pBuffer_U = 0;
			 double alpha = 1.0;
			 bsrsv2Info_t info_L = 0, info_U = 0;

			 const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
			 const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
			 const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
			 const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
			 const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;


			 // first rhs - only CPU
			 gpuErrchk(cusolverSpCreateCsrluInfoHost(&info));
			 gpuErrchk(cusolverSpXcsrluAnalysisHost(cusolverSpH, n, nnzA, descr_A, h_RowsA, h_ColsA, info));
			 gpuErrchk(cusolverSpDcsrluBufferInfoHost(cusolverSpH, n, nnzA, descr_A, h_ValsA, h_RowsA, h_ColsA, info, &size_internal, &size_lu));
			 buffer_cpu = (void*)malloc(sizeof(char) * size_lu);
			 gpuErrchk(cusolverSpDcsrluFactorHost(cusolverSpH, n, nnzA, descr_A, h_ValsA, h_RowsA, h_ColsA, info, pivot_threshold, buffer_cpu));		

			 // assume P*A*Q^T = L*U
			 // load info about P,Q,L,U

			 gpuErrchk(cusolverSpXcsrluNnzHost(cusolverSpH, &nnzL, &nnzU, info));

			 gpuErrchk(cudaMemGetInfo(&freeMem, &totalMem));

			 needMem += (n + 1) * sizeof(int) + nnzL * sizeof(int) + nnzL * sizeof(double);
			 needMem += (n + 1) * sizeof(int) + nnzU * sizeof(int) + nnzU * sizeof(double);
			 needMem += 3 * n * sizeof(double);

			 if (needMem > freeMem)
			 {
				 free(buffer_cpu); buffer_cpu = nullptr;
				 free(h_x); h_x = nullptr;
				 fprintf(stderr, "Not enough memory : need %d free %d", needMem, freeMem);
				 return -1;
			 }

			 h_P = (int*)malloc(sizeof(int) * n);
			 h_Q = (int*)malloc(sizeof(int) * n);

			 h_RowsL = (int*)malloc(sizeof(int) * (n + 1));
			 h_ColsL = (int*)malloc(sizeof(int) * nnzL);
			 h_ValsL = (double*)malloc(sizeof(double) * nnzL);

			 h_RowsU = (int*)malloc(sizeof(int) * (n + 1));
			 h_ColsU = (int*)malloc(sizeof(int) * nnzU);
			 h_ValsU = (double*)malloc(sizeof(double) * nnzU);



			 gpuErrchk(cusolverSpDcsrluExtractHost(cusolverSpH, h_P, h_Q, descr_A, h_ValsL, h_RowsL, h_ColsL, descr_A, h_ValsU, h_RowsU, h_ColsU, info, buffer_cpu));

			 SparseVectorInt P;
			 P.AddData(h_P, n);
			 P.fprint(n, "C:/MatlabRepository/GPU/P.txt");
			 SparseVectorInt Q;
			 Q.AddData(h_Q, n);
			 Q.fprint(n, "C:/MatlabRepository/GPU/Q.txt");
			 free(buffer_cpu); buffer_cpu = nullptr;



			 gpuErrchk(cudaMalloc((void**)&d_RowsL, (n + 1) * sizeof(int)));
			 gpuErrchk(cudaMalloc((void**)&d_ColsL, nnzL * sizeof(int)));
			 gpuErrchk(cudaMalloc((void**)&d_ValsL, nnzL * sizeof(double)));

			 gpuErrchk(cudaMalloc((void**)&d_RowsU, (n + 1) * sizeof(int)));
			 gpuErrchk(cudaMalloc((void**)&d_ColsU, nnzU * sizeof(int)));
			 gpuErrchk(cudaMalloc((void**)&d_ValsU, nnzU * sizeof(double)));

			 gpuErrchk(cudaMalloc((void**)&d_b, n * sizeof(double)));
			 gpuErrchk(cudaMalloc((void**)&d_y, n * sizeof(double)));
			 gpuErrchk(cudaMalloc((void**)&d_x, n * sizeof(double)));

			 gpuErrchk(cudaMemcpy(d_RowsL, h_RowsL, (n + 1) * sizeof(int), cudaMemcpyHostToDevice)); free(h_RowsL); h_RowsL = nullptr;
			 gpuErrchk(cudaMemcpy(d_ColsL, h_ColsL, nnzL * sizeof(int), cudaMemcpyHostToDevice)); free(h_ColsL); h_ColsL = nullptr;
			 gpuErrchk(cudaMemcpy(d_ValsL, h_ValsL, nnzL * sizeof(double), cudaMemcpyHostToDevice)); free(h_ValsL); h_ValsL = nullptr;

			 gpuErrchk(cudaMemcpy(d_RowsU, h_RowsU, (n + 1) * sizeof(int), cudaMemcpyHostToDevice)); free(h_RowsU); h_RowsU = nullptr;
			 gpuErrchk(cudaMemcpy(d_ColsU, h_ColsU, nnzU * sizeof(int), cudaMemcpyHostToDevice)); free(h_ColsU); h_ColsU = nullptr;
			 gpuErrchk(cudaMemcpy(d_ValsU, h_ValsU, nnzU * sizeof(double), cudaMemcpyHostToDevice)); free(h_ValsU); h_ValsU = nullptr;




			 // start triangular solve
			 gpuErrchk(cusparseCreateBsrsv2Info(&info_L));
			 gpuErrchk(cusparseCreateBsrsv2Info(&info_U));

			 gpuErrchk(cusparseCreateMatDescr(&descr_L));
			 gpuErrchk(cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO));
			 gpuErrchk(cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL));
			 gpuErrchk(cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER));
			 gpuErrchk(cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT));

			 gpuErrchk(cusparseCreateMatDescr(&descr_U));
			 gpuErrchk(cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO));
			 gpuErrchk(cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL));
			 gpuErrchk(cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER));
			 gpuErrchk(cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT));



			 gpuErrchk(cusparseDbsrsv2_bufferSize(cusparseH, CUSPARSE_DIRECTION_ROW, trans_L, n, nnzL, descr_L, d_ValsL, d_RowsL, d_ColsL, 1, info_L, &pBufferSize_L));
			 gpuErrchk(cusparseDbsrsv2_bufferSize(cusparseH, CUSPARSE_DIRECTION_ROW, trans_U, n, nnzU, descr_U, d_ValsU, d_RowsU, d_ColsU, 1, info_U, &pBufferSize_U));

			 pBufferSize = std::max(pBufferSize_L, pBufferSize_U);

			 gpuErrchk(cudaMalloc((void**)&pBuffer_L, pBufferSize_L));
			 gpuErrchk(cudaMalloc((void**)&pBuffer_U, pBufferSize_U));

			 gpuErrchk(cusparseDbsrsv2_analysis(cusparseH, CUSPARSE_DIRECTION_ROW, trans_L, n, nnzL, descr_L, d_ValsL, d_RowsL, d_ColsL, 1, info_L, policy_L, pBuffer_L));
			 gpuErrchk(cusparseDbsrsv2_analysis(cusparseH, CUSPARSE_DIRECTION_ROW, trans_U, n, nnzU, descr_U, d_ValsU, d_RowsU, d_ColsU, 1, info_U, policy_U, pBuffer_U));

			 h_x = (double*)malloc(n * sizeof(double));
			 for (int i = 0; i < nrhs; i++)
			 {
				 gpuErrchk(cudaMemcpy(d_b, h_b + i * n, n * sizeof(double), cudaMemcpyHostToDevice));
				 gpuErrchk(cusparseDbsrsv2_solve(cusparseH, CUSPARSE_DIRECTION_ROW, trans_L, n, nnzL, &alpha, descr_L, d_ValsL, d_RowsL, d_ColsL, 1, info_L, d_b, d_y, policy_L, pBuffer_L));
				 gpuErrchk(cusparseDbsrsv2_solve(cusparseH, CUSPARSE_DIRECTION_ROW, trans_U, n, nnzU, &alpha, descr_U, d_ValsU, d_RowsU, d_ColsU, 1, info_U, d_y, d_x, policy_U, pBuffer_U));
				 gpuErrchk(cudaMemcpy(h_x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
				 x.AddData(h_x, n);
			 }
			 gpuErrchk(cudaFree(pBuffer_L)); pBuffer_L = nullptr;
			 gpuErrchk(cudaFree(pBuffer_U)); pBuffer_U = nullptr;
		 }

		 free(h_x); h_x = nullptr;
		 freeGPUMemory();
		 return 0;
	 }


		void LinearSolvercuSOLVERRF_ALLGPU::freeGPUMemory()
		{
			if (d_RowsA) { gpuErrchk(cudaFree(d_RowsA)); d_RowsA = nullptr; }	
			if (d_ColsA) { gpuErrchk(cudaFree(d_ColsA)); d_ColsA = nullptr; }	
			if (d_ValsA) { gpuErrchk(cudaFree(d_ValsA)); d_ValsA = nullptr; }	
	
			if (d_x) { gpuErrchk(cudaFree(d_x)); d_x = nullptr; }	
			if (d_y) { gpuErrchk(cudaFree(d_x)); d_x = nullptr; }	
			if (d_b) { gpuErrchk(cudaFree(d_b)); d_b = nullptr; }	
		}
		int LinearSolvercuSOLVERRF_ALLGPU::SolveRightSide(SparseMatrix& A, SparseVector& b, SparseVector& x)
		{
			A.GetInfo(n, nnzA);
			A.GetDataCSR(&h_ValsA, &h_RowsA, &h_ColsA);
			b.GetData(&h_b);
			int nb = 0, nrhs = 0;
			b.GetInfo(nb, nrhs);
			if (nb != n || nrhs <= 0)
			{
				return -1;
			}
		 
		 
			gpuErrchk(cudaMalloc((void**)&d_ValsA, nnzA * sizeof(double)));
			gpuErrchk(cudaMalloc((void**)&d_ColsA, nnzA * sizeof(int)));
			gpuErrchk(cudaMalloc((void**)&d_RowsA, (n + 1) * sizeof(int)));

			gpuErrchk(cudaMalloc((void**)&d_b, n * sizeof(double)));
			gpuErrchk(cudaMalloc((void**)&d_x, nnzA * sizeof(int)));

			gpuErrchk(cudaMemcpy(d_ValsA, h_ValsA, nnzA * sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(d_ColsA, h_ColsA, nnzA * sizeof(int), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(d_RowsA, h_RowsA, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));

			gpuErrchk(cudaMemcpy(d_b, h_b, n * sizeof(double), cudaMemcpyHostToDevice));
		 
		 
		 
		 
			// cuda data
			size_t freeMem = 0;
			size_t needMem = 0;
			size_t totalMem = 0;

		 
		 
		 
			// cuSolverSP handle - always need
			cusolverSpHandle_t cusolverSpH = NULL;
			cusolverSpCreate(&cusolverSpH);

			// Matrix description
			cusparseMatDescr_t descr_A = 0;
			cusparseCreateMatDescr(&descr_A);

			cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_GENERAL);
			cusparseSetMatIndexBase(descr_A, CUSPARSE_INDEX_BASE_ZERO);






			double tol = 1e-12;
			int singularity = 0; // if < 0 => matrix is singular 


		// cuSparse handle 
		// need only rhs > 1
		cusparseHandle_t cusparseH = NULL;
		cusparseCreate(&cusparseH);

		// cuSolverSp data for input LU decomposition to device solve
		csrcholInfo_t info = NULL;
		void* buffer_gpu = NULL;
		void* buffer_cpu = NULL;

		size_t size_perm = 0;
		size_t size_internal = 0;
		size_t size_chol = 0;
		const double pivot_threshold = 0.0;

		int* P = (int*)malloc(sizeof(int) * n);
		int* I = (int*)malloc(sizeof(int) * nnzA);
		int* h_RowsB = (int*)malloc(sizeof(int) * (n+1));
		int* h_ColsB = (int*)malloc(sizeof(int) * nnzA);
		double* h_ValsB = (double*)malloc(sizeof(double) * nnzA);

		memcpy(h_RowsB, h_RowsA, sizeof(int) * (n + 1));
		memcpy(h_ColsB, h_ColsA, sizeof(int) * nnzA);

#pragma omp parallel for 
		for (int j = 0; j < nnzA; j++) {
			I[j] = j;
		}

		gpuErrchk(cusolverSpXcsrsymamdHost(cusolverSpH, n, nnzA, descr_A, h_RowsA, h_ColsA, P));
		gpuErrchk(cusolverSpXcsrperm_bufferSizeHost(cusolverSpH, n, n, nnzA, descr_A, h_RowsB, h_ColsB, P, P, &size_perm));
		buffer_cpu = (void*)malloc(sizeof(char) * size_perm);

		gpuErrchk(cusolverSpXcsrpermHost(cusolverSpH, n, n, nnzA, descr_A, h_ColsA, h_RowsA, P, P, I, buffer_cpu));
		for (int j = 0; j < nnzA; j++) {
			h_ValsB[j] = h_ValsA[I[j]];
		}

		// factor
		gpuErrchk(cusolverSpCreateCsrcholInfo(&info));
		gpuErrchk(cusolverSpXcsrcholAnalysis(cusolverSpH, n, nnzA, descr_A, d_RowsA, d_ColsA, info));
		gpuErrchk(cusolverSpDcsrcholBufferInfo(cusolverSpH, n, nnzA, descr_A, d_ValsA, d_RowsA, d_ColsA, info, &size_internal, &size_chol));
		
		gpuErrchk(cudaMalloc((void**)&buffer_gpu, sizeof(char) * size_chol));
		//buffer_cpu = (void*)malloc(sizeof(char) * size_chol);
		//gpuErrchk(cusolverSpDcsrcholSetup(cusolverSpH, n, nnzA, descr_A, d_ValsA, d_RowsA, d_ColsA, 0.0, info));
		
		h_x = (double*)malloc(n * sizeof(double));
		gpuErrchk(cusolverSpDcsrcholFactor(cusolverSpH, n, nnzA, descr_A, d_ValsA, d_RowsA, d_ColsA, info, buffer_gpu));
		gpuErrchk(cusolverSpDcsrcholZeroPivot(cusolverSpH, info, tol, &singularity));


		
		//gpuErrchk(cudaMemcpy(h_x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
		//x.AddData(h_x, n);
		for (int i = 0; i < nrhs; i++)
		{
			gpuErrchk(cudaMemcpy(d_b, h_b + i * n, n * sizeof(double), cudaMemcpyHostToDevice));
			gpuErrchk(cusolverSpDcsrcholSolve(cusolverSpH, n, d_b, d_x, info, buffer_gpu));
			gpuErrchk(cudaMemcpy(h_x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
			x.AddData(h_x, n);
		}
		gpuErrchk(cudaFree(buffer_gpu)); buffer_gpu = nullptr;
		free(buffer_cpu); buffer_cpu = nullptr;
		free(h_x); h_x = nullptr;
		freeGPUMemory();
		return 0;
		}
}