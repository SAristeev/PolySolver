#pragma once
#include "LinearSolver_IMPL.h"
#include <mpi.h>
#include <amgx_c.h>
#include <amgx_config.h>
#include <cuda_runtime.h>


namespace SPARSE {

	class LinearSolverAMGX : public LinearSolver {
		int		n;
		int		nnzA;
		int* h_RowsA = nullptr; // CPU <int>    n+1
		int* h_ColsA = nullptr; // CPU <int>    nnzA
		double* h_ValsA = nullptr; // CPU <double> nnzA 
		double* h_x = nullptr; // GPU <double> n
		double* h_x_all = nullptr; // GPU <double> n
		double* h_b = nullptr; // CPU <double> n

		int* d_RowsA = nullptr; // CPU <int>    n+1
		int* d_ColsA = nullptr; // CPU <int>    nnzA
		double* d_ValsA = nullptr; // CPU <double> nnzA 
		double* d_x = nullptr; // GPU <double> n
		double* d_b = nullptr; // CPU <double> n
	public:
		int SolveRightSide(SparseMatrix& A,
			SparseVector& b,
			SparseVector& x) final;
	};
	
}