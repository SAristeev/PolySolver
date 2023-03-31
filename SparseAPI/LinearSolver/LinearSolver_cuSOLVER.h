#pragma once
#include "LinearSolver_IMPL.h"

#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusolverRf.h>



namespace SPARSE {

	
	class LinearSolvercuSOLVERSP : public LinearSolver {
	private:
		double* d_Vals, *d_x, *d_b;
		int* d_Rows, * d_Cols;
		double* h_Vals, *h_b, *h_x;
		int* h_Rows, * h_Cols;
		int n, nnz;
	public:
		LinearSolvercuSOLVERSP(): LinearSolver("cuSOLVERSP") {}
		virtual int SolveRightSide(SparseMatrix &A,
			SparseVector &b,
			SparseVector &x) final;
		virtual int SetSettingsFromJSON(json settings) final { return 0; };
	};

	class LinearSolvercuSOLVERRF : public LinearSolver {
	private:
		int n;

		int		nnzA;
		int*    d_RowsA = nullptr; // GPU <int>    n+1
		int*    d_ColsA = nullptr; // GPU <int>    nnzA
		double* d_ValsA = nullptr; // GPU <double> nnzA
		int*    h_RowsA = nullptr; // CPU <int>    n+1
		int*    h_ColsA = nullptr; // CPU <int>    nnzA
		double* h_ValsA = nullptr; // CPU <double> nnzA
		
		int		nnzL;
		int*    d_RowsL = nullptr; // GPU <int>    n+1
		int*    d_ColsL = nullptr; // GPU <int>    nnzL
		double* d_ValsL = nullptr; // GPU <double> nnzL
		int*    h_RowsL = nullptr; // CPU <int>    n+1
		int*    h_ColsL = nullptr; // CPU <int>    nnzL
		double* h_ValsL = nullptr; // CPU <double> nnzL
		
		int		nnzU;
		int*    d_RowsU = nullptr; // GPU <int>    n+1
		int*    d_ColsU = nullptr; // GPU <int>    nnzU
		double* d_ValsU = nullptr; // GPU <double> nnzU
		int*    h_RowsU = nullptr; // CPU <int>    n+1
		int*    h_ColsU = nullptr; // CPU <int>    nnzU
		double* h_ValsU = nullptr; // CPU <double> nnzU
		
		//int*    d_P     = nullptr; // GPU <int>    n
		int*    h_P     = nullptr; // CPU <int>    n
		
		//int*    d_Q     = nullptr; // GPU <int>    n
		int*    h_Q     = nullptr; // CPU <int>    n

		double* d_x     = nullptr; // GPU <double> n
		double* h_x     = nullptr; // CPU <double> n

		double* d_b     = nullptr; // GPU <double> n
		double* h_b     = nullptr; // CPU <double> n

		double* d_y     = nullptr; // GPU <double> n
		double* h_y     = nullptr; // CPU <double> n

	public:
		LinearSolvercuSOLVERRF() : LinearSolver("cuSOLVER CPU Factor + cuSPARSE") {}
		int SolveRightSide(SparseMatrix& A,
			SparseVector& b,
			SparseVector& x) final;
		virtual int SetSettingsFromJSON(json settings) final { return 0; };
		void freeGPUMemory();
	};


	class LinearSolvercuSOLVERRF_ALLGPU : public LinearSolver {
	private:
		int n;

		int		nnzA;
		int* d_RowsA = nullptr; // GPU <int>    n+1
		int* d_ColsA = nullptr; // GPU <int>    nnzA
		double* d_ValsA = nullptr; // GPU <double> nnzA
		
		int* h_RowsA = nullptr; // CPU <int>    n+1
		int* h_ColsA = nullptr; // CPU <int>    nnzA
		double* h_ValsA = nullptr; // CPU <double> nnzA

		//int*    d_P     = nullptr; // GPU <int>    n
		int* h_P = nullptr; // CPU <int>    n

		//int*    d_Q     = nullptr; // GPU <int>    n
		int* h_Q = nullptr; // CPU <int>    n

		double* d_x = nullptr; // GPU <double> n
		double* h_x = nullptr; // CPU <double> n

		double* d_b = nullptr; // GPU <double> n
		double* h_b = nullptr; // CPU <double> n

		double* d_y = nullptr; // GPU <double> n
		double* h_y = nullptr; // CPU <double> n

	public:
		LinearSolvercuSOLVERRF_ALLGPU() : LinearSolver("cuSOLVER CPU Factor + cuSPARSE") {}
		int SolveRightSide(SparseMatrix& A,
			SparseVector& b,
			SparseVector& x) final;
		virtual int SetSettingsFromJSON(json settings) final { return 0; };
		void freeGPUMemory();
	};

}