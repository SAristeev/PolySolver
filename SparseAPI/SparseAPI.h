#pragma once
#include<fstream>
#include<list>
#include<vector>
#include<string>
#include<format>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>

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


inline void gpuAssert(cusparseStatus_t code, const char* file, int line, bool abort = false)
{
	switch (code) {
	case CUSPARSE_STATUS_NOT_INITIALIZED:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_NOT_INITIALIZED %s %d\n", file, line);

	case CUSPARSE_STATUS_ALLOC_FAILED:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_ALLOC_FAILED %s %d\n", file, line);

	case CUSPARSE_STATUS_INVALID_VALUE:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_INVALID_VALUE %s %d\n", file, line);

	case CUSPARSE_STATUS_ARCH_MISMATCH:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_ARCH_MISMATCH %s %d\n", file, line);

	case CUSPARSE_STATUS_MAPPING_ERROR:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_MAPPING_ERROR %s %d\n", file, line);

	case CUSPARSE_STATUS_EXECUTION_FAILED:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_EXECUTION_FAILED %s %d\n", file, line);

	case CUSPARSE_STATUS_INTERNAL_ERROR:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_INTERNAL_ERROR %s %d\n", file, line);

	case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED %s %d\n", file, line);
	}
}

namespace SPARSE {

	class SparseMatrix {
	private:
		std::vector<double> Vals;
		std::vector<int> RowsCSR;
		std::vector<int> RowsCOO;
		std::vector<int> Cols;
		int n;
		int nnz;
		void Clear() { Vals.clear(); RowsCSR.clear(); RowsCOO.clear(); Cols.clear(); n = 0; nnz = 0; }
	public:
		SparseMatrix() { Clear(); }
		~SparseMatrix() { Clear(); }
		void CopyData(double* array, int len) { Clear(); Vals.insert(Vals.end(), &array[0], &array[len]); }
		void GetInfo(int&n, int&nnz) { n = this->n; nnz = this->nnz; }
		void GetDataCSR(double**Vals, int** RowsCSR, int**Cols) { *Vals = this->Vals.data(); *RowsCSR = this->RowsCSR.data(); *Cols = this->Cols.data(); }
		void SetDataCSR(double* Vals, int* RowsCSR, int* Cols, int n, int nnz) {
			Clear(); this->n = n; this->nnz = nnz;
			this->Vals.insert(this->Vals.end(), &Vals[0], &Vals[nnz]); 
			this->RowsCSR.insert(this->RowsCSR.end(), &RowsCSR[0], &RowsCSR[n + 1]); 
			this->Cols.insert(this->Cols.end(), &Cols[0], &Cols[nnz]); }

		int freadCSR(std::string FileName);
		int freadMatrixMarket(std::string FileName);
		int fprintCSR(std::string FileName);
	};

	class SparseVector {
	private:
		std::vector<double> Vals;		
		int n;
		int nrhs;
		
	public:
		SparseVector() { Clear(); }
		~SparseVector() { Clear(); }
		int SetOnes(int n, int nrhs);
		void Clear() { Vals.clear(); n = 0; nrhs = 0; }
		//void SetNrhs(int nrhs) { this->nrhs = nrhs; }
		//void SetData(double* array, int len) { Clear(); n = len; nrhs = 1; Vals.insert(Vals.end(), &array[0], &array[len]); }
		int AddData(double* array, int len) {
			if (n != 0 && n != len) {
				return -1;
			}
			if (n == 0 && nrhs == 0) 
			{
				n = len; 
			} 
			nrhs++;
			Vals.insert(Vals.end(), &array[0], &array[len]);
			return 0; 
			}
		int AddData(std::string FileName);

		int GetData(double** Vals) { *Vals = this->Vals.data(); return 0; }
		int GetData(size_t i, double** Vals) { *Vals = this->Vals.data() + n*i; return 0; }
		int GetInfo(int &n, int& nrhs) { n = this->n; nrhs = this->nrhs; return 0; }

		//int fread(std::string FileName);
		int fprint(int n, std::string FileName);
	};



	class SparseVectorInt {
	private:
		std::vector<int> Vals;
		int n;
		int nrhs;
		void Clear() { Vals.clear(); n = 0; nrhs = 0; }
	public:
		SparseVectorInt() { Clear(); }
		~SparseVectorInt() { Clear(); }
		int SetOnes(int n, int nrhs);

		//void SetNrhs(int nrhs) { this->nrhs = nrhs; }
		//void SetData(double* array, int len) { Clear(); n = len; nrhs = 1; Vals.insert(Vals.end(), &array[0], &array[len]); }
		int AddData(int* array, int len) {
			if (n != 0 && n != len) {
				return -1;
			}
			if (n == 0 && nrhs == 0)
			{
				n = len;
			}
			nrhs++;
			Vals.insert(Vals.end(), &array[0], &array[len]);
			return 0;
		}
		int AddData(std::string FileName);

		int GetData(int** Vals) { *Vals = this->Vals.data(); return 0; }
		int GetInfo(int& n, int& nrhs) { n = this->n; nrhs = this->nrhs; return 0; }

		//int fread(std::string FileName);
		int fprint(int n, std::string FileName);
	};
}