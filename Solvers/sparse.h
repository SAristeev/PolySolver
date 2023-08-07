#pragma once
#include<fstream>
#include<list>
#include<vector>
#include<string>
#include<format>


namespace SPARSE {

	class SparseMatrix {
	private:
		std::vector<double> Vals;
		std::vector<int> RowsCSR;
		std::vector<int> RowsCOO;
		std::vector<int> Cols;
		int n;
		int nnz;
	public:
		SparseMatrix() { Clear(); }
		~SparseMatrix() { Clear(); }
		void Clear() { Vals.clear(); RowsCSR.clear(); RowsCOO.clear(); Cols.clear(); n = 0; nnz = 0; }
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

		int fprint(int n, std::string FileName);
	};
}