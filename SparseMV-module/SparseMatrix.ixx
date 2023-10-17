export module SparseMatrix;

import <vector>;
import <string>;
import <format>;
import <fstream>;
import <iostream>;

export namespace SPARSE {
	template<typename IT, typename VT>
	concept Matrix = requires(IT ind, VT val){
		{static_cast<int>(ind)};  {static_cast<double>(val)};
	};

	template<class IT, class VT>
	requires Matrix<IT, VT>
	class SparseMatrix final {
	private:
		std::vector<VT> Vals;
		std::vector<IT> RowsCSR;
		std::vector<IT> Cols;
		size_t n = 0;
		size_t nz = 0;
	public:
		size_t size() const { return n; }
		size_t nnz() const { return nz; }
		//void getVals(VT** rhs) { *rhs = this->Vals.data(); }
		//void getRows(IT** rhs) { *rhs = this->RowsCSR.data(); }
		//void getCols(IT** rhs) { *rhs = this->Cols.data(); }
		VT const * const getVals() const { return Vals.data(); }
		IT const * const getRows() const { return RowsCSR.data(); }
		IT const * const getCols() const { return Cols.data(); }
		std::vector<VT>& getValsVals() const { return Vals; }
		std::vector<IT>& getValsRows() const { return RowsCSR; }
		std::vector<IT>& getValsCols() const { return Cols; }
		void freadCSR(std::string FileName);
		void fprintCSR(std::string FileName) const;

		VT norm() const;
		bool checkSymmetric() const;
		std::vector<VT> multiplication(const std::vector<VT>& b) const;
	};

	template<class IT, class VT>
	requires Matrix<IT, VT>
	void SparseMatrix<IT, VT>::freadCSR(std::string FileName) {
		std::ifstream file(FileName);

		if (!file.is_open()) {
			throw std::exception(std::string("No such file: " + FileName + "\nCan't open matrix").c_str());
		}

		file >> n >> nz;

		RowsCSR.resize(n + 1);
		Cols.resize(nz);
		Vals.resize(nz);
		// TODO: Nan check
		for (size_t i = 0; i < n + 1; i++) { file >> RowsCSR[i]; }
		for (size_t i = 0; i < nz; i++) { file >> Cols[i]; }
		for (size_t i = 0; i < nz; i++) { file >> Vals[i]; }

		file.close();
	}

	template<class IT, class VT>
	requires Matrix<IT, VT>
	void SparseMatrix<IT, VT>::fprintCSR(std::string FileName) const {
		std::ofstream file(FileName);

		if (!file.is_open()) {
			throw std::exception((std::string("No such file: " + FileName + "\nCan't open vector")).c_str());
		}

		file << n << " " << nz << std::endl;
		if (n == 0 || nz == 0) { file.close(); return; }
		for (int i = 0; i < n + 1; i++) { file << RowsCSR[i] << std::endl; }
		for (int i = 0; i < nz; i++) { file << Cols[i] << std::endl; }
		for (int i = 0; i < nz; i++) { file << std::format("{:.16e}", Vals[i]); if (i != nz - 1) { file << std::endl; } }

		file.close();
	}


	template<class IT, class VT>
	requires Matrix<IT, VT>
	VT SparseMatrix<IT, VT>::norm() const {
		size_t j = 0;
		VT max = 0, cur = 0, currow = 0;
		while (j < nz) {
			if (j == RowsCSR[currow]) {
				max = max > cur ? max : cur;
				cur = Vals[j];
				currow++;
			}
			else {
				cur += Vals[j];
			}
			j++;
		}
		return max;
	}

	template<class IT, class VT>
	requires Matrix<IT, VT>
	bool SparseMatrix<IT, VT>::checkSymmetric() const {
		VT curRowSumRes = 0, normRes = 0;
		VT curRowSum = 0, norm = 0;
		IT curRow = 0;
		for (IT i = 0; i < nz; i++) {
			if (i >= RowsCSR[curRow + 1]) {
				curRow++;
				normRes = curRowSumRes > normRes ? curRowSumRes : normRes;
				norm = curRowSum > norm ? curRowSum : norm;
				curRowSumRes = 0;
				curRowSum = 0;

			}
			IT len = RowsCSR[Cols[i] + 1] - RowsCSR[Cols[i]];
			bool isFind = false;
			for (int j = 0; j < len; j++) {
				if (Cols[RowsCSR[Cols[i]] + j] == curRow) {
					curRowSumRes += abs(Vals[i] - Vals[RowsCSR[Cols[i]] + j]);
					isFind = true;
					break;
				}
			}
			if (!isFind) {
				curRowSumRes += abs(Vals[i]);
			}
			curRowSum += abs(Vals[i]);
		}
		VT normTolerance = 1e-6;
		if (normRes / norm < normTolerance) {
			return true;
		}
		return false;
	}

	template<class IT, class VT>
	requires Matrix<IT, VT>
	std::vector<VT> SparseMatrix<IT, VT>::multiplication(const std::vector<VT>& b) const {
		if (this->n != b.size()) {
			throw std::exception(std::string("SpMV: Matrix and vector have diff dim").c_str());
		}
		std::vector<VT> x(this->n, 0);
		for (int i = 0; i < this->n; ++i) {
			for (int j = this->RowsCSR[i]; j < this->RowsCSR[i + 1]; ++j) {
				x[i] += this->Vals[j] * b[this->Cols[j]];
			}
		}
		return x;
	}	
}
