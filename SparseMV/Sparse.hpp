#pragma once
#include "SparseMatrix.cpp"
#include "SparseVector.cpp"
#include <string>

namespace SPARSE {
	enum class MATRIX_PART {
		FULL = 0,
		UPPER = 1,
		LOWER = 2
	};

	template<class IT, class VT>
	struct Case {
		bool isSymmetric = false;
		MATRIX_PART matrixPart = MATRIX_PART::FULL;
		SparseMatrix<IT, VT> A;
		SparseVector<VT> b;
		SparseVector<VT> x;
		void fread(std::string path);

	};
	template<class IT, class VT>
	void Case<IT, VT>::fread(std::string path) {
		A.freadCSR(path + "/A.txt");
		isSymmetric = A.checkSymmetric();
		b.fread(path + "/b.vec");
		x = b;
	}

	template<class IT, class VT>
	SparseVector<VT> multiplication(const SparseMatrix<IT, VT> & A, const SparseVector<VT> & b) {
		if (A.size() != b.size()) {
			throw std::exception(std::string("SpMV: Matrix and vector have diff dim").c_str());
		}
		size_t n = A.size();
		size_t nrhs = b.nrhs();

		SparseVector<VT> res;
		for (int i = 0; i < nrhs; ++i) {
			res.addVector(A.multiplication(b.getVector(i)));
		}		
		return res;
	}

	template<class VT>
	SparseVector<VT> subtraction(const SparseVector<VT>& lhs, const SparseVector<VT>& rhs) {
		if (lhs.nrhs() != rhs.nrhs() || lhs.size() != rhs.size()) {
			throw std::exception("Vectors have different dim or nrhs");
		}
		SparseVector<VT> res;
		for (int i = 0; i < lhs.nrhs(); ++i) {
			std::vector<VT> tmp(lhs.size());
			for (int j = 0; j < lhs.size(); ++j) {
				tmp[j] = lhs.getVals(i)[j] - rhs.getVals(i)[j];
			}
			res.addVector(tmp);
		}
		return res;
	}
}
