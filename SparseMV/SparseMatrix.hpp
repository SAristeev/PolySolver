#pragma once
#include <vector>
#include <string>
#include <format>
#include <fstream>
#include <iostream>

namespace SPARSE {
	template<class IT, class VT>
	class SparseMatrix final {
	private:
		std::vector<VT> Vals;
		std::vector<IT> RowsCSR;
		std::vector<IT> Cols;
		size_t n = 0;
		size_t nz = 0;
	public:
		size_t size() const { return n; }
		const size_t& getn() const { return n; }
		size_t nnz() const { return nz; }
		//void getVals(VT** rhs) { *rhs = this->Vals.data(); }
		//void getRows(IT** rhs) { *rhs = this->RowsCSR.data(); }
		//void getCols(IT** rhs) { *rhs = this->Cols.data(); }
		VT const* const getVals() const { return Vals.data(); }
		IT const* const getRows() const { return RowsCSR.data(); }
		IT const* const getCols() const { return Cols.data(); }
		const std::vector<VT>& getValsVals() const { return Vals; }
		const std::vector<IT>& getValsRows() const { return RowsCSR; }
		const std::vector<IT>& getValsCols() const { return Cols; }
		void freadCSR(std::string FileName);
		void fprintCSR(std::string FileName) const;

		VT norm() const;
		bool checkSymmetric() const;
		std::vector<VT> multiplication(const std::vector<VT>& b) const;
	};
}