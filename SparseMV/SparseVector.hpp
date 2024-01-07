#pragma once
#include <random>
#include <vector>
#include <string>
#include <format>
#include <fstream>
#include <iostream>

namespace SPARSE {
	template<class VT>
	class SparseVector final {
	private:
		std::vector<std::vector<VT>> Vals;
	public:
		size_t size() const { return Vals.at(0).size(); }
		size_t nrhs() const { return Vals.size(); }
		void getVals(VT** rhs) { *rhs = this->Vals.at(0).data(); }
		void getVals(size_t i, VT** rhs) { *rhs = this->Vals.at(i).data(); }
		VT const* const getConstVals() const { return Vals.at(0).data(); }
		VT const* const getConstVals(size_t i) const { return Vals.at(i).data(); }

		VT* const getVals() const { return const_cast<VT*>(Vals.at(0).data()); }
		VT* const getVals(size_t i) const { return const_cast<VT*>(Vals.at(i).data()); }

		std::vector<VT> getVector() const { return Vals.at(0); }
		std::vector<VT> getVector(size_t i) const { return Vals.at(i); }
		//void addZeros();

		void addVector(const std::vector<VT>& rhs);
		void addRandom();
		void fread(std::string FileName);
		void fprint(std::string FileName) const;
		VT norm(size_t i = 0) const;
	};
}