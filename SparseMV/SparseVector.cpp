#include "SparseVector.hpp"

namespace SPARSE {
	template<class VT>
	void SparseVector<VT>::addRandom() {
		if (nrhs() == 0) {
			throw std::exception("Random couldn't be first");
		}
		size_t n = size();
		std::vector<VT> rhs(n);
		std::mt19937 engine;
		engine.seed(std::time(nullptr));
		for (int i = 0; i < n; ++i) { rhs[i] = static_cast<VT>(engine()); }
		Vals.push_back(rhs);
	}

	template<class VT>
	void SparseVector<VT>::addVector(const std::vector<VT>& rhs){
		if (nrhs() != 0 && size() != rhs.size()) {
			throw std::exception("Vectors have different dim");
		}
		Vals.push_back(rhs);
	}

	template<class VT>
	void SparseVector<VT>::fread(std::string FileName) {
		std::ifstream file(FileName);

		if (!file.is_open()) {
			throw std::exception(std::string("No such file: " + FileName + "\nCan't open matrix").c_str());
		}
		size_t n;
		file >> n;
		Vals.resize(1);
		Vals[0].resize(n);
		// TODO: Nan check
		for (size_t i = 0; i < n; i++) { file >> Vals[0][i]; }

		file.close();
	}

	template<class VT>
	void SparseVector<VT>::fprint(std::string FileName) const {
		for (size_t i = 0; i < Vals.size(); ++i) {
			std::ofstream file(FileName + std::to_string(i));

			if (!file.is_open()) {
				throw std::exception((std::string("No such file: " + FileName + "\nCan't open vector")).c_str());
			}
			size_t n = Vals[i].size();
			file << n << std::endl;
			if (n == 0) { file.close(); return; }
			for (int j = 0; j < n; j++) { file << std::format("{:.16e}", Vals[i][j]); if (j != n - 1) { file << std::endl; } }

			file.close();
		}
	}

	template<class VT>
	VT SparseVector<VT>::norm(size_t i) const {
		VT res = 0;
		for (int j = 0; j < Vals[i].size(); ++j) {
			res += Vals[i][j] * Vals[i][j];
		}
		return sqrt(res);
	}

}