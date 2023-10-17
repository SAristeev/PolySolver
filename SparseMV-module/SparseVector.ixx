export module SparseVector;
import <random>;
import <vector>;
import <string>;
import <format>;
import <fstream>;
import <iostream>;

export namespace SPARSE {
	template<typename VT>
	concept Vector = requires(VT val) {
		{static_cast<double>(val)};
	};

	template<class VT>
	requires Vector<VT>
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

		VT * const getVals() const { return const_cast<VT*>(Vals.at(0).data()); }
		VT * const getVals(size_t i) const { return const_cast<VT*>(Vals.at(i).data()); }

		std::vector<VT> getVector() const { return Vals.at(0); }
		std::vector<VT> getVector(size_t i) const { return Vals.at(i); }
		//void addZeros();
		
		void addVector(const std::vector<VT>& rhs);
		void addRandom();
		void fread(std::string FileName);
		void fprint(std::string FileName) const;
		VT norm(size_t i = 0) const;
	};


	template<class VT>
	requires Vector<VT>
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
	requires Vector<VT>
	void SparseVector<VT>::addVector(const std::vector<VT>& rhs){
		if (nrhs() != 0 && size() != rhs.size()) {
			throw std::exception("Vectors have different dim");
		}
		Vals.push_back(rhs);
	}

	template<class VT>
	requires Vector<VT>
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
	requires Vector<VT>
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
	requires Vector<VT>
	VT SparseVector<VT>::norm(size_t i) const {
		VT res = 0;
		for (int j = 0; j < Vals[i].size(); ++j) {
			res += Vals[i][j] * Vals[i][j];
		}
		return sqrt(res);
	}

}