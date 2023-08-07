 #include"sparse.h"

namespace SPARSE{
	int SparseVector::SetOnes(int n, int nrhs) {
		this->nrhs++;
		this->n = n;
		std::vector<double> tmp(n * nrhs); 
		std::fill(tmp.begin(), tmp.end(), 1);
		Vals.insert(Vals.end(), tmp.begin(),tmp.end());
		return 0;
	}

	int SparseVector::AddData(std::string FileName)
	{
		std::ifstream file(FileName);

		if (!file.is_open())
		{
			throw std::exception((std::string("No such file: " + FileName + "\nCan't open vector")).c_str());
			return -1;
		}

		std::vector<double> tmp;

		nrhs++;
		file >> n;
		if (n != this->n) {
			return -1;
		}
		tmp.resize(n);

		for (int i = 0; i < n; i++)
		{
			file >> tmp[i];
		}

		Vals.insert(Vals.end(), tmp.begin(), tmp.end());

		file.close();

		return 0;
	}

	int SparseVector::fprint(int n, std::string FileName)
	{
		std::ofstream file(FileName);

		if (!file.is_open())
		{
			return -1;
		}

		file << this->n << std::endl;

		for (int i = 0; i < this->n; i++)
		{	

			file << std::format("{:.20e}", Vals[i + n]); // with zero
			if (i != this->n - 1) {
				file << std::endl;
			}
		}

		file.close();
		return 0;
	}

	int SparseVectorInt::fprint(int n, std::string FileName)
	{
		std::ofstream file(FileName);

		if (!file.is_open())
		{
			return -1;
		}

		file << n << std::endl;

		for (int i = 0; i < n; i++)
		{
			file << Vals[i];
			if (i != n - 1) {
				file << std::endl;
			}
		}

		file.close();
		return 0;
	}

}