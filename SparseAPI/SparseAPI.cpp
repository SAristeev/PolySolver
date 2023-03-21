#include"SparseAPI.h"


namespace SPARSE{
	int SparseMatrix::freadCSR(std::string FileName)
	{
		Clear();
		std::ifstream file(FileName);

		if (!file.is_open()) 
		{
			return -1;
		}

		file >> n >> nnz;

		RowsCSR.resize(n + 1);
		Cols.resize(nnz);
		Vals.resize(nnz);

		for (int i = 0; i < n + 1; i++)
		{
			file >> RowsCSR[i];
		}
		for (int i = 0; i < nnz; i++)
		{
			file >> Cols[i];
		}

		for (int i = 0; i < nnz; i++)
		{
			file >> Vals[i];
		}

		file.close();

		return 0;
	}

	int SparseMatrix::freadMatrixMarket(std::string FileName) // not ready
	{
		Clear();
		std::ifstream file(FileName);

		if (!file.is_open())
		{
			return -1;
		}
		
		if (file.peek() == '%')
		{
			std::string tmp;
			std::getline(file,tmp);
		}
		//double n1;
		file >> n >> n >> nnz;


		RowsCSR.resize(n + 1);
		RowsCOO.resize(nnz);
		Cols.resize(nnz);
		Vals.resize(nnz);

		for (int i = 0; i < nnz; i++)
		{
			file >> Cols[i] >> RowsCOO[i] >> Vals[i];
		}
		for (int i = 0; i < nnz; i++)
		{
			Cols[i]--;
		}
		RowsCSR[0] = 0;
		int tmp = 0;
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < nnz; j++) 
			{
				if (RowsCOO[j] == i + 1) 
				{
					tmp++;
				}
			}
			RowsCSR[i + 1] = tmp;
		}
		file.close();
		return 0;
	}


	int SparseMatrix::fprintCSR(std::string FileName)
	{
		std::ofstream file(FileName);

		if (!file.is_open())
		{
			return -1;
		}

		file << n << " " << nnz << std::endl;

		for (int i = 0; i < n + 1; i++)
		{
			file << RowsCSR[i] << std::endl;
		}
		for (int i = 0; i < nnz; i++)
		{
			file << Cols[i] << std::endl;
		}
		for (int i = 0; i < nnz; i++)
		{
			file << std::format("{:.16e}", Vals[i]);
			if (i != nnz - 1) {
				file << std::endl;
			}
		}

		file.close();
		return 0;
}

	int SparseVector::SetOnes(int n, int nrhs) {
		Clear(); 
		nrhs++;
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



	/*int SparseVector::fread(std::string FileName)
	{
		Clear();
		std::ifstream file(FileName);

		if (!file.is_open())
		{
			return -1;
		}

		file >> n;

		Vals.resize(n);

		for (int i = 0; i < n; i++)
		{
			file >> Vals[i];
		}

		file.close();

		return 0;
	}*/
	
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