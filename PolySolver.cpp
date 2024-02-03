#include "PolySolver.hpp"

void CreateSolvers(const json& settings, std::set<std::unique_ptr<LinearSolver>>& solvers) {

	ObjectSolverFactory<LinearSolver, SolverID> LinearFactory;
	InitLinearSolvers(LinearFactory);

	for (auto& solver : settings["LinearProblem"]["solvers"]) {
		if (polysolver::table.find(solver) != polysolver::table.end()) {
			solvers.emplace(LinearFactory.get(polysolver::table.at(solver)));
		}
		else {
			throw std::runtime_error("Can't find solver: " + solver);
		}
    }
}


void freadCSR(std::string FileName, std::vector<double>& vals, std::vector<MKL_INT>& cols, std::vector<MKL_INT>& rows) {
	std::ifstream file(FileName);

	if (!file.is_open()) {
		throw std::exception(std::string("No such file: " + FileName + "\nCan't open matrix").c_str());
	}
	int n, nz;
	file >> n >> nz;

	rows.resize(n + 1);
	cols.resize(nz);
	vals.resize(nz);
	// TODO: Nan check
	for (size_t i = 0; i < n + 1; i++) { file >> rows[i]; }
	for (size_t i = 0; i < nz; i++) { file >> cols[i]; }
	for (size_t i = 0; i < nz; i++) { file >> vals[i]; }

	file.close();
}

void fread(std::string FileName, std::vector<double>& vals) {
	std::ifstream file(FileName);

	if (!file.is_open()) {
		throw std::exception(std::string("No such file: " + FileName + "\nCan't open vector").c_str());
	}
	size_t n;
	file >> n;
	vals.resize(n);
	for (size_t i = 0; i < n; i++) { file >> vals[i]; }

	file.close();
}

double resudial(const std::vector<double>& vals,
const std::vector<MKL_INT>& cols,
const std::vector<MKL_INT>& rows,
const std::vector<double>& b,
const std::vector<double>& x){
	MKL_INT n = rows.size() - 1;
	MKL_INT nnz = rows[n];
	double alpha = 1.;

	const char trans = 'n';
	std::vector<double> r(b.size());
	
	mkl_cspblas_dcsrgemv(&trans, &n, vals.data(), rows.data(), cols.data(), x.data(), r.data());
	
	double res = 0;
	for (int i = 0; i < b.size(); i++) {
		res+= (r[i] - b[i]) * (r[i] - b[i]);
	}
	return sqrt(res);
}