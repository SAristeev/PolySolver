#include "my_cg.hpp"

int LinearSolver_my_cg::Solve(const std::vector<double>& vals,
    const std::vector<MKL_INT>& cols,
    const std::vector<MKL_INT>& rows,
    const std::vector<double>& b,
    std::vector<double>& x
    ) {
	std::cout << "My CG - IC0: start" << std::endl;
	MKL_INT n = rows.size() - 1;
	MKL_INT nnz = rows[n];

	// init x0
	std::fill(x.begin(), x.end(), 1.0);


	// r,z
	std::vector<double> r(b.size());
	std::vector<double> z(b.size());
	const char trans = 'n';
	mkl_cspblas_dcsrgemv(&trans, &n, vals.data(), rows.data(), cols.data(), x.data(), r.data());
	// z0 = r0
	std::copy(r.begin(), r.end(), z.begin());

	// Az 
	std::vector<double> Az(b.size());

	// check ||B||
	double b_norm = 0;
	for (int i = 0; i < n; i++) {
		r[i] = -b[i];
		b_norm += b[i] * b[i];
	}
	b_norm = sqrt(b_norm);

	// settings
	double tolerance = 1e-2;
	int max_iter = 100;

	// internal
	bool is_convergenced = false;
	double aplha;
	int iter = 0;
	do {
		// 1. find alpha
		// compute inner prod
		double rho = 0;
		for (int i = 0; i < n; i++) {
			rho += r[i] * r[i];
		}

		// Az
		mkl_cspblas_dcsrgemv(&trans, &n, vals.data(), rows.data(), cols.data(), z.data(), Az.data());
		// (Az, z) dot product
		double az = 0;
		for (int i = 0; i < n; i++) {
			az += Az[i] * z[i];
		}

		// alpha
		aplha = rho / az;

		double rr = 0;
		for (int i = 0; i < n; i++) {
			// 2. correct x
			x[i] += aplha * z[i];
			// 3. correct r
			r[i] -= aplha * z[i];
			// 4. compute rr
			rr += r[i];
		}
		// beta
		double beta = rr / rho;
		double cur_res = 0;
		// z(k) = r(k) + beta * z(k-1)
		for (int i = 0; i < n; i++) {
			z[i] = r[i] + beta * z[i];
			cur_res += r[i] * r[i];
		}

		cur_res = sqrt(cur_res);
		double cur_rel = cur_res / b_norm;
		std::cout << cur_rel << std::endl;

		is_convergenced = cur_rel < tolerance;
		iter++;
	} while (!is_convergenced && iter < max_iter);

	std::cout << "My CG - IC0: end" << std::endl;
	return 0;
}
