#include "my_pcg.hpp"

int LinearSolver_my_pcg::Solve(const std::vector<double>& vals,
    const std::vector<MKL_INT>& cols,
    const std::vector<MKL_INT>& rows,
    const std::vector<double>& b,
    std::vector<double>& x
    ) {
	std::cout << "My PCG - IC0: start" << std::endl;
	MKL_INT n = rows.size() - 1;
	MKL_INT nnz = rows[n];

	// init x0
	std::fill(x.begin(), x.end(), 1.0);


	// r,z
	std::vector<double> r(n);
	std::vector<double> q(n);
	std::vector<double> p(n);
	std::vector<double> z(n);
	std::vector<double> t(n);
	const char trans = 'n';
	mkl_cspblas_dcsrgemv(&trans, &n, vals.data(), rows.data(), cols.data(), x.data(), r.data());


	// check ||B||
	double b_norm = 0;
	for (int i = 0; i < n; i++) {
		r[i] = b[i] - r[i];
		b_norm += (r[i]) * (r[i]);
	}
	b_norm = sqrt(b_norm);

	// settings
	double tolerance = 1e-9;
	int max_iter = 1000;

	// internal
	bool is_convergenced = false;
	double aplha, beta, rho = 0., rho_prev;
	int iter = 0;
	double cur_res = 0; double cur_rel = 0;
	do {
		// precond
		if (0) {
			const char upper = 'u';
			const char lower = 'l';
			const char trans = 'n';
			const char diag = 'n';
			mkl_cspblas_dcsrtrsv(&lower, &trans, &diag, &n, vals.data(), rows.data(), cols.data(), r.data(), t.data());
			mkl_cspblas_dcsrtrsv(&lower, &trans, &diag, &n, vals.data(), rows.data(), cols.data(), t.data(), z.data());
		}
		else {
			std::copy(r.begin(), r.end(), z.begin());
		}
		rho_prev = rho;

		// \rho = r^T * z
		rho = 0;
		for (int i = 0; i < n; i++) {
			rho += r[i] * z[i];
		}
		if (iter == 0) {
			// q0 = r0
			std::copy(z.begin(), z.end(), p.begin());
		}
		else {
			// beta = rho / rho_prev
			beta = rho / rho_prev;
			for (int i = 0; i < n; i++) {
				// p = z + beta * p 
				//p[i] = z[i] + beta * p[i];
				z[i] += beta * p[i];
				p[i] = z[i];
			}
		}

		// q = A * p
		mkl_cspblas_dcsrgemv(&trans, &n, vals.data(), rows.data(), cols.data(), p.data(), q.data());

		//  dot product
		double pq = 0;
		for (int i = 0; i < n; i++) {
			pq += p[i] * q[i];
		}

		// alpha = rho / (p^T * q)
		aplha = rho / pq;

		for (int i = 0; i < n; i++) {
			// x = x + alpha * q
			x[i] += aplha * p[i];
			// r = r - alpha * q
			r[i] -= aplha * q[i];
		}
		//for (auto num : x) { std::cout << std::setprecision(20) << num << std::endl; } std::cout << std::endl;

		double cur_res = 0;
		for (int i = 0; i < n; i++) {
			cur_res += r[i] * r[i];
		}
		cur_res = sqrt(cur_res);
		cur_rel = cur_res / b_norm;
		std::cout << cur_rel << std::endl;

		is_convergenced = cur_rel < tolerance;
		iter++;
		if (std::isnan(cur_rel)) {
			std::cout << "Nan detected" << std::endl;
			break;
		}
	} while (!is_convergenced && iter < max_iter);

	std::cout << "	iter_number: " << iter << std::endl;
	std::cout << "	resudial:    " << cur_rel << std::endl;
	std::cout << "My PCG - IC0: end" << std::endl;
	return 0;
}
