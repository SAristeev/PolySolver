#include "mkl_cg.hpp"

int LinearSolver_mkl_cg::Solve(const std::vector<double>& vals,
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
	std::vector<double> r(n);
	std::vector<double> q(n);
	std::vector<double> p(n);
	const char trans = 'n';
	mkl_cspblas_dcsrgemv(&trans, &n, vals.data(), rows.data(), cols.data(), x.data(), r.data());
	

	// check ||B||
	double b_norm = 0;
	for (int i = 0; i < n; i++) {
		r[i] = b[i] - r[i];
		b_norm += b[i] * b[i];
	}
	b_norm = sqrt(b_norm);

	// settings
	double tolerance = 1e-9;
	int max_iter = 1000;

	// internal
	bool is_convergenced = false;
	double aplha, rho = 0., rho_prev;
	int iter = 0;
	double cur_res = 0; double cur_rel = 0;
	do {
		rho_prev = rho;

		// \rho = r^T * r
		rho = 0;
		for (int i = 0; i < n; i++) {
			rho += r[i] * r[i];
		}
		if (iter == 0) {
			// q0 = r0
			std::copy(r.begin(), r.end(), p.begin());
		}
		else {
			// beta = rho / rho_prev
			double beta = rho / rho_prev;
			for (int i = 0; i < n; i++) {
				// p = r + beta * p 
				p[i] = r[i] + beta * p[i];
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
		
		double cur_res = 0;
		for (int i = 0; i < n; i++) {
			cur_res += r[i] * r[i];
		}
		cur_res = sqrt(cur_res);
		double cur_rel = cur_res / b_norm;
		std::cout << cur_rel << std::endl;

		is_convergenced = cur_rel < tolerance;
		iter++;
	} while (!is_convergenced && iter < max_iter);

	std::cout << "	iter_number: " << iter << std::endl;
	std::cout << "	resudial:    " << cur_rel << std::endl;
	std::cout << "My PCG - IC0: end" << std::endl;
	return 0;
}
