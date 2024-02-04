#pragma once
#include "implementation.hpp"
#include <iostream>
#include <format>
#include <algorithm>

#include <cublas_v2.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <cuda_runtime.h>



class LinearSolver_cuda_cg : public LinearSolver {
public:
	int Solve(const std::vector<double>& vals,
		const std::vector<MKL_INT>& cols,
		const std::vector<MKL_INT>& rows,
		const std::vector<double>& b,
		std::vector<double>& x
	);
};
