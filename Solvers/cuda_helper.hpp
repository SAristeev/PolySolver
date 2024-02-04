#pragma once
#include <cstdlib>
#include <cublas_v2.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = false)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

inline void gpuAssert(cusolverStatus_t code, const char* file, int line, bool abort = false)
{
	switch (code)
	{
	case CUSOLVER_STATUS_NOT_INITIALIZED:
		fprintf(stderr, "GPUassert: CUSOLVER NOT INITIALIZED %s %d\n", file, line);
		break;
	case CUSOLVER_STATUS_ALLOC_FAILED:
		fprintf(stderr, "GPUassert: CUSOLVER ALLOC FAILED %s %d\n", file, line);
		break;
	case CUSOLVER_STATUS_INVALID_VALUE:
		fprintf(stderr, "GPUassert: CUSOLVER CUSOLVER INVALID VALUE %s %d\n", file, line);
		break;
	case CUSOLVER_STATUS_ARCH_MISMATCH:
		fprintf(stderr, "GPUassert: CUSOLVER CUSOLVER ARCH MISMATCH %s %d\n", file, line);
		break;
	case CUSOLVER_STATUS_EXECUTION_FAILED:
		fprintf(stderr, "GPUassert: CUSOLVER EXECUTION FAILED %s %d\n", file, line);
		break;
	case CUSOLVER_STATUS_INTERNAL_ERROR:
		fprintf(stderr, "GPUassert: CUSOLVER INTERNAL ERROR %s %d\n", file, line);
		break;
	case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		fprintf(stderr, "GPUassert: CUSOLVER MATRIX TYPE NOT SUPPORTED %s %d\n", file, line);
		break;
	default:
		break;
	}
}

inline void gpuAssert(cublasStatus_t code, const char* file, int line, bool abort = false)
{
	if (code != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cublasGetStatusString(code), file, line); \
			if (abort) exit(code);
	}
}

inline void gpuAssert(cusparseStatus_t code, const char* file, int line, bool abort = false)
{
	switch (code) {
	case CUSPARSE_STATUS_NOT_INITIALIZED:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_NOT_INITIALIZED %s %d\n", file, line);
		break;
	case CUSPARSE_STATUS_ALLOC_FAILED:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_ALLOC_FAILED %s %d\n", file, line);
		break;
	case CUSPARSE_STATUS_INVALID_VALUE:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_INVALID_VALUE %s %d\n", file, line);
		break;
	case CUSPARSE_STATUS_ARCH_MISMATCH:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_ARCH_MISMATCH %s %d\n", file, line);
		break;
	case CUSPARSE_STATUS_MAPPING_ERROR:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_MAPPING_ERROR %s %d\n", file, line);
		break;
	case CUSPARSE_STATUS_EXECUTION_FAILED:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_EXECUTION_FAILED %s %d\n", file, line);
		break;
	case CUSPARSE_STATUS_INTERNAL_ERROR:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_INTERNAL_ERROR %s %d\n", file, line);
		break;
	case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		fprintf(stderr, "GPUassert: CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED %s %d\n", file, line);
		break;
	}
}