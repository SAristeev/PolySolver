#include"kernel.h"
#include<iostream>
#include<map>
#include<exception>
#include "LinearSolver/LinearSolver_PARDISO.h"


namespace KERNEL {
#if defined(WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

	double second(void)
	{
		LARGE_INTEGER t;
		static double oofreq;
		static int checkedForHighResTimer;
		static BOOL hasHighResTimer;

		if (!checkedForHighResTimer) {
			hasHighResTimer = QueryPerformanceFrequency(&t);
			oofreq = 1.0 / (double)t.QuadPart;
			checkedForHighResTimer = 1;
		}
		if (hasHighResTimer) {
			QueryPerformanceCounter(&t);
			return (double)t.QuadPart * oofreq;
		}
		else {
			return (double)GetTickCount() / 1000.0;
		}
	}

#elif defined(__linux__) || defined(__QNX__)
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
	double second(void)
	{
		struct timeval tv;
		gettimeofday(&tv, NULL);
		return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
	}
#endif


	void InitLinearSolvers(SPARSE::ObjectSolverFactory<SPARSE::LinearSolver, SPARSE::SolverID>& LinearFactory) {
#ifdef USE_cuSOLVER
		LinearFactory.add<SPARSE::LinearSolvercuSOLVERSP>(SPARSE::SolverID::cuSOLVERSP);
		LinearFactory.add<SPARSE::LinearSolvercuSOLVERRF>(SPARSE::SolverID::cuSOLVERRF);
		LinearFactory.add<SPARSE::LinearSolvercuSOLVERRF_ALLGPU>(SPARSE::SolverID::cuSOLVERRF_ALLGPU);
#endif // USE_cuSOLVER
#ifdef USE_AMGX
		LinearFactory.add<SPARSE::LinearSolverAMGX>(SPARSE::SolverID::AMGX);
#endif // USE_AMGX
		//LinearFactory.add<SPARSE::LinearSolverPARDISO>(SPARSE::SolverID::PARDISO);
	}
}
