#include "LinearSolver_AMGCL.h"
#include <sstream>
#include <string>

namespace SOLVER {

    int LinearSolverAMGCL_32_d::Solve(const SPARSE::Case<int, double>& rhs) {

        int n = rhs.A.size();
        int nnzA = rhs.A.nnz();

        const int* h_RowsA = rhs.A.getRows();
        const int* h_ColsA = rhs.A.getCols();
        const double* h_ValsA = rhs.A.getVals();

        double* h_b = rhs.b.getVals();
        double* h_x = rhs.x.getVals();

        
        std::vector<double> x = rhs.x.getVector();
        std::vector<double> b = rhs.b.getVector();

        auto A = std::make_tuple(n,
            amgcl::make_iterator_range(h_RowsA, h_RowsA + n + 1),
            amgcl::make_iterator_range(h_ColsA, h_ColsA + h_RowsA[n]),
            amgcl::make_iterator_range(h_ValsA, h_ValsA + h_RowsA[n])
        );

        
        
        amgcl::profiler<> prof("5x5");
        
        typedef amgcl::backend::builtin<double> SBackend;
        typedef amgcl::backend::builtin<double> PBackend;
        
        typedef amgcl::make_solver<
            amgcl::amg<
            PBackend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
            >,
            amgcl::solver::bicgstab<SBackend>
        > Solver;


        // Initialize the solver with the system matrix:
        prof.tic("setup");
        Solver solve(A);
        prof.toc("setup");

        // Show the mini-report on the constructed solver:
        std::cout << solve << std::endl;

        // Solve the system with the zero initial approximation:
        int iters;
        double error;
        

        prof.tic("solve");
        std::tie(iters, error) = solve(A, b, x);
        prof.toc("solve");

        // Output the number of iterations, the relative error,
        // and the profiling data:
        std::cout << "Iters: " << iters << std::endl
            << "Error: " << error << std::endl
            << prof << std::endl;

        return 0;

    }
}