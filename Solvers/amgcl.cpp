#include "amgcl.hpp"

int LinearSolver_AMGCL::Solve(const std::vector<double>& vals,
    const std::vector<MKL_INT>& cols,
    const std::vector<MKL_INT>& rows,
    const std::vector<double>& b,
    std::vector<double>& x
) {
    // The profiler:
    amgcl::profiler<> prof("poisson3Db");
    //ptrdiff_t Av;
    int n = rows.size() - 1;
    auto A = std::tie(n, rows, cols, vals);
    typedef amgcl::backend::builtin<double> SBackend;
    typedef amgcl::backend::builtin<double> PBackend;

    typedef amgcl::make_solver<
        amgcl::amg<
        PBackend,
        amgcl::coarsening::ruge_stuben,
        amgcl::relaxation::spai0
        >,
        amgcl::solver::cg<SBackend>
    > Solver;

    Solver::params prm;
    //prm.solver.M = 100;
    prm.solver.tol = 1e-3;
    prm.solver.maxiter = 10000;
    prm.solver.verbose = true;
    // Initialize the solver with the system matrix:
    prof.tic("setup");
    Solver solve(A, prm);
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
