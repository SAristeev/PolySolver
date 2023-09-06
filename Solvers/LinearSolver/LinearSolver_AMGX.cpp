#include "LinearSolver_AMGX.h"
#include <sstream>
#include <string>

namespace SOLVER {

    int LinearSolverAMGX_32_d::Solve(const SPARSE::Case<int, double>& rhs) {

        int n = rhs.A.size();
        int nnzA = rhs.A.nnz();

        const int* h_RowsA = rhs.A.getRows();
        const int* h_ColsA = rhs.A.getCols();
        const double* h_ValsA = rhs.A.getVals();

        double* h_b = rhs.b.getVals();
        double* h_x = rhs.x.getVals();

        //library handles
        AMGX_Mode mode;
        AMGX_config_handle cfg;
        AMGX_resources_handle rsrc;
        AMGX_matrix_handle _A;
        AMGX_vector_handle _b, _x;
        AMGX_solver_handle solver;

        //status handling
        AMGX_SOLVE_STATUS status;


        int block_dimx = 1, block_dimy = 1, block_size;


        /* init */
        AMGX_SAFE_CALL(AMGX_initialize());

        /* system */
        mode = AMGX_mode_dDDI;


        std::string cfg_str = "config_version=2, ";
        cfg_str += "exception_handling=1, ";
        cfg_str += "solver(AMGX_SOLVER)=PCG, ";
        cfg_str += "AMGX_SOLVER:preconditioner(AMGX_PRECONDITIONER)=JACOBI_L1, ";
        cfg_str += "AMGX_PRECONDITIONER:max_iters=1, ";
        cfg_str += "AMGX_SOLVER:monitor_residual=1, ";
        cfg_str += "AMGX_SOLVER:obtain_timings=1, ";
        cfg_str += "AMGX_SOLVER:print_solve_stats=1, ";
        cfg_str += "AMGX_PRECONDITIONER:print_grid_stats=1, ";
        cfg_str += "AMGX_SOLVER:max_iters=50000, ";
        cfg_str += "AMGX_SOLVER:tolerance=1e-9, ";
        cfg_str += "AMGX_SOLVER:convergence=RELATIVE_MAX";
        
        
        AMGX_SAFE_CALL(AMGX_config_create(&cfg, cfg_str.c_str()));
        //AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, "C:/LIBS64/AMGX/lib/configs/FIDESYS_FGMRES_STANDART.json"));

        AMGX_resources_create_simple(&rsrc, cfg);
        AMGX_matrix_create(&_A, rsrc, mode);
        AMGX_vector_create(&_x, rsrc, mode);
        AMGX_vector_create(&_b, rsrc, mode);
        AMGX_solver_create(&solver, rsrc, mode, cfg);



        block_size = block_dimx * block_dimy;

        int nrings = 1;
        AMGX_config_get_default_number_of_rings(cfg, &nrings);



        AMGX_SAFE_CALL(AMGX_pin_memory((void*)h_x, n * block_dimx * sizeof(double)));
        AMGX_SAFE_CALL(AMGX_pin_memory((void*)h_b, n * block_dimx * sizeof(double)));
        AMGX_SAFE_CALL(AMGX_pin_memory((void*)h_ColsA, nnzA * sizeof(int)));
        AMGX_SAFE_CALL(AMGX_pin_memory((void*)h_RowsA, (n + 1) * sizeof(int)));
        AMGX_SAFE_CALL(AMGX_pin_memory((void*)h_ValsA, nnzA * block_size * sizeof(double)));


        AMGX_matrix_upload_all(_A, n, nnzA, 1, 1, h_RowsA, h_ColsA, h_ValsA, nullptr);

        AMGX_vector_bind(_x, _A);
        AMGX_vector_bind(_b, _A);

        AMGX_vector_upload(_b, n, block_dimx, h_b);

        AMGX_vector_set_zero(_x, n, block_dimx);
        AMGX_solver_setup(solver, _A);
        AMGX_solver_solve(solver, _b, _x);
        AMGX_solver_get_status(solver, &status);
        AMGX_vector_download(_x, h_x);

        AMGX_unpin_memory((void*)h_x);
        AMGX_unpin_memory((void*)h_b);
        AMGX_unpin_memory((void*)h_ColsA);
        AMGX_unpin_memory((void*)h_RowsA);
        AMGX_unpin_memory((void*)h_ValsA);

        /* destroy resources, matrix, vector and solver */
        AMGX_solver_destroy(solver);
        AMGX_vector_destroy(_x);
        AMGX_vector_destroy(_b);
        AMGX_matrix_destroy(_A);
        AMGX_resources_destroy(rsrc);
        /* destroy config (need to use AMGX_SAFE_CALL after this point) */
        AMGX_SAFE_CALL(AMGX_config_destroy(cfg));
        /* shutdown and exit */
        AMGX_SAFE_CALL(AMGX_finalize());

        cudaDeviceReset();
        return 0;
    }
}