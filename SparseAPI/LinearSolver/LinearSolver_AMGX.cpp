#include "LinearSolver_AMGX.h"

namespace SPARSE {
    int LinearSolverAMGX::SetSettingsFromJSON(json settingsJSON) {
        settings.configAMGX = settingsJSON["config"];
        settings.tolerance = settingsJSON["tolerance"];
        settings.max_iter = settingsJSON["max_iter"];
        return 0;
    };

	int LinearSolverAMGX::SolveRightSide(SparseMatrix& A,
		SparseVector& b,
		SparseVector& x) {
        
        //library handles
        AMGX_Mode mode;
        AMGX_config_handle cfg;
        AMGX_resources_handle rsrc;
        AMGX_matrix_handle _A;
        AMGX_vector_handle _b, _x;
        AMGX_solver_handle solver;

        //status handling
        AMGX_SOLVE_STATUS status;
        
        
        A.GetInfo(n, nnzA);
        A.GetDataCSR(&h_ValsA, &h_RowsA, &h_ColsA);
        b.GetData(&h_b);
        x.SetOnes(n, 1);
        x.GetData(&h_x_all);
        h_x = (double*)malloc(n * sizeof(double));
        int nb = 0, nrhs = 0;
        b.GetInfo(nb, nrhs);
        if (nb != n) {
            return -1;
        }

        int block_dimx = 1, block_dimy = 1, block_size;
        

        /* init */
        AMGX_SAFE_CALL(AMGX_initialize());
        /* system */

        mode = AMGX_mode_dDDI;

        AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, settings.configAMGX.c_str()));

        //TODO: set precision
        


        AMGX_resources_create_simple(&rsrc, cfg);
        AMGX_matrix_create(&_A, rsrc, mode);
        AMGX_vector_create(&_x, rsrc, mode);
        AMGX_vector_create(&_b, rsrc, mode);
        AMGX_solver_create(&solver, rsrc, mode, cfg);



        block_size = block_dimx * block_dimy;

        int nrings = 1;
        AMGX_config_get_default_number_of_rings(cfg, &nrings);



        AMGX_SAFE_CALL(AMGX_pin_memory(h_x, n * block_dimx * sizeof(double)));
        AMGX_SAFE_CALL(AMGX_pin_memory(h_b, n * block_dimx * sizeof(double)));
        AMGX_SAFE_CALL(AMGX_pin_memory(h_ColsA, nnzA * sizeof(int)));
        AMGX_SAFE_CALL(AMGX_pin_memory(h_RowsA, (n + 1) * sizeof(int)));
        AMGX_SAFE_CALL(AMGX_pin_memory(h_ValsA, nnzA * block_size * sizeof(double)));

        
        AMGX_matrix_upload_all(_A, n, nnzA, 1, 1, h_RowsA, h_ColsA, h_ValsA, nullptr);
        
        AMGX_vector_bind(_x, _A);
        AMGX_vector_bind(_b, _A);

        AMGX_vector_upload(_b, n, block_dimx, h_b);
        AMGX_vector_set_zero(_x, n, block_dimx);
        AMGX_solver_setup(solver, _A);
        AMGX_solver_solve(solver, _b, _x);
        AMGX_solver_get_status(solver, &status);
        AMGX_vector_download(_x,h_x);

        AMGX_unpin_memory(h_x);
        AMGX_unpin_memory(h_b);
        AMGX_unpin_memory(h_ColsA);
        AMGX_unpin_memory(h_RowsA);
        AMGX_unpin_memory(h_ValsA);

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

        gpuErrchk(cudaDeviceReset());
        return 0;
	}
    
}