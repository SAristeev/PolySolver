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

        /*int64_t* h_ColsA64_t = (int64_t*)malloc(nnzA * sizeof(int64_t));
        for (int i = 0; i < nnzA; i++) {
            h_ColsA64_t[i] = h_ColsA[i];
        }*/

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

        //int64_t nlocal64 = n / nranks;
        //int last = n % nranks;

        //if (rank <= last - 1) {
        //    nlocal64++;
        //}
        //int64_t* partition_offsets64 = (int64_t*)malloc((nranks + 1) * sizeof(int64_t));
        //int* partition_offsets = (int*)malloc((nranks + 1) * sizeof(int));
        //partition_offsets64[0] = 0;
        //
        //
        //int* locals = (int*)malloc(nranks * sizeof(int));
        //locals[0] = int(nlocal64);
        //MPI_Allgather(&nlocal64, 1, MPI_INT64_T, &partition_offsets64[1], 1, MPI_INT64_T, mpi_comm);
        //for (int i = 2; i < nranks + 1; ++i) {
        //    partition_offsets64[i] += partition_offsets64[i - 1];
        //    locals[i - 1] = partition_offsets64[i] - partition_offsets64[i - 1];

        //}
        //locals[1] = partition_offsets64[2] - partition_offsets64[1];
        //int nglobal = partition_offsets64[nranks]; // last element always has global number of rows

        //AMGX_distribution_handle dist;
        //AMGX_distribution_create(&dist, cfg);
        //AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_OFFSETS, partition_offsets64);
        //AMGX_matrix_upload_distributed(_A, nglobal, int(nlocal64), nnzA, block_dimx, block_dimy, h_RowsA, h_ColsA64_t, h_ValsA, NULL, dist);
        //AMGX_distribution_destroy(dist);
        AMGX_matrix_upload_all(_A, n, nnzA, 1, 1, h_RowsA, h_ColsA, h_ValsA, nullptr);
        


        AMGX_vector_bind(_x, _A);
        AMGX_vector_bind(_b, _A);

        AMGX_vector_upload(_b, n, block_dimx, h_b);
        AMGX_vector_set_zero(_x, n, block_dimx);

        //MPI_Barrier(mpi_comm);
        
        AMGX_solver_setup(solver, _A);
        AMGX_solver_solve(solver, _b, _x);
        //MPI_Barrier(mpi_comm);
        
        AMGX_solver_get_status(solver, &status);


        AMGX_vector_download(_x,h_x);

        /*MPI_Barrier(mpi_comm);
        for (int i = 0; i < nranks + 1; i++) {
            partition_offsets[i] = partition_offsets64[i];
        }
        
        MPI_Gatherv(h_x, nlocal64, MPI_DOUBLE, h_x_all, locals, partition_offsets, MPI_DOUBLE, 0, mpi_comm);
     
        
        free(partition_offsets64);
        free(partition_offsets);*/
        //MPI_Barrier(mpi_comm);
        AMGX_unpin_memory(h_x);
        AMGX_unpin_memory(h_b);
        AMGX_unpin_memory(h_ColsA);
        AMGX_unpin_memory(h_RowsA);
        AMGX_unpin_memory(h_ValsA);
        /*std::string FileName("err_");
        std::ofstream file(FileName + std::to_string(rank) + "_.txt");

        if (!file.is_open())
        {
            return -1;
        }

        file << n << std::endl;

        for (int i = 0; i < n; i++)
        {
            file << std::format("{:.20e}", h_x_all[i]);
            if (i != n - 1) {
                file << std::endl;
            }
        }*/

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

        //MPI_Finalize();
        gpuErrchk(cudaDeviceReset());
        return 0;
	}
    
}