#include "LinearSolver_AMGX.h"
#include <sstream>
namespace SPARSE {

    void applyAMGXSettings(AMGX_config_handle& cfg){
        AMGX_config_add_parameters(&cfg, "config_version=2, print_solve_stats=1");
        AMGX_config_add_parameters(&cfg, "config_version=2, solver(MY)=PCG");
        AMGX_config_add_parameters(&cfg, "config_version=2, MY:print_solve_stats=1");
        AMGX_config_add_parameters(&cfg, "config_version=2, MY:obtain_timings=1");
        AMGX_config_add_parameters(&cfg, "config_version=2, MY:monitor_residual=1");
        AMGX_config_add_parameters(&cfg, "config_version=2, MY:convergence=ABSOLUTE");
        AMGX_config_add_parameters(&cfg, "config_version=2, MY:tolerance=1e-9");
        AMGX_config_add_parameters(&cfg, "config_version=2, MY:norm=L2");
        AMGX_config_add_parameters(&cfg, "config_version=2, MY:max_iters=100000");
        AMGX_config_add_parameters(&cfg, "config_version=2, MY:preconditioner(MY_PREC)=NOSOLVER");        
    }

    int LinearSolverAMGX::SetSettingsFromJSON(json settingsJSON) {

        try{
            nrhs = settingsJSON["n_rhs"];
        }
        catch (const std::exception& ex) {
            std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
            std::cerr << ex.what() << std::endl;
            throw std::exception("Invalid n_rhs");
        }
        
        try {
            settings.configs_path = settingsJSON["AMGX_settings"]["configs_path"];
        }
        catch (const std::exception& ex) {
            std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
            std::cerr << ex.what() << std::endl;
            throw std::exception("Invalid AMGX configs_path");
        }

        try {
            settings.n_configs = settingsJSON["AMGX_settings"]["n_configs"];
        }
        catch (std::exception &ex) {
            std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
            std::cerr << ex.what() << std::endl;
            std::cerr << "Setting default setting [LinearProblem][AMGX_settings][n_configs] = 1" << std::endl;
            settings.n_configs = settingsJSON["AMGX_settings"]["n_configs"] = 1;
        }
        
        try{
            int cur_config = 0;
            for (auto config : settingsJSON["AMGX_settings"]["configs"]) {
                if (cur_config >= settings.n_configs) {
                    break;
                }
                settings.configsAMGX.push_back(config);
                cur_config++;
            }
            //this->AddConfigToName(settings.configsAMGX.at(this->curConfig));
        }
        catch (const std::exception &ex)
        {
            std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
            throw std::exception("Invalid AMGX configname");
        }

        try{
            settings.tolerance = settingsJSON["AMGX_settings"]["tolerance"];
        }
        catch (const std::exception& ex){
            std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
            std::cerr << ex.what() << std::endl;
            std::cerr << "Setting default setting [LinearProblem][AMGX_settings][tolerance] = 1e-6" << std::endl;
            settings.tolerance = 1e-6;
        }

        try{
            settings.max_iter = settingsJSON["AMGX_settings"]["max_iter"];
        }
        catch (const std::exception& ex){
            std::cerr << "Error at parameter in JSON PolySolver config" << std::endl;
            std::cerr << ex.what() << std::endl;
            std::cerr << "Setting default setting [LinearProblem][AMGX_settings][max_iter] = 100" << std::endl;
            settings.max_iter = 100;
        }
        return 0;
    };

	int LinearSolverAMGX::SolveRightSide(SparseMatrix& A,
		SparseVector& b,
		SparseVector& x) {

        A.GetInfo(n, nnzA);
        A.GetDataCSR(&h_ValsA, &h_RowsA, &h_ColsA);

        b.GetData(&h_b);

        int nb = 0, nrhsdum = 0;
        b.GetInfo(nb, nrhsdum);
        if (nb != n) {
            return -1;
        }
        x.SetOnes(n, nrhs);
        x.GetData(&h_x);

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


        AMGX_SAFE_CALL(AMGX_config_create(&cfg, "config_version=2"));

        //applyAMGXSettings(cfg);

        std::string configName = settings.configs_path + "/" + settings.configsAMGX[this->curConfig] + ".json";
        AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, configName.c_str()));
        
        
        std::ostringstream tolstream;
        tolstream << settings.tolerance;
        
        std::string tolstr = "config_version=2, main: tolerance=" + tolstream.str();
        std::string maxitstr = "config_version=2, main: max_iters=" + std::to_string(settings.max_iter);
        
        AMGX_config_add_parameters(&cfg, tolstr.c_str());
        AMGX_config_add_parameters(&cfg, maxitstr.c_str());
        AMGX_config_add_parameters(&cfg, "config_version=2, main: convergence=ABSOLUTE");
        //AMGX_config_add_parameters(&cfg, "config_version=2, main: obtain_timings=0");
        //AMGX_config_add_parameters(&cfg, "config_version=2, main: print_solve_stats=0");

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
        AMGX_vector_download(_x, h_x);

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

        cudaDeviceReset();
        return 0;
	}
    
}