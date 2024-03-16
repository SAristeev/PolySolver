#include "hypre.hpp"

int LinearSolver_hypre::Solve(const std::vector<double>& vals_,
    const std::vector<MKL_INT>& cols_,
    const std::vector<MKL_INT>& rows_,
    const std::vector<double>& b_,
    std::vector<double>& x_
) {
    int n = rows_.size() - 1;
    int nnz = rows_[n];

    HYPRE_IJMatrix A;
    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_IJVector b;
    HYPRE_ParVector par_b;
    HYPRE_IJVector x;
    HYPRE_ParVector par_x;

    HYPRE_Solver solver, precond;

   

    /* Initialize HYPRE */
    HYPRE_Initialize();

    
    int ilower = 1;
    int iupper = n;
   

    
    /* Create the matrix.
       Note that this is a square matrix, so we indicate the row partition
       size twice (since number of rows = number of cols) */
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);

    /* Choose a parallel csr format storage (see the User's Manual) */
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);

    /* Initialize before setting coefficients */
    HYPRE_IJMatrixInitialize(A);

    int nrows = n;
    std::vector<int> ncols(n);
    std::vector<int> rows(n);
    std::vector<int> cols(nnz);
    for (int i = 0; i < n; i++) 
    {
        ncols[i] = rows_[i + 1] - rows_[i];
        rows[i] = i + 1;
    }

    for (int i = 0; i < nnz; ++i)
    {
        cols[i] = cols_[i] + 1;
    }

    HYPRE_IJMatrixSetValues(A, nrows, ncols.data(), rows.data(), cols.data(), vals_.data());
    

    /* Assemble after setting the coefficients */
    HYPRE_IJMatrixAssemble(A);

    /* Note: for the testing of small problems, one may wish to read
       in a matrix in IJ format (for the format, see the output files
       from the -print_system option).
       In this case, one would use the following routine:
       HYPRE_IJMatrixRead( <filename>, MPI_COMM_WORLD,
                           HYPRE_PARCSR, &A );
       <filename>  = IJ.A.out to read in what has been printed out
       by -print_system (processor numbers are omitted).
       A call to HYPRE_IJMatrixRead is an *alternative* to the
       following sequence of HYPRE_IJMatrix calls:
       Create, SetObjectType, Initialize, SetValues, and Assemble
    */


    /* Get the parcsr matrix object to use */
    HYPRE_IJMatrixGetObject(A, (void**)&parcsr_A);


    /* Create the rhs and solution */
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &b);
    HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(b);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &x);
    HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(x);

    
    HYPRE_IJVectorSetValues(b, n, rows.data(), b_.data());
    HYPRE_IJVectorSetValues(x, n, rows.data(), x_.data());
    


    HYPRE_IJVectorAssemble(b);
    /*  As with the matrix, for testing purposes, one may wish to read in a rhs:
        HYPRE_IJVectorRead( <filename>, MPI_COMM_WORLD,
                                  HYPRE_PARCSR, &b );
        as an alternative to the
        following sequence of HYPRE_IJVectors calls:
        Create, SetObjectType, Initialize, SetValues, and Assemble
    */
    HYPRE_IJVectorGetObject(b, (void**)&par_b);

    HYPRE_IJVectorAssemble(x);
    HYPRE_IJVectorGetObject(x, (void**)&par_x);

    /* Choose a solver and solve the system */

   
    {
        int num_iterations;
        double final_res_norm;

        /* Create solver */
        HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);

        /* Set some parameters (See Reference Manual for more parameters) */
        HYPRE_PCGSetMaxIter(solver, 50000); /* max iterations */
        HYPRE_PCGSetTol(solver, 1e-7); /* conv. tolerance */
        HYPRE_PCGSetTwoNorm(solver, 1); /* use the two norm as the stopping criteria */
        HYPRE_PCGSetPrintLevel(solver, 2); /* print solve info */
        HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */

        /* Now set up the AMG preconditioner and specify any parameters */
        HYPRE_BoomerAMGCreate(&precond);
        HYPRE_BoomerAMGSetPrintLevel(precond, 1); /* print amg solution info */
        HYPRE_BoomerAMGSetCoarsenType(precond, 6);
        HYPRE_BoomerAMGSetOldDefault(precond);
        HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
        HYPRE_BoomerAMGSetNumSweeps(precond, 1);
        HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
        HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */

        /* Set the PCG preconditioner */
        HYPRE_PCGSetPrecond(solver, (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
            (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup, precond);

        /* Now setup and solve! */
        HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
        HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);

        /* Run info - needed logging turned on */
        HYPRE_PCGGetNumIterations(solver, &num_iterations);
        HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

        printf("\n");
        printf("Iterations = %d\n", num_iterations);
        printf("Final Relative Residual Norm = %e\n", final_res_norm);
        printf("\n");


        /* Destroy solver and preconditioner */
        HYPRE_ParCSRPCGDestroy(solver);
        HYPRE_BoomerAMGDestroy(precond);
    }
    

    /* Save the solution for GLVis visualization, see vis/glvis-ex5.sh */

    /* Clean up */
    HYPRE_IJMatrixDestroy(A);
    HYPRE_IJVectorDestroy(b);
    HYPRE_IJVectorDestroy(x);

    /* Finalize HYPRE */
    HYPRE_Finalize();

    return (0);
}
