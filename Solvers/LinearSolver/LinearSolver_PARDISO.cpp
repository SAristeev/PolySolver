#include "LinearSolver_PARDISO.h"
#include <iostream>

namespace SOLVER {
    const char* ErrorByCodePARDISO(MKL_INT Code) {
        const char* RetVal = "unknown error";
        switch (Code) {
        case -1:
            RetVal = "input inconsistent"; break;
        case -2:
            RetVal = "not enough memory"; break;
        case -3:
            RetVal = "reordering problem"; break;
        case -4:
            RetVal = "zero pivot, numerical factorization or iterative refinement problem"; break;
        case -5:
            RetVal = "unclassified (internal) error"; break;
        case -6:
            RetVal = "preordering failed (matrix types 11, 13 only)"; break;
        case -7:
            RetVal = "diagonal matrix problem"; break;
        case -8:
            RetVal = "32-bit integer overflow problem"; break;
        case -9:
            RetVal = "not enough memory for OOC"; break;
        case -10:
            RetVal = "problems with opening OOC temporary files"; break;
        case -11:
            RetVal = "read/write problems with the OOC data file"; break;
        case -22:
            RetVal = "not enough memory for direct method"; break;
        default:
            break;
        }
        return RetVal;
    }
    void LinearSolverPARDISO_32_d::initParams() {
        // Setup Pardiso control parameters
        for (int i = 0; i < 64; i++) {
            _iparm[i] = 0;
        }
            
        _iparm[0] = 1;  /* No solver default */
        _iparm[1] = 3; /* Fill-in reordering from METIS */ // !!! = 0
        /* Numbers of processors, value of OMP_NUM_THREADS */
        _iparm[2] = 1;
        _iparm[3] = 0; /* No iterative-direct algorithm */
        _iparm[4] = 0; /* No user fill-in reducing permutation */
        _iparm[5] = 0; /* If =0 then write solution only into x. If =1 then the RightHandSide-array will replaced by the solution*/
        _iparm[6] = 0; /* Not in use */
        _iparm[7] = 2; /* Max numbers of iterative refinement steps */
        _iparm[8] = 0; /* Not in use */
        _iparm[11] = 0; /* Not in use */
        if (1) { // sym by default
            _iparm[9] = 8;
            _iparm[10] = 0; /* Disable scaling. Default for symmetric indefinite matrices. */
            _iparm[12] = 0; /* Maximum weighted matching algorithm is switched-off (default for symmetric). Try iparm[12] = 1 in case of inappropriate accuracy */
        }
        else {
            _iparm[9] = 13;
            _iparm[10] = 1; /* Use nonsymmetric permutation and scaling MPS */
            _iparm[12] = 1; /*1!!! Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
        }
        //_iparm[12] = 1; /*1!!! Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
        _iparm[13] = 0; /* Output: Number of perturbed pivots */
        _iparm[14] = 0; /* Not in use */
        _iparm[15] = 0; /* Not in use */
        _iparm[16] = 0; /* Not in use */
        _iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
        _iparm[18] = -1; /* Output: Mflops for LU factorization */
        _iparm[19] = 0; /* Output: Numbers of CG Iterations */
        _iparm[26] = 1; /* Matrix Checker */
        if (1) // double by default
            _iparm[27] = 0;
        else
            _iparm[27] = 1;
       
        _iparm[59] = 0;

        _iparm[34] = 1; // zero-indexing

        for (int i = 0; i < 64; i++) {
            _pt[i] = 0;
        }
        
    }
    int LinearSolverPARDISO_32_d::Solve(const SPARSE::Case<int, double> & rhs) {
        initParams();
        MKL_INT n = rhs.A.size();
        MKL_INT nnz = rhs.A.nnz();

        const MKL_INT *h_RowsA = rhs.A.getRows();
        const MKL_INT *h_ColsA = rhs.A.getCols();
        const double *h_ValsA = rhs.A.getVals();

        double *h_b = rhs.b.getVals();
        double *h_x = rhs.x.getVals();
        MKL_INT nrhs = rhs.b.nrhs();
        double ddum = 0.0;
        MKL_INT maxfct = 1;
        MKL_INT msglvl = 0;
        MKL_INT mnum = 1;
        MKL_INT mtype = 11;
        MKL_INT idum = 0;
        MKL_INT phase = 11;
        MKL_INT error = 0;

        //phase11
        
        pardiso(&_pt[0], (MKL_INT*)&maxfct, (MKL_INT*)&mnum, (MKL_INT*)&mtype, (MKL_INT*)&phase, (MKL_INT*)&n,
            (void*)h_ValsA, (MKL_INT*)h_RowsA, (MKL_INT*)h_ColsA,
            (MKL_INT*)&idum, (MKL_INT*)&nrhs, (MKL_INT*)&_iparm[0], (MKL_INT*)&msglvl, &ddum, &ddum, (MKL_INT*)&error);
       
        
        //phase22
        phase = 22;
        pardiso((MKL_INT*)&_pt[0], (MKL_INT*)&maxfct, (MKL_INT*)&mnum, (MKL_INT*)&mtype, (MKL_INT*)&phase, (MKL_INT*)&n,
            (void*)h_ValsA, (MKL_INT*)h_RowsA, (MKL_INT*)h_ColsA,
            (MKL_INT*)&idum, (MKL_INT*)&nrhs, (MKL_INT*)&_iparm[0], (MKL_INT*)&msglvl, &ddum, &ddum, (MKL_INT*)&error);
        
        //phase33
        phase = 33;
        pardiso((MKL_INT*)&_pt[0], (MKL_INT*)&maxfct, (MKL_INT*)&mnum, (MKL_INT*)&mtype, (MKL_INT*)&phase, (MKL_INT*)&n,
            (void*)h_ValsA, (MKL_INT*)h_RowsA, (MKL_INT*)h_ColsA,
            (MKL_INT*)&idum, (MKL_INT*)&nrhs, (MKL_INT*)&_iparm[0], (MKL_INT*)&msglvl, (void*)h_b, (void*)h_x, (MKL_INT*)&error);       

        //phase -1
        phase = -1;
        pardiso(&_pt[0], (MKL_INT*)&maxfct, (MKL_INT*)&mnum, (MKL_INT*)&mtype, (MKL_INT*)&phase, (MKL_INT*)&n,
            (void*)h_ValsA, (MKL_INT*)h_RowsA, (MKL_INT*)h_ColsA,
            (MKL_INT*)&idum, (MKL_INT*)&nrhs, (MKL_INT*)&_iparm[0], (MKL_INT*)&msglvl, (void*)h_b, (void*)h_x, (MKL_INT*)&error);        
        mkl_free_buffers();
        return 0;
    }


    void LinearSolverPARDISO_64_d::initParams() {
        // Setup Pardiso control parameters
        for (int i = 0; i < 64; i++) {
            _iparm[i] = 0;
        }

        _iparm[0] = 1;  /* No solver default */
        _iparm[1] = 3; /* Fill-in reordering from METIS */ // !!! = 0
        /* Numbers of processors, value of OMP_NUM_THREADS */
        _iparm[2] = 1;
        _iparm[3] = 0; /* No iterative-direct algorithm */
        _iparm[4] = 0; /* No user fill-in reducing permutation */
        _iparm[5] = 0; /* If =0 then write solution only into x. If =1 then the RightHandSide-array will replaced by the solution*/
        _iparm[6] = 0; /* Not in use */
        _iparm[7] = 2; /* Max numbers of iterative refinement steps */
        _iparm[8] = 0; /* Not in use */
        _iparm[11] = 0; /* Not in use */
        if (1) { // sym by default
            _iparm[9] = 8;
            _iparm[10] = 0; /* Disable scaling. Default for symmetric indefinite matrices. */
            _iparm[12] = 0; /* Maximum weighted matching algorithm is switched-off (default for symmetric). Try iparm[12] = 1 in case of inappropriate accuracy */
        }
        else {
            _iparm[9] = 13;
            _iparm[10] = 1; /* Use nonsymmetric permutation and scaling MPS */
            _iparm[12] = 1; /*1!!! Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
        }
        //_iparm[12] = 1; /*1!!! Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
        _iparm[13] = 0; /* Output: Number of perturbed pivots */
        _iparm[14] = 0; /* Not in use */
        _iparm[15] = 0; /* Not in use */
        _iparm[16] = 0; /* Not in use */
        _iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
        _iparm[18] = -1; /* Output: Mflops for LU factorization */
        _iparm[19] = 0; /* Output: Numbers of CG Iterations */
        _iparm[26] = 1; /* Matrix Checker */
        if (1) // double by default
            _iparm[27] = 0;
        else
            _iparm[27] = 1;

        _iparm[59] = 0;

        _iparm[34] = 1; // zero-indexing

        for (int i = 0; i < 64; i++) {
            _pt[i] = 0;
        }

    }
    int LinearSolverPARDISO_64_d::Solve(const SPARSE::Case<long long int, double>& rhs) {
        initParams();
        long long int n = rhs.A.size();
        long long int nnz = rhs.A.nnz();

        const long long int* h_RowsA = rhs.A.getRows();
        const long long int* h_ColsA = rhs.A.getCols();
        const double* h_ValsA = rhs.A.getVals();

        double* h_b = rhs.b.getVals();
        double* h_x = rhs.x.getVals();
        long long int nrhs = rhs.b.nrhs();
        double ddum = 0.0;
        long long int maxfct = 1;
        long long int msglvl = 0;
        long long int mnum = 1;
        long long int mtype = 11;
        long long int idum = 0;
        long long int phase = 11;
        long long int error = 0;

        //phase11

        pardiso_64(&_pt[0], (long long int*)&maxfct, (long long int*)&mnum, (long long int*)&mtype, (long long int*)&phase, (long long int*)&n,
            (void*)h_ValsA, (long long int*)h_RowsA, (long long int*)h_ColsA,
            (long long int*)&idum, (long long int*)&nrhs, (long long int*)&_iparm[0], (long long int*)&msglvl, &ddum, &ddum, (long long int*)&error);


        //phase22
        phase = 22;
        pardiso_64((long long int*)&_pt[0], (long long int*)&maxfct, (long long int*)&mnum, (long long int*)&mtype, (long long int*)&phase, (long long int*)&n,
            (void*)h_ValsA, (long long int*)h_RowsA, (long long int*)h_ColsA,
            (long long int*)&idum, (long long int*)&nrhs, (long long int*)&_iparm[0], (long long int*)&msglvl, &ddum, &ddum, (long long int*)&error);

        //phase33
        phase = 33;
        pardiso_64((long long int*)&_pt[0], (long long int*)&maxfct, (long long int*)&mnum, (long long int*)&mtype, (long long int*)&phase, (long long int*)&n,
            (void*)h_ValsA, (long long int*)h_RowsA, (long long int*)h_ColsA,
            (long long int*)&idum, (long long int*)&nrhs, (long long int*)&_iparm[0], (long long int*)&msglvl, (void*)h_b, (void*)h_x, (long long int*)&error);

        //phase -1
        phase = -1;
        pardiso_64(&_pt[0], (long long int*)&maxfct, (long long int*)&mnum, (long long int*)&mtype, (long long int*)&phase, (long long int*)&n,
            (void*)h_ValsA, (long long int*)h_RowsA, (long long int*)h_ColsA,
            (long long int*)&idum, (long long int*)&nrhs, (long long int*)&_iparm[0], (long long int*)&msglvl, (void*)h_b, (void*)h_x, (long long int*)&error);
        mkl_free_buffers();
        return 0;
    }
}