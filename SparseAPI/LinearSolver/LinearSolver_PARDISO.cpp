#include "LinearSolver_PARDISO.h"
#include <iostream>

namespace SPARSE {

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
     

    void LinearSolverPARDISO::initParams() {
        for (int i = 0; i < 64; i++) {
            _pt[i] = 0;
        }
        for (int i = 0; i < 64; i++) {
            _iparm[i] = 0;
        }
        _iparm[0] = 1;  /* No solver default */
        _iparm[1] = 0; /* Fill-in reordering from METIS */
        /* Numbers of processors, value of OMP_NUM_THREADS */
        _iparm[2] = 1;
        _iparm[3] = 0; /* No iterative-direct algorithm */
        _iparm[4] = 0; /* No user fill-in reducing permutation */
        _iparm[5] = 0; /* If =0 then write solution only into x. If =1 then the RightHandSide-array will replaced by the solution*/
        _iparm[6] = 0; /* Not in use */
        _iparm[7] = 2; /* Max numbers of iterative refinement steps */
        _iparm[8] = 0; /* Not in use */
        _iparm[11] = 0; /* Not in use */
        if (0) {  /* Symmetric matrixes don't support already */
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
        _iparm[27] = 0; /* Double by default */
        _iparm[59] = 0; /* OOC disabled by defualt */

        _iparm[34] = 1; /*Zero-indexing by defalut */
    }

    int LinearSolverPARDISO::SolveRightSide(SparseMatrix& A,
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

        
        
        double ddum = 0.0;
        MKL_INT maxfct = 1;
        MKL_INT msglvl = 1;
        MKL_INT mnum = 1;
        MKL_INT mtype = 11;
        MKL_INT idum = 0;
        MKL_INT phase = 11;
        MKL_INT error = 0;
        
        //phase11

        initParams();

        pardiso(&_pt[0], (MKL_INT*)& maxfct, (MKL_INT*)&mnum, (MKL_INT*)&mtype, (MKL_INT*)&phase, (MKL_INT*)&n, 
            h_ValsA, (MKL_INT*)h_RowsA, (MKL_INT*)h_ColsA,
            (MKL_INT*)&idum, (MKL_INT*)&nrhs, (MKL_INT*)&_iparm[0], (MKL_INT*)&msglvl, &ddum, &ddum, (MKL_INT*)&error);
        
        //phase22
        phase = 22;
        pardiso(&_pt[0], (MKL_INT*)&maxfct, (MKL_INT*)&mnum, (MKL_INT*)&mtype, (MKL_INT*)&phase, (MKL_INT*)&n,
            h_ValsA, (MKL_INT*)h_RowsA, (MKL_INT*)h_ColsA,
            (MKL_INT*)&idum, (MKL_INT*)&nrhs, (MKL_INT*)&_iparm[0], (MKL_INT*)&msglvl, &ddum, &ddum, (MKL_INT*)&error);

        //phase33
        phase = 33;
        pardiso(&_pt[0], (MKL_INT*)&maxfct, (MKL_INT*)&mnum, (MKL_INT*)&mtype, (MKL_INT*)&phase, (MKL_INT*)&n,
            h_ValsA, (MKL_INT*)h_RowsA, (MKL_INT*)h_ColsA,
            (MKL_INT*)&idum, (MKL_INT*)&nrhs, (MKL_INT*)&_iparm[0], (MKL_INT*)&msglvl, h_b, h_x, (MKL_INT*)&error);

        return 0;
    }
}