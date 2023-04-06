#include"LinearSolver_PARDISO.h"

namespace SPARSE {

    int LinearSolverPARDISO::SolveRightSide(SparseMatrix& A,
        SparseVector& b,
        SparseVector& x) {

        A.GetInfo(n, nnzA);
        A.GetDataCSR(&h_ValsA, &h_RowsA, &h_ColsA);
        b.GetData(&h_b);
        x.SetOnes(n, 1);
        x.GetData(&h_x);
        //h_x = (double*)malloc(n * sizeof(double));
        int nb = 0, nrhs = 0;
        b.GetInfo(nb, nrhs);
        if (nb != n) {
            return -1;
        }

        
        return 0;
    }
}