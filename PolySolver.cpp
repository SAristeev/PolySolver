#include<iostream>
#include<cstdlib>
#include"Kernel/Kernel.h"
#include"SparseAPI/SparseAPI.h"


int main(int argc, char** argv)
{
    using namespace KERNEL;
    std::string CaseName("5x5");
    
    //ProblemCase Case(CaseName, SPARSE::SolverID::AMGX);
    double time = 0;
    ProblemCase Case("C:/WorkDirectory/PolySolver_build/Debug/config.json");
    std::cout << "Solve time: " << time << std::endl;
    
    /*double absnorm1, absnorm2, absnorminf;
    double relnorm1, relnorm2, relnorminf;
    Case.Check(absnorm1, absnorm2, absnorminf, relnorm1, relnorm2, relnorminf);
    std::cout << "Absolute" << std::endl;
    std::cout << "L1 = " << absnorm1 << " L2 = " << absnorm2 << " Linf = " << absnorminf << std::endl;
    std::cout << "Resudial" << std::endl;
    std::cout << "L1 = " << relnorm1 << " L2 = " << relnorm2 << " Linf = " << relnorminf << std::endl;
    Case.print();*/
    return 0;
}