#include<iostream>
#include<cstdlib>
#include"Kernel/Kernel.h"
#include"SparseAPI/SparseAPI.h"


int main(int argc, char** argv)
{
    using namespace KERNEL;
    
    double time = 0;
    ProblemCase Case("C:/WorkDirectory/config.json");
    Case.start();
    
    return 0;
}