#include<iostream>
#include<cstdlib>
#include"Kernel/Kernel.h"
#include"SparseAPI/SparseAPI.h"



int main(int argc, char** argv)
{
    using namespace KERNEL;  
    ProblemCase Case("C:/WorkDirectory/config.json");
    Case.start();
    
    return 0;
}