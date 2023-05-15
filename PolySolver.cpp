#include<iostream>
#include<cstdlib>
#include"Kernel/Kernel.h"
#include"SparseAPI/SparseAPI.h"



int main(int argc, char** argv)
{
    using namespace KERNEL;
	try{
		std::system("color A");
		ProblemCase Case("C:/WorkDirectory/PolySolver/config.json");
		Case.start();
	}
	catch (const std::exception& ex){
		std::cerr << ex.what() << std::endl;
	}
    
    
    return 0;
}