#include<iostream>
#include<cstdlib>
#include"Solvers/kernel.h"




int main(int argc, char** argv)
{
	using namespace KERNEL;
	try {
		std::system("color A");
		ProblemCase<int, double> Case;
		Case.setConfig("C:/WorkDirectory/PolySolver/config.json");
		Case.start();
	}
	catch (const std::exception& ex) {
		std::cerr << ex.what() << std::endl;
	}


	return 0;
}