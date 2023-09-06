import Sparse;

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include"../Solvers/kernel.h"
#include<boost/test/included/unit_test.hpp>
//#include<boost/timer/timer.hpp>

BOOST_AUTO_TEST_CASE(AMGX) {
    SPARSE::Case<int, double> case_5x5;
    BOOST_CHECK_NO_THROW(case_5x5.fread("C:/WorkDirectory/Cases/5x5"));
    SOLVER::LinearSolverAMGX_32_d solver;
    solver.Solve(case_5x5);
}

BOOST_AUTO_TEST_CASE(AMGCL) {
    SPARSE::Case<int, double> case_5x5;
    BOOST_CHECK_NO_THROW(case_5x5.fread("C:/WorkDirectory/Cases/5x5"));
    SOLVER::LinearSolverAMGCL_32_d solver;
    solver.Solve(case_5x5);
}

BOOST_AUTO_TEST_CASE(Parsiso) {
    SPARSE::Case<int, double> case_5x5;
    BOOST_CHECK_NO_THROW(case_5x5.fread("C:/WorkDirectory/Cases/5x5"));
    SOLVER::LinearSolverPARDISO_32_d solver;
    solver.Solve(case_5x5);
}

BOOST_AUTO_TEST_CASE(Parsiso64) {
    SPARSE::Case<long long int, double> case_5x5;
    BOOST_CHECK_NO_THROW(case_5x5.fread("C:/WorkDirectory/Cases/5x5"));
    SOLVER::LinearSolverPARDISO_64_d solver;
    solver.Solve(case_5x5);
}

BOOST_AUTO_TEST_SUITE(Collaboration)
//BOOST_AUTO_TEST_CASE(Parsiso_AMGX) {
//    SPARSE::Case<int, double> case_5x5;
//    BOOST_CHECK_NO_THROW(case_5x5.fread("C:/WorkDirectory/Cases/3031_iter"));
//
//    //boost::timer::cpu_timer timer;
//
//    //SOLVER::LinearSolverPARDISO_32_d solver1;
//    SOLVER::LinearSolverAMGX_32_d solver2;
//    //solver1.Solve(case_5x5);
//    solver2.Solve(case_5x5);
//    //std::cout << timer.format() << std::endl;
//}
BOOST_AUTO_TEST_SUITE_END() // Collaboration