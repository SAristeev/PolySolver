import Sparse;

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include<boost/test/included/unit_test.hpp>
#include<filesystem>
#include<cmath>
namespace fs = std::filesystem;
namespace SPARSETest {
    
    auto matrix_dir = fs::temp_directory_path().string() + "SparseMV/";
    auto matrix_5x5 = fs::temp_directory_path().string() + "SparseMV/5x5.txt";
    auto b_5 = fs::temp_directory_path().string() + "SparseMV/b_5.vec";
    auto x_5 = fs::temp_directory_path().string() + "SparseMV/x_5.vec";
    auto matrix_IntelExample = fs::temp_directory_path().string() + "SparseMV/IntelExample.txt";
}

BOOST_AUTO_TEST_CASE(Preparation) {
    fs::create_directory(fs::temp_directory_path().string() + "SparseMV");
    std::ofstream fileA5(SPARSETest::matrix_5x5);
    fileA5 << "5 13" << std::endl;
    fileA5 << "0 5 7 9 11 13" << std::endl;
    fileA5 << "0 1 2 3 4 0 1 0 2 0 3 0 4" << std::endl;
    fileA5 << "8 1 1 1 1 1 2 1 2 1 2 1 2" << std::endl;
    fileA5.close();
    
    std::ofstream fileIntelExample(SPARSETest::matrix_IntelExample);
    fileIntelExample << "5 13" << std::endl;
    fileIntelExample << "0 3 5 8 11 13" << std::endl;
    fileIntelExample << "0 1 3 0 1 2 3 4 0 2 3 1 4" << std::endl;
    fileIntelExample << "1 -1 -3 -2 5 4 6 4 -4 2 7 8 -5" << std::endl;
    fileIntelExample.close();

    std::ofstream fileb5(SPARSETest::b_5);
    fileb5 << "5" << std::endl;
    fileb5 << "1 2 2 2 2" << std::endl;
    fileb5.close();

    std::ofstream filex5(SPARSETest::x_5);
    filex5 << "5" << std::endl;
    filex5 << "-5.0e-01, 1.25e+00, 1.25e+00, 1.25e+00, 1.25e+00" << std::endl;
    filex5.close();
    
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_SUITE(Matrix)
BOOST_AUTO_TEST_SUITE(IO)
BOOST_AUTO_TEST_CASE(EmptyRead) {
    SPARSE::SparseMatrix<int, double> A;
    BOOST_CHECK_THROW(A.freadCSR(""), std::exception);
}
BOOST_AUTO_TEST_CASE(EmptyWrite) {
    SPARSE::SparseMatrix<int, double> A;
    BOOST_CHECK_THROW(A.fprintCSR(""), std::exception);
}
BOOST_AUTO_TEST_CASE(ReadWrite) {
    SPARSE::SparseMatrix<int, double> A;
    BOOST_CHECK_NO_THROW(A.freadCSR(SPARSETest::matrix_5x5));
    BOOST_CHECK_NO_THROW(A.fprintCSR(SPARSETest::matrix_dir + "5x5_copy.txt"));
}
BOOST_AUTO_TEST_SUITE_END() // Matrix/IO
BOOST_AUTO_TEST_SUITE(Data)
BOOST_AUTO_TEST_CASE(NormMaxbyRow) {
    SPARSE::SparseMatrix<int, int> A;
    BOOST_CHECK_NO_THROW(A.freadCSR(SPARSETest::matrix_5x5));
    int norm = A.norm();
    BOOST_REQUIRE_EQUAL(norm, 12);
}

BOOST_AUTO_TEST_CASE(SymmTrue) {
    SPARSE::SparseMatrix<int, double> A;
    BOOST_CHECK_NO_THROW(A.freadCSR(SPARSETest::matrix_5x5));
    BOOST_CHECK(A.checkSymmetric());
}
BOOST_AUTO_TEST_CASE(SymmFalse) {
    SPARSE::SparseMatrix<int, double> A;
    BOOST_CHECK_NO_THROW(A.freadCSR(SPARSETest::matrix_IntelExample));
    BOOST_CHECK(!A.checkSymmetric());
}

BOOST_AUTO_TEST_SUITE_END() // Matrix/Data
BOOST_AUTO_TEST_SUITE(Template)
BOOST_AUTO_TEST_CASE(TNotComplie) {
    //SPARSE::SparseMatrix<int, std::vector<double>> A;
}
BOOST_AUTO_TEST_CASE(Tint) {
    SPARSE::SparseMatrix<int, double> A;
    BOOST_CHECK_NO_THROW(A.freadCSR(SPARSETest::matrix_5x5));
    BOOST_CHECK(A.checkSymmetric());
}
BOOST_AUTO_TEST_CASE(Tsize_t) {
    SPARSE::SparseMatrix<size_t, double> A;
    BOOST_CHECK_NO_THROW(A.freadCSR(SPARSETest::matrix_5x5));
    BOOST_CHECK(A.checkSymmetric());
}
BOOST_AUTO_TEST_CASE(Tint64_t) {
    SPARSE::SparseMatrix<int64_t, double> A;
    BOOST_CHECK_NO_THROW(A.freadCSR(SPARSETest::matrix_5x5));
    BOOST_CHECK(A.checkSymmetric());
}
// fast way to UB
BOOST_AUTO_TEST_CASE(Tfloat) {
    SPARSE::SparseMatrix<float, double> A;
    BOOST_CHECK_NO_THROW(A.freadCSR(SPARSETest::matrix_5x5));
    BOOST_CHECK(A.checkSymmetric());
}
BOOST_AUTO_TEST_CASE(Tdouble) {
    SPARSE::SparseMatrix<double, double> A;
    BOOST_CHECK_NO_THROW(A.freadCSR(SPARSETest::matrix_5x5));
    BOOST_CHECK(A.checkSymmetric());
}
BOOST_AUTO_TEST_SUITE_END() // Matrix/Template
BOOST_AUTO_TEST_SUITE_END() // Matrix

BOOST_AUTO_TEST_SUITE(Vector)
BOOST_AUTO_TEST_SUITE(IO)
BOOST_AUTO_TEST_CASE(EmptyRead) {
    SPARSE::SparseVector<double> b;
    BOOST_CHECK_THROW(b.fread(""), std::exception);
}
BOOST_AUTO_TEST_CASE(DummyWrite) {
    SPARSE::SparseVector<double> b;
    BOOST_CHECK_NO_THROW(b.fprint("dummy"));
}
BOOST_AUTO_TEST_CASE(ReadWrite) {
    SPARSE::SparseVector<double> b;
    BOOST_CHECK_NO_THROW(b.fread(SPARSETest::b_5));
    BOOST_CHECK_NO_THROW(b.fprint(SPARSETest::matrix_dir + "b_5_copy.txt"));
}
BOOST_AUTO_TEST_SUITE_END() // Vector/IO
BOOST_AUTO_TEST_SUITE(Date)
BOOST_AUTO_TEST_CASE(AddRandom) {
    SPARSE::SparseVector<double> b;
    BOOST_CHECK_NO_THROW(b.fread(SPARSETest::b_5));
    BOOST_CHECK_NO_THROW(b.addRandom());
}
BOOST_AUTO_TEST_CASE(AddVector) {
    SPARSE::SparseVector<double> b;
    BOOST_CHECK_NO_THROW(b.fread(SPARSETest::b_5));
    std::vector<double> rhs = { 1,2,3,4,5 };
    BOOST_CHECK_NO_THROW(b.addVector(rhs));
}
BOOST_AUTO_TEST_CASE(AddVectorThrow) {
    SPARSE::SparseVector<double> b;
    BOOST_CHECK_NO_THROW(b.fread(SPARSETest::b_5));
    std::vector<double> rhs = { 1,2,3,4 };
    BOOST_CHECK_THROW(b.addVector(rhs), std::exception);
}
BOOST_AUTO_TEST_CASE(subtraction) {
    SPARSE::SparseVector<double> b;
    BOOST_CHECK_NO_THROW(b.fread(SPARSETest::b_5));
    SPARSE::SparseVector<double> b1 = b;
    SPARSE::SparseVector<double> tmp = SPARSE::subtraction(b,b1);
    BOOST_CHECK(tmp.norm() < 1e-15);
}

BOOST_AUTO_TEST_CASE(Subtraction1000times) {
    SPARSE::SparseVector<double> b;
    BOOST_CHECK_NO_THROW(b.fread(SPARSETest::b_5));
    for (int i = 0; i < 9999; ++i) {
        b.addRandom();
    }
    SPARSE::SparseVector<double> b1 = b;
    SPARSE::SparseVector<double> tmp = SPARSE::subtraction(b, b1);
    BOOST_CHECK(tmp.norm() < 1e-15);
}
BOOST_AUTO_TEST_SUITE_END() // Vector/Data
BOOST_AUTO_TEST_SUITE_END() // Vector

BOOST_AUTO_TEST_SUITE(MatrixVector)
BOOST_AUTO_TEST_CASE(OneSpMV) {
    SPARSE::SparseMatrix<int, double> A;
    BOOST_CHECK_NO_THROW(A.freadCSR(SPARSETest::matrix_5x5));
    std::vector<double> x = { -5.0e-01, 1.25e+00, 1.25e+00, 1.25e+00, 1.25e+00 };
    std::vector<double> answ = { 1.0e+00, 2.0e+00, 2.0e+00, 2.0e+00, 2.0e+00 };
    std::vector<double> b = A.multiplication(x);
    double res = 0;
    for (int i = 0; i < 5; ++i) {
        res += abs(b[i] - answ[i]);
    }
    BOOST_CHECK(res < 1e-15);
}
BOOST_AUTO_TEST_CASE(SpMVDiffDims) {
    SPARSE::SparseMatrix<int, double> A;
    BOOST_CHECK_NO_THROW(A.freadCSR(SPARSETest::matrix_5x5));
    std::vector<double> x = { 0,0,0,0 };
    BOOST_CHECK_THROW(A.multiplication(x), std::exception);
}
BOOST_AUTO_TEST_CASE(SpMV) {
    SPARSE::SparseMatrix<int, double> A;
    SPARSE::SparseVector<double> x;
    BOOST_CHECK_NO_THROW(A.freadCSR(SPARSETest::matrix_5x5));
    BOOST_CHECK_NO_THROW(x.fread(SPARSETest::x_5));
    SPARSE::SparseVector<double> b = SPARSE::multiplication(A, x);
}
BOOST_AUTO_TEST_SUITE_END() // MatrixVector


BOOST_AUTO_TEST_CASE(Cleaning) {
    BOOST_CHECK(std::filesystem::remove_all(SPARSETest::matrix_dir));
}