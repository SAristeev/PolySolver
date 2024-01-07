cmake_minimum_required(VERSION 3.17.0 FATAL_ERROR)
project(PolySolver VERSION 0.3.0 LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/CMake")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})	

add_subdirectory("SparseMV")

add_executable(main "main.cpp"
					"PolySolver.cpp" 
					"PolySolver.hpp"
					"Solvers/implementation.cpp"
					"Solvers/implementation.hpp"
					"Solvers/pardiso.cpp"
					"Solvers/pardiso.hpp")

target_link_libraries(main LINK_PRIVATE SparseMV)
target_include_directories(main PUBLIC
$<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(main PUBLIC $<LINK_ONLY:MKL::MKL>)