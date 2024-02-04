cmake_minimum_required(VERSION 3.17.0 FATAL_ERROR)
project(PolySolver VERSION 0.3.0 LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/CMake")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
option(POLYSOLVER_USE_CUDA OFF)


set(SRCS "main.cpp"
	"PolySolver.cpp" 
	"PolySolver.hpp"
	"Solvers/implementation.cpp"
	"Solvers/implementation.hpp"
	"Solvers/pardiso.cpp"
	"Solvers/pardiso.hpp"
	"Solvers/amgcl.cpp"
	"Solvers/amgcl.hpp"
	"Solvers/mkl_cg.cpp"
	"Solvers/mkl_cg.hpp"
	"Solvers/mkl_pcg.cpp"
	"Solvers/mkl_pcg.hpp")


if(POLYSOLVER_USE_CUDA)
	find_package(CUDAToolkit REQUIRED)
	set(SRCS ${SRCS}
	"Solvers/cuda_helper.hpp"
	"Solvers/cuda_pcg.cpp"
	"Solvers/cuda_pcg.hpp"
	"Solvers/cuda_cg.cpp"
	"Solvers/cuda_cg.hpp" 
	)
endif(POLYSOLVER_USE_CUDA)

find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})	

add_executable(main ${SRCS})
					
target_include_directories(main PUBLIC "external")
add_compile_definitions("AMGCL_NO_BOOST")

target_include_directories(main PUBLIC
$<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(main PUBLIC $<LINK_ONLY:MKL::MKL>)

if(POLYSOLVER_USE_CUDA)
	add_compile_definitions("POLYSOLVER_USE_CUDA")
	target_include_directories(main PUBLIC ${CUDAToolkit_INCLUDE_DIRS}) 
	target_link_libraries(main PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::cusparse)
endif(POLYSOLVER_USE_CUDA)