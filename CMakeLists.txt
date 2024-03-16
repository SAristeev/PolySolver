cmake_minimum_required(VERSION 3.17.0 FATAL_ERROR)
project(PolySolver VERSION 0.3.0 LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/CMake")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
option(POLYSOLVER_USE_CUDA OFF)
option(POLYSOLVER_USE_HYPRE OFF)


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
	
if(POLYSOLVER_USE_HYPRE)
	find_package(HYPRE REQUIRED)
	set(SRCS ${SRCS}
	"Solvers/hypre.hpp"
	"Solvers/hypre.cpp"
	)
endif(POLYSOLVER_USE_HYPRE)

if(POLYSOLVER_USE_CUDA)
	find_package(CUDAToolkit REQUIRED)
	find_package(AMGX REQUIRED)
	set(SRCS ${SRCS}
	"Solvers/cuda_helper.hpp"
	"Solvers/cuda_pcg.cpp"
	"Solvers/cuda_pcg.hpp"
	"Solvers/cuda_cg.cpp"
	"Solvers/cuda_cg.hpp" 
	"Solvers/amgx.cpp"
	"Solvers/amgx.hpp"
	)
endif(POLYSOLVER_USE_CUDA)

find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})	
find_package(MPI REQUIRED)

add_executable(main ${SRCS})
					
target_include_directories(main PUBLIC "external")
add_compile_definitions("AMGCL_NO_BOOST")

target_include_directories(main PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(main PUBLIC $<LINK_ONLY:MKL::MKL>)

target_link_libraries(main PUBLIC MPI::MPI_C MPI::MPI_CXX)
target_include_directories(main PUBLIC ${MPI_C_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_DIRS})

if(POLYSOLVER_USE_HYPRE)
	target_include_directories(main PRIVATE "$ENV{HYPRE_ROOT}/include") 
	target_link_libraries(main PRIVATE HYPRE::HYPRE)
endif(POLYSOLVER_USE_HYPRE)

if(POLYSOLVER_USE_CUDA)
	add_compile_definitions("POLYSOLVER_USE_CUDA")
	target_include_directories(main PRIVATE ${CUDAToolkit_INCLUDE_DIRS}) 
	target_link_libraries(main PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::cusparse)
	
	target_include_directories(main PRIVATE ${AMGX_INCLUDE_PATH}) 
	target_link_libraries(main PRIVATE ${AMGX_LIBRARIES_amgx} ${AMGX_LIBRARIES_amgxsh})
endif(POLYSOLVER_USE_CUDA)