cmake_minimum_required(VERSION 3.17.0 FATAL_ERROR)

set (CMAKE_CXX_STANDARD 20)

file(GLOB Solvers_SRCS_ALL kernel.cpp ${Solvers_SRCS_ALL})
file(GLOB Solvers_HDRS_ALL kernel.h   ${Solvers_HDRS_ALL})

source_group(Kernel FILES kernel.h)
source_group(Kernel FILES kernel.cpp)


file(GLOB Solvers_HDRS_ALL sparse.h   ${Solvers_HDRS_ALL})
file(GLOB Solvers_SRCS_ALL sparse_vector.cpp ${Solvers_SRCS_ALL})
file(GLOB Solvers_SRCS_ALL sparse_matrix.cpp ${Solvers_SRCS_ALL})

source_group(Sparse FILES sparse.h)
source_group(Sparse FILES sparse_vector.cpp)
source_group(Sparse FILES sparse_matrix.cpp)



file(GLOB Solvers_SRCS_ALL LinearSolver/LinearSolver_IMPL.cpp ${Solvers_SRCS_ALL})
file(GLOB Solvers_HDRS_ALL LinearSolver/LinearSolver_IMPL.h   ${Solvers_HDRS_ALL})

source_group(LinearLib FILES LinearSolver/LinearSolver_IMPL.h)
source_group(LinearLib FILES LinearSolver/LinearSolver_IMPL.cpp)
if(USE_PARDISO)
	file(GLOB Solvers_SRCS_ALL LinearSolver/LinearSolver_PARDISO.cpp ${Solvers_SRCS_ALL})
	file(GLOB Solvers_HDRS_ALL LinearSolver/LinearSolver_PARDISO.h   ${Solvers_HDRS_ALL})

	source_group(LinearLib FILES LinearSolver/LinearSolver_PARDISO.h)
	source_group(LinearLib FILES LinearSolver/LinearSolver_PARDISO.cpp)
endif(USE_PARDISO)

if(USE_cuSOLVER)
	file(GLOB Solvers_SRCS_ALL LinearSolver/LinearSolver_cuSOLVER.cpp ${Solvers_SRCS_ALL})
	file(GLOB Solvers_HDRS_ALL LinearSolver/LinearSolver_cuSOLVER.h   ${Solvers_HDRS_ALL})

	source_group(LinearLib FILES LinearSolver/LinearSolver_cuSOLVER.h)
	source_group(LinearLib FILES LinearSolver/LinearSolver_cuSOLVER.cpp)
endif(USE_cuSOLVER)

if(USE_AMGX)
	file(GLOB Solvers_SRCS_ALL LinearSolver/LinearSolver_AMGX.cpp ${Solvers_SRCS_ALL})
	file(GLOB Solvers_HDRS_ALL LinearSolver/LinearSolver_AMGX.h   ${Solvers_HDRS_ALL})

	source_group(LinearLib FILES LinearSolver/LinearSolver_AMGX.h)
	source_group(LinearLib FILES LinearSolver/LinearSolver_AMGX.cpp)
endif(USE_AMGX)


add_library(Solvers ${Solvers_SRCS_ALL} ${Solvers_HDRS_ALL})

target_include_directories(Solvers PRIVATE ${MKL_INCLUDE_DIR})
target_link_libraries(Solvers PRIVATE ${MKL_LIBRARIES})

if(USE_cuSOLVER)
	add_definitions(-DUSE_cuSOLVER)
	target_include_directories(Solvers PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
	target_link_libraries(Solvers PRIVATE CUDA::cudart CUDA::cusolver CUDA::cusparse)
elseif(USE_AMGX)
	target_include_directories(Solvers PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
	target_link_libraries(Solvers PRIVATE CUDA::cudart)
endif(USE_cuSOLVER)

if(USE_AMGX)
	add_definitions(-DUSE_AMGX)
	target_include_directories(Solvers PRIVATE "$ENV{AMGX_DIR}/include")
	target_link_libraries(Solvers PRIVATE "$ENV{AMGX_DIR}/lib/amgxsh.lib")
	target_link_libraries(Solvers PRIVATE "$ENV{AMGX_DIR}/lib/amgxsh.lib")
endif(USE_AMGX)