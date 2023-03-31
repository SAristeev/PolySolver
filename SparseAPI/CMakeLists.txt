cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

set (CMAKE_CXX_STANDARD 20)

file(GLOB SparseAPI_HDRS_ALL
	*.h
	LinearSolver/*.h
	EigenSolver/*.h
	)

file(GLOB SparseAPI_SRCS_ALL
	*.cpp
	LinearSolver/*.cpp
	EigenSolver/*.cpp
	)

source_group(LinearSolver FILES LinearSolver/LinearSolver_IMPL.h)
source_group(LinearSolver FILES LinearSolver/LinearSolver_IMPL.cpp)

source_group(LinearSolver FILES LinearSolver/LinearSolver_AMGX.h)
source_group(LinearSolver FILES LinearSolver/LinearSolver_AMGX.cpp)

source_group(LinearSolver FILES LinearSolver/LinearSolver_Ginkgo.h)
source_group(LinearSolver FILES LinearSolver/LinearSolver_Ginkgo.cpp)

source_group(LinearSolver FILES LinearSolver/LinearSolver_cuSOLVER.h)
source_group(LinearSolver FILES LinearSolver/LinearSolver_cuSOLVER.cpp)

add_library(SparseAPI ${SparseAPI_SRCS_ALL} ${SparseAPI_HDRS_ALL})


target_include_directories(SparseAPI PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(SparseAPI PRIVATE CUDA::cudart CUDA::cusolver CUDA::cusparse)



if(USE_GINKGO)
	target_include_directories(SparseAPI PRIVATE ${Ginkgo_INCLUDE_DIR})
	target_link_libraries(SparseAPI PRIVATE "${Ginkgo_LIBRARIES_DIR}/ginkgo.lib" "${Ginkgo_LIBRARIES_DIR}/ginkgo_cuda.lib")
endif(USE_GINKGO)

if(USE_AMGX)	
	target_include_directories(SparseAPI PRIVATE "$ENV{AMGX_DIR}/include")
	message("FIND AMGX: $ENV{AMGX_DIR}/include")
	target_link_libraries(SparseAPI PRIVATE "$ENV{AMGX_DIR}/lib/amgxsh.lib")
	target_link_libraries(SparseAPI PRIVATE "$ENV{AMGX_DIR}/lib/amgxsh.lib")
endif(USE_AMGX)