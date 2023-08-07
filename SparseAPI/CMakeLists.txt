cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

set (CMAKE_CXX_STANDARD 20)


file(GLOB SparseAPI_SRCS_ALL SparseAPI.cpp ${SparseAPI_SRCS_ALL})
file(GLOB SparseAPI_HDRS_ALL SparseAPI.h   ${SparseAPI_HDRS_ALL})

file(GLOB SparseAPI_SRCS_ALL LinearSolver/LinearSolver_IMPL.cpp ${SparseAPI_SRCS_ALL})
file(GLOB SparseAPI_HDRS_ALL LinearSolver/LinearSolver_IMPL.h   ${SparseAPI_HDRS_ALL})

source_group(SparseAPI FILES SparseAPI.h)
source_group(SparseAPI FILES SparseAPI.cpp)

source_group(LinearSolver FILES LinearSolver/LinearSolver_IMPL.h)
source_group(LinearSolver FILES LinearSolver/LinearSolver_IMPL.cpp)

if(USE_PARDISO)
	file(GLOB SparseAPI_SRCS_ALL LinearSolver/LinearSolver_PARDISO.cpp ${SparseAPI_SRCS_ALL})
	file(GLOB SparseAPI_HDRS_ALL LinearSolver/LinearSolver_PARDISO.h   ${SparseAPI_HDRS_ALL})

	source_group(LinearSolver FILES LinearSolver/LinearSolver_PARDISO.h)
	source_group(LinearSolver FILES LinearSolver/LinearSolver_PARDISO.cpp)
endif(USE_PARDISO)

if(USE_cuSOLVER)
	file(GLOB SparseAPI_SRCS_ALL LinearSolver/LinearSolver_cuSOLVER.cpp ${SparseAPI_SRCS_ALL})
	file(GLOB SparseAPI_HDRS_ALL LinearSolver/LinearSolver_cuSOLVER.h   ${SparseAPI_HDRS_ALL})

	source_group(LinearSolver FILES LinearSolver/LinearSolver_cuSOLVER.h)
	source_group(LinearSolver FILES LinearSolver/LinearSolver_cuSOLVER.cpp)
endif(USE_cuSOLVER)

if(USE_AMGX)
	file(GLOB SparseAPI_SRCS_ALL LinearSolver/LinearSolver_AMGX.cpp ${SparseAPI_SRCS_ALL})
	file(GLOB SparseAPI_HDRS_ALL LinearSolver/LinearSolver_AMGX.h   ${SparseAPI_HDRS_ALL})

	source_group(LinearSolver FILES LinearSolver/LinearSolver_AMGX.h)
	source_group(LinearSolver FILES LinearSolver/LinearSolver_AMGX.cpp)
endif(USE_AMGX)

if(USE_GINKGO)
	source_group(LinearSolver FILES LinearSolver/LinearSolver_Ginkgo.h)
	source_group(LinearSolver FILES LinearSolver/LinearSolver_Ginkgo.cpp)
endif(USE_GINKGO)


add_library(SparseAPI ${SparseAPI_SRCS_ALL} ${SparseAPI_HDRS_ALL})


target_include_directories(SparseAPI PRIVATE ${MKL_INCLUDE_DIR})
target_link_libraries(SparseAPI PRIVATE ${MKL_LIBRARIES})

if(USE_cuSOLVER)
	add_definitions(-DUSE_cuSOLVER)
	target_include_directories(SparseAPI PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
	target_link_libraries(SparseAPI PRIVATE CUDA::cudart CUDA::cusolver CUDA::cusparse)
elseif(USE_AMGX)
	target_include_directories(SparseAPI PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
	target_link_libraries(SparseAPI PRIVATE CUDA::cudart)
endif(USE_cuSOLVER)

if(USE_AMGX)
	add_definitions(-DUSE_AMGX)
	target_include_directories(SparseAPI PRIVATE "$ENV{AMGX_DIR}/include")
	target_link_libraries(SparseAPI PRIVATE "$ENV{AMGX_DIR}/lib/amgxsh.lib")
	target_link_libraries(SparseAPI PRIVATE "$ENV{AMGX_DIR}/lib/amgxsh.lib")
endif(USE_AMGX)

if(USE_GINKGO)
	add_definitions(-DUSE_GINKGO)
	target_include_directories(SparseAPI PRIVATE ${Ginkgo_INCLUDE_DIR})
	target_link_libraries(SparseAPI PRIVATE "${Ginkgo_LIBRARIES_DIR}/ginkgo.lib" "${Ginkgo_LIBRARIES_DIR}/ginkgo_cuda.lib")
endif(USE_GINKGO)

