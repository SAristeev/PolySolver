cmake_minimum_required(VERSION 3.17.0 FATAL_ERROR)

source_group(Kernel FILES kernel.h)
source_group(Kernel FILES kernel.cpp)
source_group(LinearLib FILES LinearSolver/LinearSolver_IMPL.h)
source_group(LinearLib FILES LinearSolver/LinearSolver_IMPL.cpp)

if(POLYSOLVER_USE_MKL)
source_group(LinearLib FILES LinearSolver/LinearSolver_PARDISO.h)
source_group(LinearLib FILES LinearSolver/LinearSolver_PARDISO.cpp)
endif(POLYSOLVER_USE_MKL)

if(POLYSOLVER_USE_AMGX)
source_group(LinearLib FILES LinearSolver/LinearSolver_AMGX.h)
source_group(LinearLib FILES LinearSolver/LinearSolver_AMGX.cpp)
endif(POLYSOLVER_USE_AMGX)

if(POLYSOLVER_USE_AMGCL)
source_group(LinearLib FILES LinearSolver/LinearSolver_AMGCL.h)
source_group(LinearLib FILES LinearSolver/LinearSolver_AMGCL.cpp)
endif(POLYSOLVER_USE_AMGCL)


file(GLOB SRCS kernel.cpp ${SRCS})
file(GLOB HDRS kernel.h   ${HDRS})
file(GLOB SRCS LinearSolver/LinearSolver_IMPL.cpp ${SRCS})
file(GLOB HDRS LinearSolver/LinearSolver_IMPL.h   ${HDRS})

if(POLYSOLVER_USE_MKL)
file(GLOB SRCS LinearSolver/LinearSolver_PARDISO.cpp ${SRCS})
file(GLOB HDRS LinearSolver/LinearSolver_PARDISO.h   ${HDRS})
endif(POLYSOLVER_USE_MKL)

if(POLYSOLVER_USE_AMGX)
file(GLOB SRCS LinearSolver/LinearSolver_AMGX.cpp ${SRCS})
file(GLOB HDRS LinearSolver/LinearSolver_AMGX.h   ${HDRS})
endif(POLYSOLVER_USE_AMGX)

if(POLYSOLVER_USE_AMGCL)
file(GLOB SRCS LinearSolver/LinearSolver_AMGCL.cpp ${SRCS})
file(GLOB HDRS LinearSolver/LinearSolver_AMGCL.h   ${HDRS})
endif(POLYSOLVER_USE_AMGCL)

	

add_library(Solvers ${SRCS} ${HDRS})

if(POLYSOLVER_USE_MKL)
	target_include_directories(Solvers PUBLIC MKL::MKL)
	target_include_directories(Solvers PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
endif(POLYSOLVER_USE_MKL)

if(POLYSOLVER_USE_CUDA)
	target_include_directories(Solvers PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
	target_link_libraries(Solvers LINK_PUBLIC CUDA::cudart CUDA::cusolver CUDA::cusparse)
endif(POLYSOLVER_USE_CUDA)

if(POLYSOLVER_USE_AMGX)
	target_include_directories(Solvers PUBLIC ${AMGX_INCLUDE_PATH})
	target_link_libraries(Solvers LINK_PUBLIC ${AMGX_LIBRARIES_amgx})
	target_link_libraries(Solvers LINK_PUBLIC ${AMGX_LIBRARIES_amgxsh})
endif(POLYSOLVER_USE_AMGX)

if(POLYSOLVER_USE_AMGCL)
	target_include_directories(Solvers PUBLIC "${CMAKE_SOURCE_DIR}/external")
	target_include_directories(Solvers PUBLIC ${Boost_INCLUDE_DIRS})
	target_link_libraries(Solvers LINK_PUBLIC ${Boost_LIBRARIES})
endif(POLYSOLVER_USE_AMGCL)

target_link_libraries(Solvers PRIVATE SparseMV)
target_include_directories(Solvers PRIVATE SparseMV)
set_target_properties(Solvers PROPERTIES FOLDER "PolySolver")