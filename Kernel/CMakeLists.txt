cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

set (CMAKE_CXX_STANDARD 20)

file(GLOB Kernel_SRSC_ *.cpp)
file(GLOB Kernel_HDRS_ *.h *.hpp)

set(Kernel_SRSC ${Kernel_SRSC_})
set(Kernel_HDRS ${Kernel_HDRS_})


add_library(Kernel ${Kernel_SRSC} ${Kernel_HDRS})


target_include_directories(Kernel PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(Kernel PRIVATE CUDA::cudart CUDA::cusolver CUDA::cusparse)


if(USE_AMGX)
	target_include_directories(Kernel PRIVATE "$ENV{AMGX_DIR}/include")
	target_link_libraries(Kernel PRIVATE "$ENV{AMGX_DIR}/lib/amgxsh.lib")
	if(MPI_FOUND)
		target_include_directories(Kernel PRIVATE "${MPI_INCLUDE_PATH}")
	endif(MPI_FOUND)
endif(USE_AMGX)