
cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)
project(PolySolver VERSION 0.1.0 LANGUAGES CXX C)
set (CMAKE_CXX_STANDARD 20)

option(USE_FLOAT "enable single precision" OFF)
option(USE_64_BIT_INT "enable x64 integer" OFF)

option(USE_GINKGO "find and enable Ginkgo" OFF)
option(USE_AMGX "find and enable AMGX" OFF)
option(USE_PETSc "find and enable PETSc" OFF)
option(USE_PARDISO "find and enable PARDISO" OFF)


if(USE_64_BIT_INT)
	add_definitions(-DUSE_64_BIT_INT)
endif(USE_64_BIT_INT)



find_package(CUDAToolkit)
message("FIND CUDA: " ${CUDAToolkit_BIN_DIR})


if(USE_GINKGO)
	find_package(Ginkgo)
	message("FIND Ginkgo: " ${Ginkgo_DIR})
	set(Ginkgo_INCLUDE_DIR  "${Ginkgo_DIR}/../../../include")
	set(Ginkgo_LIBRARIES_DIR "${Ginkgo_DIR}/../../../lib")
	message("     Ginkgo include dir: " ${Ginkgo_INCLUDE_DIR} )
	message("     Ginkgo library dir: " ${Ginkgo_LIBRARIES_DIR})
endif(USE_GINKGO)

if(USE_AMGX)
	message("FIND AMGX: " $ENV{AMGX_DIR})
endif(USE_AMGX)



add_executable(PolySolver "PolySolver.cpp")


target_include_directories(PolySolver PRIVATE ${CUDAToolkit_INCLUDE_DIRS})


if(USE_AMGX)
	target_include_directories(PolySolver PRIVATE "$ENV{AMGX_DIR}/include")
	target_link_libraries(PolySolver PRIVATE "$ENV{AMGX_DIR}/lib/amgxsh.lib")
endif(USE_AMGX)


add_subdirectory("SparseAPI")
target_link_libraries(PolySolver LINK_PUBLIC SparseAPI)

add_subdirectory("Kernel")
target_link_libraries(PolySolver LINK_PUBLIC Kernel)