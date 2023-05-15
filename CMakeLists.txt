cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)
project(PolySolver VERSION 0.1.0 LANGUAGES CXX C)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/CMake")
set (CMAKE_CXX_STANDARD 20)


option(USE_GPU "enable GPU" ON)

option(USE_cuSOLVER "find and enable cuSOLVER" OFF)
option(USE_GINKGO "find and enable Ginkgo" OFF)
option(USE_AMGX "find and enable AMGX" OFF)
option(USE_PETSc "find and enable PETSc" OFF)


if(NOT USE_GPU)
	set(USE_AMGX OFF)
	set(USE_cuSOLVER OFF)
endif(NOT USE_GPU)

find_package(MKL REQUIRED)
message("\nFIND MKL VERSION: ${MKL_VERSION}")
message("     MKL_INCLUDE_DIR: ${MKL_INCLUDE_DIR}")
message("     MKL_LIBRARIES: ${MKL_LIBRARIES}")

if(USE_cuSOLVER)
	find_package(CUDAToolkit)
	message("\nFIND CUDA: " ${CUDAToolkit_BIN_DIR})
endif(USE_cuSOLVER)

if(USE_GINKGO)
	find_package(Ginkgo)
	message("\nFIND Ginkgo: " ${Ginkgo_DIR})
	set(Ginkgo_INCLUDE_DIR  "${Ginkgo_DIR}/../../../include")
	set(Ginkgo_LIBRARIES_DIR "${Ginkgo_DIR}/../../../lib")
	message("     Ginkgo include dir: " ${Ginkgo_INCLUDE_DIR} )
	message("     Ginkgo library dir: " ${Ginkgo_LIBRARIES_DIR})
endif(USE_GINKGO)

if(USE_AMGX)
	message("\nFIND AMGX: " $ENV{AMGX_DIR})
endif(USE_AMGX)


add_executable(PolySolver "PolySolver.cpp")

target_include_directories(PolySolver PRIVATE ${MKL_INCLUDE_DIR})

if(USE_cuSOLVER)
	target_include_directories(PolySolver PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif(USE_cuSOLVER)

if(USE_AMGX)
	target_include_directories(PolySolver PRIVATE "$ENV{AMGX_DIR}/include")
endif(USE_AMGX)


add_subdirectory("SparseAPI")
target_link_libraries(PolySolver LINK_PUBLIC SparseAPI)

add_subdirectory("Kernel")
target_link_libraries(PolySolver LINK_PUBLIC Kernel)