cmake_minimum_required(VERSION 3.17.0 FATAL_ERROR)
project(PolySolver VERSION 0.3.0 LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/CMake")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(POLYSOLVER_TEST          "enable Boost tests"          OFF)
option(POLYSOLVER_USE_CUDA      "enable CUDA-powered solvers" OFF)
option(POLYSOLVER_USE_MKL       "enable Intel MKL"            OFF)
option(POLYSOLVER_USE_CUSOLVER  "enable Nvidia cuSOLVER"      OFF)
option(POLYSOLVER_USE_AMGX      "enable Nvidia AMGX"          OFF)
option(POLYSOLVER_USE_AMGCL     "enable AMGCL      "          OFF)

if(NOT POLYSOLVER_USE_CUDA)
	set(POLYSOLVER_USE_AMGX OFF)
	set(POLYSOLVER_USE_cuSOLVER OFF)
endif(NOT POLYSOLVER_USE_CUDA)

if(POLYSOLVER_USE_MKL)
	find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})	
	add_compile_definitions(POLYSOLVER_USE_MKL)
endif(POLYSOLVER_USE_MKL)

if(POLYSOLVER_USE_CUDA)
	find_package(CUDAToolkit REQUIRED)
	add_compile_definitions(POLYSOLVER_USE_CUDA)
endif(POLYSOLVER_USE_CUDA)

if(POLYSOLVER_USE_AMGX)
	find_package(AMGX REQUIRED)
	add_compile_definitions(POLYSOLVER_USE_AMGX)
endif(POLYSOLVER_USE_AMGX)

if(POLYSOLVER_USE_AMGCL)
	add_compile_definitions(POLYSOLVER_USE_AMGCL)
endif(POLYSOLVER_USE_AMGCL)

if(POLYSOLVER_TEST)
	include(PrepareTests)
	find_package(Boost REQUIRED)
endif(POLYSOLVER_TEST)

add_subdirectory("SparseMV")
add_subdirectory("Solvers")

add_executable(main "PolySolver.cpp")
	
if(POLYSOLVER_TEST)
	PrepareTests()
	if(WIN32)
		configure_file(${CMAKE_SOURCE_DIR}/CMake/VS_tests_template
		${CMAKE_CURRENT_BINARY_DIR}/PolyTests.vcxproj.user @ONLY)
	endif(WIN32)
	add_executable(PolyTests "Testing/PolyTests.cpp")
	target_include_directories(PolyTests PRIVATE ${Boost_INCLUDE_DIRS})

	target_link_libraries(PolyTests LINK_PRIVATE ${Boost_LIBRARY_DIR_RELEASE})
	target_link_libraries(PolyTests LINK_PRIVATE Solvers)

	if(POLYSOLVER_USE_MKL)
		target_include_directories(PolyTests PRIVATE ${MKL_INCLUDE_DIR})
	endif(POLYSOLVER_USE_MKL)

	if(POLYSOLVER_USE_CUDA)
		target_include_directories(PolyTests PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
	endif(POLYSOLVER_USE_CUDA)

	if(POLYSOLVER_USE_AMGX)
		target_include_directories(PolyTests PRIVATE ${AMGX_INCLUDE_PATH})
	endif(POLYSOLVER_USE_AMGX)
	set_target_properties(PolyTests PROPERTIES FOLDER "PolySolver")
endif(POLYSOLVER_TEST)