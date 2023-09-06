cmake_minimum_required(VERSION 3.17.0 FATAL_ERROR)

source_group(Module FILES Sparse.ixx)
source_group(Module FILES SparseMatrix.ixx)
source_group(Module FILES SparseVector.ixx)

file(GLOB MDLS Sparse.ixx ${MDLS})
file(GLOB MDLS SparseMatrix.ixx ${MDLS})
file(GLOB MDLS SparseVector.ixx ${MDLS})

add_library(SparseMV ${MDLS}) 
set_target_properties(SparseMV PROPERTIES FOLDER "SparseMV")

if(POLYSOLVER_TEST)
	
	source_group(Testing FILES SparseTests.cpp)
	file(GLOB SRCS SparseTests.cpp ${SRCS})

	add_executable(Testing SparseTests.cpp)
	target_include_directories(Testing PRIVATE ${Boost_INCLUDE_DIRS})
	target_link_libraries(Testing LINK_PRIVATE ${Boost_LIBRARIES})
	target_link_libraries(Testing LINK_PRIVATE SparseMV)

	set_target_properties(Testing PROPERTIES FOLDER "SparseMV")
endif(POLYSOLVER_TEST)