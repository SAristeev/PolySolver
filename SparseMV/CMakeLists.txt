cmake_minimum_required(VERSION 3.17.0 FATAL_ERROR)

add_library(SparseMV SparseMatrix.cpp
					 SparseMatrix.hpp
					 SparseVector.cpp
					 SparseVector.hpp
					 Sparse.hpp)
			
set_target_properties(SparseMV PROPERTIES FOLDER "SparseMV")