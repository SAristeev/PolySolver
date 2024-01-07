#pragma once
#include "../SparseMV/Sparse.hpp"
#include <map>
#include <unordered_map>
#include "mkl.h"

enum class SolverID {
	sPARDISO = 10,
	sAMGCL = 20
};

namespace polysolver {
	std::unordered_map<std::string, SolverID> const table = {
		{"PARDISO",SolverID::sPARDISO},
		{"AMGCL", SolverID::sAMGCL}};
}

// Base solver class 
class LinearSolver {
public:
	virtual int Solve(const SPARSE::SparseMatrix<MKL_INT, double>& A,
		const SPARSE::SparseVector<double>& b,
		SPARSE::SparseVector<double>& x
	) {
		return -1;
	}
};

// templated Base creator class 
template <class Base>
class AbstractSolverCreator
{
public:
	AbstractSolverCreator() {}
	virtual ~AbstractSolverCreator() {}
	virtual std::unique_ptr<Base> create() const = 0;
};


template<class C, class  Base>
class SolverCreator : public AbstractSolverCreator<Base>
{
public:
	SolverCreator() {}
	virtual ~SolverCreator() {}
	std::unique_ptr<Base> create() const { return std::make_unique<C>(); }
};

template<class Base, class IdType>
class ObjectSolverFactory
{
protected:
	typedef AbstractSolverCreator<Base> AbstractSolverFactory;
	typedef std::map<IdType, std::unique_ptr<AbstractSolverFactory>> SolverFactoryMap;
	SolverFactoryMap _factory;
public:

	ObjectSolverFactory() {}
	virtual ~ObjectSolverFactory() {}

	template <class C>
	void add(const IdType& id)
	{
		typename SolverFactoryMap::iterator it = _factory.find(id);
		if (it == _factory.end())
		{
			_factory.emplace(id, new SolverCreator<C, Base>());
		}
	}
	std::unique_ptr<Base> get(const IdType& id)
	{
		typename SolverFactoryMap::iterator it = _factory.find(id);
		return it->second->create();
	}
	IdType getType(const IdType& id)
	{
		typename SolverFactoryMap::iterator it = _factory.find(id);
		return it->first;
	}
};

void InitLinearSolvers(ObjectSolverFactory<LinearSolver, SolverID>& LinearFactory);
