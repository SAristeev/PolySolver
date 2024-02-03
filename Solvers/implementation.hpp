#pragma once
#include <map>
#include <unordered_map>
#include <memory>
#include <string>
#include "mkl.h"

enum class SolverID {
	ePARDISO = 10,
	eAMGCL = 20,
	eMY_CG = 30
};

namespace polysolver {
	std::unordered_map<std::string, SolverID> const table = {
		{"PARDISO",SolverID::ePARDISO},
		{"AMGCL", SolverID::eAMGCL},
		{"MY_CG", SolverID::eMY_CG} };
}

// Base solver clas
// s 
class LinearSolver {
public:
	virtual int Solve(const std::vector<double>& vals,
		const std::vector<MKL_INT>& cols,
		const std::vector<MKL_INT>& rows,
		const std::vector<double>& b,
		std::vector<double>& x
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
