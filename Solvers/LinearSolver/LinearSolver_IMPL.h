#pragma once
import Sparse;
#include<map>
#include<iostream>
#include"../json.hpp"



namespace SPARSE {

	using json = nlohmann::json;

	enum class SolverID {
		cuSOLVERSP = 1,
		cuSOLVERRF = 2,
		cuSOLVERRF0 = 3,
		cuSOLVERRF_ALLGPU = 4,
		AMGX = 5,
		PARDISO = 6,
		ALL = 99
	};

 	class LinearSolver {
	protected:
		std::string solverName;
		int curConfig;
		int	   	n;
		int	   	nnzA;
		int	   	nrhs;
		const int*    h_RowsA = nullptr; // CPU <int>    n+1
		const int*    h_ColsA = nullptr; // CPU <int>    nnzA
		const double* h_ValsA = nullptr; // CPU <double> nnzA 
		const double* h_x     = nullptr; // CPU <double> n
		const double* h_b     = nullptr; // CPU <double> n
	public:
		LinearSolver(std::string SN) { solverName = SN; }
		std::string getName() { return solverName; }
		void AddConfigToName(std::string configName) { solverName = solverName + ": " + configName; }
		void SetCurConfig(int cur) { curConfig = cur; }
		virtual int SetSettingsFromJSON(json settings) = 0;
		
		virtual int SolveRightSide(SparseMatrix<int,double> &A,
			SparseVector<double>  &b,
			SparseVector<double>&x) = 0;
		
	
		int IsReadyToSolve();
		virtual int PrepareToSolveByMatrix(SparseMatrix<int, double> A,
			SparseVector<double> b,
			SparseVector<double> x);
	};

	template <class Base>
	class AbstractSolverCreator
	{
	public:
		AbstractSolverCreator() {}
		virtual ~AbstractSolverCreator() {}
		virtual Base* create() const = 0;
	};

	
	template<class C, class  Base>
	class SolverCreator : public AbstractSolverCreator<Base>
	{
	public:
		SolverCreator() {}
		virtual ~SolverCreator() {}
		Base * create() const { return new C(); }
	};
	
	template<class Base, class IdType>
	class ObjectSolverFactory 
	{
	protected:
		typedef AbstractSolverCreator<Base> AbstractSolverFactory;
		typedef std::map<IdType, AbstractSolverFactory*> SolverFactoryMap;
		SolverFactoryMap _factory;
	public:
			
		ObjectSolverFactory() {}
		virtual ~ObjectSolverFactory() {}

		template <class C>
		void add(const IdType& id) 
		{
			registerSolver(id, new SolverCreator<C, Base>());
		}
		Base * get(const IdType& id) 
		{
			typename SolverFactoryMap::iterator it = _factory.find(id);
			return it->second->create();
		}
		IdType getType(const IdType& id) 
		{
			typename SolverFactoryMap::iterator it = _factory.find(id);
			return it->first;
		}
	protected:
		void registerSolver(const IdType& id, AbstractSolverFactory* p) 
		{
			typename SolverFactoryMap::iterator it = _factory.find(id);
			if (it == _factory.end()) 
			{
				_factory[id] = p;
			}
			else 
			{
				delete p;
			}
		}
	};


	void AddLinearImplementation(std::map<LinearSolver*, SolverID>& LinearSolvers, ObjectSolverFactory<LinearSolver, SolverID>& LinearFactory, std::string solver);


}

namespace SOLVER {

	class LinearSolver {
	protected:
		std::string solverName;
	public:
		LinearSolver(std::string SN) { solverName = SN; }
		std::string getName() { return solverName; }
		//virtual int Solve(const SPARSE::Case<int, double> & rhs) = 0;
	};

	template <class Base>
	class AbstractSolverCreator
	{
	public:
		AbstractSolverCreator() {}
		virtual ~AbstractSolverCreator() {}
		virtual Base* create() const = 0;
	};


	template<class C, class  Base>
	class SolverCreator : public AbstractSolverCreator<Base>
	{
	public:
		SolverCreator() {}
		virtual ~SolverCreator() {}
		Base* create() const { return new C(); }
	};

	template<class Base, class IdType>
	class ObjectSolverFactory
	{
	protected:
		typedef AbstractSolverCreator<Base> AbstractSolverFactory;
		typedef std::map<IdType, AbstractSolverFactory*> SolverFactoryMap;
		SolverFactoryMap _factory;
	public:

		ObjectSolverFactory() {}
		virtual ~ObjectSolverFactory() {}

		template <class C>
		void add(const IdType& id)
		{
			registerSolver(id, new SolverCreator<C, Base>());
		}
		Base* get(const IdType& id)
		{
			typename SolverFactoryMap::iterator it = _factory.find(id);
			return it->second->create();
		}
		IdType getType(const IdType& id)
		{
			typename SolverFactoryMap::iterator it = _factory.find(id);
			return it->first;
		}
	protected:
		void registerSolver(const IdType& id, AbstractSolverFactory* p)
		{
			typename SolverFactoryMap::iterator it = _factory.find(id);
			if (it == _factory.end())
			{
				_factory[id] = p;
			}
			else
			{
				delete p;
			}
		}
	};
}