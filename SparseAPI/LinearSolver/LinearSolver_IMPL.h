#pragma once
#include<map>
#include"../SparseAPI.h"



namespace SPARSE {
	
	enum class SolverID {
		cuSOLVERSP = 1,
		cuSOLVERRF = 2,
		cuSOLVERRF0 = 3,
		cuSOLVERRF_ALLGPU = 4,
		AMGX = 5,
		ALL = 99
	};
	std::string SolverID2String(SolverID ID);


	class LinearSolver {
	public:
		virtual int SolveRightSide(SparseMatrix &A,
			SparseVector  &b,
			SparseVector  &x) = 0;
		
		int IsReadyToSolve();
		int IsReadyToSolveByMatrix(SparseMatrix A,
			SparseVector b,
			SparseVector x);
		//int AddImplementation();
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



}

//template class SPARSE::ObjectSolverFactory<SPARSE::LinearSolver, SPARSE::SolverID>;
//template SPARSE::LinearSolver* SPARSE::ObjectSolverFactory<SPARSE::LinearSolver, SPARSE::SolverID>::get<SPARSE::LinearSolver>(SPARSE::SolverID);