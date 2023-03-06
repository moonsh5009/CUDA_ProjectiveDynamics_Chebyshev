#ifndef __CONSTRAINT_H__
#define __CONSTRAINT_H__

#pragma once
#include "Mesh.h"

#define CONSTRAINT_BLOCKSIZE		128u

struct CBSPDSpring {
	uint* _inos;
	REAL* _restLengths;
	REAL* _Bs;
	REAL* _ws;
	REAL* _errors;
	uint	_numSprings;
};

class CBSPDConstraint
{
public:
	Dvector<uint>	_inos;
	Dvector<REAL>	_restLengths;
	Dvector<REAL>	_Bs;
	Dvector<REAL>	_ws;
	Dvector<REAL>	_errors;
public:
	REAL			_rho;
	REAL			_underRelax;
	uint			_numNodes;
	uint			_numSprings;
public:
	CBSPDConstraint() {}
	virtual ~CBSPDConstraint() {}
public:
	void init(REAL rho, REAL underRelax) {
		_inos.clear();
		_restLengths.clear();
		_Bs.clear();
		_ws.clear();
		_errors.clear();
		_rho = rho;
		_underRelax = underRelax;
		_numNodes = 0u;
		_numSprings = 0u;
	}
	inline void extendSprings(void) {
		_inos.extend(_numSprings << 1u);
		_restLengths.extend(_numSprings);
		_Bs.extend(_numNodes);
		_ws.extend(_numSprings);
		_errors.extend(_numNodes * 3u);
	}
	inline CBSPDSpring springs(void) {
		CBSPDSpring s;
		s._inos = _inos._list;
		s._restLengths = _restLengths._list;
		s._Bs = _Bs._list;
		s._ws = _ws._list;
		s._errors = _errors._list;
		s._numSprings = _numSprings;
		return s;
	}
public:
	void addConstratins(const Dvector<uint>& es, const Dvector<REAL>& ns, REAL w);
public:
	void getOmega(uint itr, REAL& omg);
	void jacobiProject0(Dvector<REAL>& n0s, Dvector<REAL>& ms, Dvector<REAL>& newNs, REAL invdt2);
	void jacobiProject1(Dvector<REAL>& ns, Dvector<REAL>& newNs);
	void jacobiProject2(
		Dvector<REAL>& ns, Dvector<REAL>& prevNs,
		Dvector<REAL>& ms, Dvector<REAL>& Bs,
		Dvector<REAL>& newNs, REAL invdt2, REAL omg);
	void jacobiProject2(
		Dvector<REAL>& ns, Dvector<REAL>& prevNs,
		Dvector<REAL>& ms, Dvector<REAL>& Bs,
		Dvector<REAL>& newNs, REAL invdt2, 
		REAL underRelax, REAL omg, REAL& maxError);
public:
	void project(Dvector<REAL>& ns, Dvector<REAL>& ms, REAL invdt2, uint iteration);
};

#endif