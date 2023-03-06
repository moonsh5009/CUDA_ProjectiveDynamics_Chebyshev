#ifndef __CLOTH_H__
#define __CLOTH_H__

#pragma once
#include "Object.h"

class Cloth : public Object
{
public:
	CBSPDConstraint			* _constraints;
	vector<uint>			_fixed;
public:
	REAL3					_degree;
	uint					_maxIter;
public:
	Cloth() {
		init(30u);
	}
	Cloth(uint maxIter) {
		init(maxIter);
	}
	virtual ~Cloth() {}
public:
	virtual void	init(uint maxIter);
public:
	void	addCloth(Mesh* mesh, REAL mass, bool isSaved = true);
	void	fix(void);
	void	rotateFixed(REAL invdt);
public:
	void	computeExternalForce(REAL3& gravity);
	void	update(REAL3& gravity, REAL dt, REAL invdt);
public:
	void	reset(void);
};
#endif