#ifndef __OBSTACLE_H__
#define __OBSTACLE_H__

#pragma once
#include "Object.h"
#include "PrimalTree.h"

class Obstacle : public Object
{
public:
	PrimalTree			*_priTree;
public:
	vector<REAL3>		h_pivots;
	vector<REAL3>		h_rotations;
public:
	Dvector<REAL3>		d_pivots;
	Dvector<REAL3>		d_rotations;
public:
	Obstacle() {
		init();
	}
	virtual ~Obstacle() {}
public:
	virtual void	init(void);
public:
	void	addObject(Mesh* mesh, REAL mass, REAL3& pivot, REAL3& rotation, bool isSaved = true);
public:
	void	computeExternalForce(REAL invdt2);
	void	update(REAL dt, REAL invdt);
public:
	void	reset(void);
};
#endif