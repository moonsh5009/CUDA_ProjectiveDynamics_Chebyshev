#ifndef __OBJECT_H__
#define __OBJECT_H__

#pragma once
#include "CollisionSolver.h"

class Object
{
public:
	BVH						*_bvh;
	RTriangle				*_RTri;
public:
	Dvector<uint>			d_fs;
	Dvector<REAL>			d_ns;
	Dvector<REAL>			d_n0s;
	Dvector<REAL>			d_vs;
	Dvector<REAL>			d_ms;
	Dvector<REAL>			d_invMs;
	DPrefixArray<uint>		d_ses;
	DPrefixArray<uint>		d_bes;
	DPrefixArray<uint>		d_nbFs;
	DPrefixArray<uint>		d_nbNs;
	Dvector<REAL>			d_fNorms;
	Dvector<REAL>			d_nNorms;
	Dvector<uint>			d_nodePhases;
public:
	vector<uint>			h_fs;
	vector<REAL>			h_ns;
	vector<REAL>			h_ms;
	vector<REAL>			h_invMs;
	PrefixArray<uint>		h_ses;
	PrefixArray<uint>		h_bes;
	PrefixArray<uint>		h_nbFs;
	PrefixArray<uint>		h_nbNs;
	vector<REAL>			h_fNorms;
	vector<REAL>			h_nNorms;
	vector<uint>			h_nodePhases;
public:
	vector<uint>			h_fs0;
	vector<REAL>			h_ns0;
	vector<REAL>			h_ms0;
	vector<REAL>			h_invMs0;
	PrefixArray<uint>		h_ses0;
	PrefixArray<uint>		h_bes0;
	PrefixArray<uint>		h_nbFs0;
	PrefixArray<uint>		h_nbNs0;
	vector<uint>			h_nodePhases0;
public:
	Dvector<REAL>			d_forces;
	vector<REAL>			_masses;
public:
	uint					_numFaces;
	uint					_numNodes;
public:
	StreamParam				*_streams;
public:
	Object() { }
	virtual ~Object() {}
public:
	inline ObjParam param(void) {
		ObjParam p;
		p._fs = d_fs._list;
		p._ns = d_ns._list;
		p._n0s = d_n0s._list;
		p._vs = d_vs._list;
		p._ms = d_ms._list;
		p._invMs = d_invMs._list;
		p._numFaces = _numFaces;
		p._numNodes = _numNodes;
		return p;
	}
public:
	virtual void	init(void);
public:
	void	addElements(Mesh* mesh, REAL mass);
	void	initVelocities(void);
	void	initNoramls(void);
	void	initBVH(void);
public:
	void	compGravityForce(REAL3& gravity);
	void	compRotationForce(Dvector<REAL3>& pivots, Dvector<REAL3>& rotations, REAL invdt2);
	void	compPredictPosition(REAL dt);
	void	updateVelocity(REAL invdt);
	void	updateVelocity(Dvector<REAL>& ns, Dvector<REAL>& n0s, Dvector<REAL>& vs, REAL invdt, uint numNodes);
	void	updatePosition(REAL dt);
	void	updatePosition(Dvector<REAL>& ns, Dvector<REAL>& vs, REAL dt, uint numNodes);
	void	maxVelocitiy(REAL& maxVel);
	void	computeNormal(void);
	void	Damping(REAL airDamp, REAL lapDamp, uint lapIter);
public:
	void	draw(float* frontColor, float* backColor, bool smooth, bool phaseColor = false);
	void	drawWire(void);
	void	drawSurface(float* frontColor, float* backColor, bool smooth, bool phaseColor);
public:
	void	copyToDevice(void);
	void	copyToHost(void);
	void	copyNbToDevice(void);
	void	copyNbToHost(void);
	void	copyMassToDevice(void);
	void	copyMassToHost(void);
	void	copyNormToDevice(void);
	void	copyNormToHost(void);
};
#endif