#ifndef __MESH_H__
#define __MESH_H__

#pragma once
#include <fstream>
#include <string>
#include "../include/GL/freeglut.h"
#include "../include/CUDA_Custom/PrefixArray.h"

struct ObjParam {
	uint* _fs;
	REAL* _ns;
	REAL* _n0s;
	REAL* _vs;
	REAL* _invMs;
	REAL* _ms;
	uint _numNodes;
	uint _numFaces;
};

class Mesh
{
public:
	vector<uint>		_fs;
	vector<REAL>		_ns;
	//Stretch Edge
	PrefixArray<uint>	_ses;
	//Bending Edge
	PrefixArray<uint>	_bes;
	PrefixArray<uint>	_nbFs;
	PrefixArray<uint>	_nbNs;
public:
	vector<REAL>		_fnorms;
	vector<REAL>		_vnorms;
public:
	AABB				_aabb;
	uint				_numFaces;
	uint				_numVertices;
public:
	Mesh() { }
	Mesh(const char* filename, REAL3 center, REAL scale = (REAL)1.0) {
		loadObj(filename, center, scale);
	}
	Mesh(const char* filename) {
		loadObj(filename);
	}
	~Mesh() {}
public:
	void	loadObj(const char* filename, REAL3 center, REAL scale);
	void	loadObj(const char* filename);
	void	moveCenter(REAL3 center, REAL scale);
	void	buildAdjacency(void);
	void	computeNormal(void);
	void	rotate(REAL3 degree);
};

static __inline__ __device__ 
void getIfromParam(const uint* is, uint i, uint* ino) {
	i *= 3u;
	ino[0] = is[i++];
	ino[1] = is[i++];
	ino[2] = is[i];
}
static __inline__ __device__ 
void getXfromParam(const REAL* X, uint* ino, REAL3* ps) {
	uint i = ino[0] * 3u;
	ps[0].x = X[i++]; ps[0].y = X[i++]; ps[0].z = X[i];
	i = ino[1] * 3u;
	ps[1].x = X[i++]; ps[1].y = X[i++]; ps[1].z = X[i];
	i = ino[2] * 3u;
	ps[2].x = X[i++]; ps[2].y = X[i++]; ps[2].z = X[i];
}

#endif