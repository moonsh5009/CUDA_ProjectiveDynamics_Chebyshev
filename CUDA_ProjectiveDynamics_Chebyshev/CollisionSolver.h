#ifndef __COLLISION_SOLVER_H__
#define __COLLISION_SOLVER_H__

#pragma once
#include "BVH.h"

//----------------------------------------------
#define USED_LASTBVTT
//----------------------------------------------
//#define USED_SDF
//----------------------------------------------
//#define COLLISION_TESTTIMER
//----------------------------------------------

struct ContactElem {
	bool _isFV;
	REAL _info;
	uchar _isObs; // 0: cloth cloth, 1: cloth, obs, 2: obs, cloth
	REAL3 _norm;
	uint _i[4];
	REAL _w[2];
};
struct ContactElemParam {
	ContactElem	*_elems;
	uint		*d_tmp;
	uint		_size;
};
class ContactElems {
public:
	Dvector<ContactElem> _elems;
	Dvector<uint>		d_tmp;
	uint				_size;
public:
	ContactElems() { d_tmp.resize(1); _size = 0u; }
	~ContactElems() {}
public:
	inline void resize(void) {
		_elems.resize(_size);
	}
	inline void resize(uint size) {
		_elems.resize(size);
		_size = size;
	}
	inline void extend(void) {
		_elems.extend(_size);
	}
	inline void clear(void) {
		_elems.clear();
		d_tmp.clear();
		_size = 0u;
	}
	inline ContactElemParam param(void) {
		ContactElemParam p;
		p._elems = _elems._list;
		p.d_tmp = d_tmp._list;
		p._size = _size;
		return p;
	}
};
struct ContactElem_CMP
{
	__host__ __device__
		bool operator()(const ContactElem& a, const ContactElem& b) {
		if (a._info != b._info)
			return a._info < b._info;
		if (a._i[0] != b._i[0])
			return a._i[0] < b._i[0];
		if (a._i[1] != b._i[1])
			return a._i[1] < b._i[1];
		if (a._i[2] != b._i[2])
			return a._i[2] < b._i[2];
		return a._i[3] < b._i[3];
	}
};

struct ContactElemSDF {
	REAL _dist;
	REAL3 _norm;
	REAL _w[2];
};
struct ContactElemSDFParam {
	ContactElemSDF	*_felems;
	ContactElemSDF	*_nelems;
	uint			*d_tmp;
	uint			_size;
};
class ContactElemsSDF {
public:
	Dvector<ContactElemSDF> _felems;
	Dvector<ContactElemSDF> _nelems;
public:
	ContactElemsSDF() {}
	~ContactElemsSDF() {}
public:
	inline void resize(uint numFaces, uint numNodes) {
		_felems.resize(numFaces);
		_nelems.resize(numNodes);
	}
	inline ContactElemSDFParam param(void) {
		ContactElemSDFParam p;
		p._felems = _felems._list;
		p._nelems = _nelems._list;
		return p;
	}
};

typedef vector<set<uint2, uint2_CMP>> RIZone;
struct RIZoneParam {
	uint2	*_ids;
	uint	*_zones;
	uint	_size;
};
class DRIZone {
public:
	Dvector<uint2>	_ids;
	Dvector<uint>	_zones;
public:
	DRIZone() {}
	virtual ~DRIZone() {}
public:
	inline void clear(void) {
		_ids.clear();
		_zones.clear();
	}
	inline RIZoneParam param(void) {
		RIZoneParam p;
		p._ids = _ids._list;
		p._zones = _zones._list;
		p._size = _zones.size() - 1u;
		return p;
	}
};

#endif