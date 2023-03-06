#ifndef __PRIMAL_TREE_H__
#define __PRIMAL_TREE_H__

#pragma once
#include "BVH.h"
#include <algorithm>

struct FaceNormal {
	REAL3 eNorm[3];
	REAL3 fNorm;
};
struct MeshDevice {
	REAL3* vNorms;
	FaceNormal* fNorms;
};
struct KDNode
{
	REAL divPos;
	union {
		uint divAxis;
		uint index;
	};
	int level;
};
struct KDTree {
	uint maxLevel;
	uint leafNum;
	AABB aabb;
	uint fnum;
	MeshDevice mesh;
public:
	KDNode* nodes;
	AABB* aabbs;
	uint* inds;
	uint* faces;
};

struct PRINode {
	REAL3 pos;
	REAL dist;
	uint child;
	int level;
};
struct PRITree {
	PRINode *_nodes;
	REAL3	_center;
	REAL	_half;
	REAL	_scale;
};

class PrimalTree {
public:
	PRITree d_tree;
public:
	uint _maxLevel;
	uint _nodeNum;
public:
	REAL _error;
public:
	PrimalTree() {
		d_tree._scale = 1.0;
		_error = 0.0000625;
		initConstant();
	}
	virtual ~PrimalTree() {
		CUDA_CHECK(cudaFree(d_tree._nodes));
	}
public:
	inline void initKDTreeDevice(KDTree& tree) {
		uint nodeNum = (1 << tree.maxLevel + 1) - 1;
		CUDA_CHECK(cudaMalloc((void**)&tree.nodes, nodeNum * sizeof(KDNode)));
		CUDA_CHECK(cudaMalloc((void**)&tree.aabbs, nodeNum * sizeof(AABB)));
		CUDA_CHECK(cudaMalloc((void**)&tree.inds, (tree.leafNum + 1u) * sizeof(uint)));
		CUDA_CHECK(cudaMalloc((void**)&tree.faces, tree.fnum * sizeof(uint)));
	}
	inline void destroyKDTreeDevice(KDTree& tree) {
		CUDA_CHECK(cudaFree(tree.nodes));
		CUDA_CHECK(cudaFree(tree.aabbs));
		CUDA_CHECK(cudaFree(tree.inds));
		CUDA_CHECK(cudaFree(tree.faces));
		CUDA_CHECK(cudaFree(tree.mesh.fNorms));
		CUDA_CHECK(cudaFree(tree.mesh.vNorms));
	}
	static void initConstant(void);
public:
	void buildKDTree(
		KDTree& tree, const ObjParam& h_objParams, const PrefixArray<uint>& nbFs,
		const vector<REAL>& fNorms, const vector<REAL>& nNorms);
	void buildTree(const ObjParam& h_objParams, const ObjParam& d_objParams, 
		const PrefixArray<uint>& nbFs, const vector<REAL>& fNorms, const vector<REAL>& nNorms, 
		REAL delta, uint maxLevel);
public:
	void draw(const AABB& aabb);
	void draw(void);
};

#endif