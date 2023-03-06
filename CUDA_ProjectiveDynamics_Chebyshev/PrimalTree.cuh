#ifndef __PRIMAL_TREE_CUH__
#define __PRIMAL_TREE_CUH__

#pragma once
#include "PrimalTree.h"
#include "../include/CUDA_Custom/DeviceManager.h"
#include "BVH.h"

#define FIND_TRISDFDIST_ITERATION	30
#define INV_GOLDEN_RATIO 0.3819660112501052;//1 - 1 / ((sqrt(5.0) + 1.0) * 0.5);

__constant__ uchar D_NUM_P[8][8];
__constant__ uchar D_NUM_C[8][8];

inline __forceinline__ __device__ REAL ssqrt(REAL d) {
	if (d == 0.0) return d;
	return d / sqrt(fabs(d));
}
inline __forceinline__ __device__ REAL Lerp(REAL d0, REAL d1, REAL x) {
	return d0 * (1.0 - x) + d1 * x;
}
inline __forceinline__ __device__ REAL Lerp(REAL d0, REAL d1) {
	return (d0 + d1) * 0.5;
}
inline __forceinline__ __device__ REAL biLerp(REAL d0, REAL d1, REAL d2, REAL d3, REAL2 p) {
	REAL x1 = d0 * (1.0 - p.x) + d1 * p.x;
	REAL x2 = d2 * (1.0 - p.x) + d3 * p.x;
	return x1 * (1.0 - p.y) + x2 * p.y;
}
inline __forceinline__ __device__ REAL biLerp(REAL d0, REAL d1, REAL d2, REAL d3) {
	return (d0 + d1 + d2 + d3) * 0.25;
}
inline __forceinline__ __device__ REAL triLerp(REAL d0, REAL d1, REAL d2, REAL d3, REAL d4, REAL d5, REAL d6, REAL d7, REAL3 p) {
	REAL x1 = d0 * (1.0 - p.x) + d1 * p.x;
	REAL x2 = d2 * (1.0 - p.x) + d3 * p.x;
	REAL x3 = d4 * (1.0 - p.x) + d5 * p.x;
	REAL x4 = d6 * (1.0 - p.x) + d7 * p.x;
	REAL y1 = x1 * (1.0 - p.y) + x2 * p.y;
	REAL y2 = x3 * (1.0 - p.y) + x4 * p.y;
	return y1 * (1.0 - p.z) + y2 * p.z;
}
inline __forceinline__ __device__ REAL triLerp(REAL d0, REAL d1, REAL d2, REAL d3, REAL d4, REAL d5, REAL d6, REAL d7) {
	return (d0 + d1 + d2 + d3 + d4 + d5 + d6 + d7) * 0.125;
}

inline __forceinline__ __device__ REAL getDistSqrPointToTri(
	REAL3 p,
	const MeshDevice& mesh,
	uint iface,
	REAL3* vs, uint* ivs)
{
	auto fnorm = mesh.fNorms[iface];
	REAL v0pDotv01 = Dot(p - vs[0], vs[1] - vs[0]);
	REAL v0pDotv02 = Dot(p - vs[0], vs[2] - vs[0]);
	REAL v01Dotv01 = Dot(vs[1] - vs[0], vs[1] - vs[0]);
	REAL v02Dotv02 = Dot(vs[2] - vs[0], vs[2] - vs[0]);
	REAL v01Dotv02 = Dot(vs[1] - vs[0], vs[2] - vs[0]);
	REAL v1pDotv12 = v0pDotv02 - v0pDotv01 - v01Dotv02 + v01Dotv01;
	REAL result = 0.0;
	bool term0 = v0pDotv01 <= 0;
	bool term1 = v01Dotv01 - v0pDotv01 <= 0;
	bool term2 = v0pDotv01 - v0pDotv02 - v01Dotv02 + v02Dotv02 <= 0;

	if (term0 && v0pDotv02 <= 0) {
		p -= vs[0];
		vs[0] = mesh.vNorms[ivs[0]];
	}
	else if (v1pDotv12 <= 0 && term1) {
		p -= vs[1];
		vs[0] = mesh.vNorms[ivs[1]];
	}
	else if (v02Dotv02 - v0pDotv02 <= 0 && term2) {
		p -= vs[2];
		vs[0] = mesh.vNorms[ivs[2]];
	}
	else if (v0pDotv01 * v01Dotv02 - v0pDotv02 * v01Dotv01 >= 0 && !term0 && !term1) {
		p -= vs[0];
		result -= v0pDotv01 * (v0pDotv01 / v01Dotv01);
		vs[0] = fnorm.eNorm[0];
	}
	else if ((v0pDotv01 - v01Dotv01) * (v02Dotv02 - v01Dotv02) - (v0pDotv02 - v01Dotv02) * (v01Dotv02 - v01Dotv01) >= 0 && !term2) {
		p -= vs[1];
		result -= v1pDotv12 * v1pDotv12 / (v01Dotv01 + v02Dotv02 - v01Dotv02 - v01Dotv02);
		vs[0] = fnorm.eNorm[1];
	}
	else if (v0pDotv02 * v01Dotv02 - v0pDotv01 * v02Dotv02 >= 0) {
		p -= vs[0];
		result -= v0pDotv02 * (v0pDotv02 / v02Dotv02);
		vs[0] = fnorm.eNorm[2];
	}
	else {
		result = Dot(fnorm.fNorm, p - vs[0]);
		return  fabs(result) * result;
	}
	result += LengthSquared(p);
	if (Dot(vs[0], p) < 0)
		return -result;
	return result;
}

inline __forceinline__ __device__ void getNearestKDNode(
	const REAL3& p, 
	const KDTree& kdTree, 
	uint& istart, 
	uint& iend,
	uint& curr) 
{
	KDNode node;
	curr = 0;
	while (1) {
		node = kdTree.nodes[curr];
		if (node.level < 0) {
			istart = kdTree.inds[node.index];
			iend = kdTree.inds[node.index + 1];
			break;
		}
		curr = (curr << 1) + 1 + (uint)(((REAL*)&p)[node.divAxis] > node.divPos);
	}
}
inline __forceinline__ __device__ void getMinDistSqrKDNode(
	const REAL3& p,
	const KDTree& kdTree,
	const ObjParam& objParams,
	const uint istart, 
	const uint iend,
	REAL& minDistSqr)
{
	REAL3 vs[3];
	uint iv[3];
	uint iface;
	for (uint i = istart; i < iend; i++) {
		iface = kdTree.faces[i];
		iv[0] = objParams._fs[iface * 3 + 0];
		iv[1] = objParams._fs[iface * 3 + 1];
		iv[2] = objParams._fs[iface * 3 + 2];
		vs[0] = make_REAL3(objParams._ns[iv[0] * 3 + 0], objParams._ns[iv[0] * 3 + 1], objParams._ns[iv[0] * 3 + 2]);
		vs[1] = make_REAL3(objParams._ns[iv[1] * 3 + 0], objParams._ns[iv[1] * 3 + 1], objParams._ns[iv[1] * 3 + 2]);
		vs[2] = make_REAL3(objParams._ns[iv[2] * 3 + 0], objParams._ns[iv[2] * 3 + 1], objParams._ns[iv[2] * 3 + 2]);

		REAL dist = getDistSqrPointToTri(p, kdTree.mesh, iface, vs, iv);
		if (fabs(minDistSqr) > fabs(dist)) minDistSqr = dist;
	}
}
inline __forceinline__ __device__ void getFinalDistSqrKDNode(
	const REAL3& p,
	const KDTree& kdTree,
	const ObjParam& objParams,
	uint nearestNode,
	REAL& minDistSqr)
{
	uint indexes[2000];
	uint indexNum = 0;
	uint curr = 1;
	uint istart, iend;
	KDNode node;
	AABB aabb;
	REAL minDist = sqrt(fabs(minDistSqr));

	while (1) {
		node = kdTree.nodes[curr];
		aabb = kdTree.aabbs[curr];
		bool isIntersect = intersect(aabb, p, minDist);
		bool isLeaf = node.level < 0;
		if (isIntersect && !isLeaf)
			curr = (curr << 1u) + 1u;
		else {
			if (isIntersect) {
				if (curr != nearestNode) {
					indexes[indexNum + 0] = kdTree.inds[node.index];
					indexes[indexNum + 1] = kdTree.inds[node.index + 1u];
					indexNum += 2u;
				}
			}
			if (curr & 1) curr++;
			else {
				if ((curr + 2u) >> (abs(node.level) + 1u))
					break;
				else {
					do
						curr = (curr - 1) >> 1;
					while ((curr & 1) == 0);
					curr++;
				}
			}
		}
		if (indexNum >= 1000) {
			for (uint i = 0; i < indexNum; i += 2) {
				istart = indexes[i];
				iend = indexes[i + 1];
				getMinDistSqrKDNode(p, kdTree, objParams, istart, iend, minDistSqr);
			}
			minDist = sqrt(fabs(minDistSqr));
			indexNum = 0;
		}
	}
	for (uint i = 0; i < indexNum; i += 2) {
		istart = indexes[i];
		iend = indexes[i + 1];
		getMinDistSqrKDNode(p, kdTree, objParams, istart, iend, minDistSqr);
	}
	/*REAL3 vs[3];
	uint iv[3];
	uint iface;
	for (uint iface = 0; iface < objParams._numFaces; iface++) {
		iv[0] = objParams._fs[iface * 3 + 0];
		iv[1] = objParams._fs[iface * 3 + 1];
		iv[2] = objParams._fs[iface * 3 + 2];
		vs[0] = make_REAL3(objParams._ns[iv[0] * 3 + 0], objParams._ns[iv[0] * 3 + 1], objParams._ns[iv[0] * 3 + 2]);
		vs[1] = make_REAL3(objParams._ns[iv[1] * 3 + 0], objParams._ns[iv[1] * 3 + 1], objParams._ns[iv[1] * 3 + 2]);
		vs[2] = make_REAL3(objParams._ns[iv[2] * 3 + 0], objParams._ns[iv[2] * 3 + 1], objParams._ns[iv[2] * 3 + 2]);
		REAL dist = getDistSqrPointToTri(p, kdTree.mesh, iface, vs, iv);
		if (fabs(minDistSqr) > fabs(dist)) minDistSqr = dist;
	}*/
}
inline __forceinline__ __device__ REAL getDistSqrKD(
	const REAL3& p,
	const KDTree& kdTree,
	const ObjParam& objParams)
{
	REAL minDistSqr = REAL_MAX;
	uint istart, iend, nearestNode;
	getNearestKDNode(p, kdTree, istart, iend, nearestNode);
	getMinDistSqrKDNode(p, kdTree, objParams, istart, iend, minDistSqr);
	getFinalDistSqrKDNode(p, kdTree, objParams, nearestNode, minDistSqr);
	return minDistSqr;
}

inline __forceinline__ __device__ bool ComparisonPRI(
	const KDTree& kdTree,
	const ObjParam& objParams, 
	REAL* dists, 
	REAL3* corners, REAL error)
{
	if (fabs(Lerp(dists[14], dists[0]) - dists[7]) > error) //x
		return false;
	if (fabs(Lerp(dists[14], dists[1]) - dists[8]) > error) //y
		return false;
	if (fabs(Lerp(dists[14], dists[3]) - dists[10]) > error) //z
		return false;
	if (fabs(biLerp(dists[14], dists[0], dists[1], dists[2]) - dists[9]) > error) //xy
		return false;
	if (fabs(biLerp(dists[14], dists[0], dists[3], dists[4]) - dists[11]) > error) //xz
		return false;
	if (fabs(biLerp(dists[14], dists[1], dists[3], dists[5]) - dists[12]) > error) //yz
		return false;
	if (fabs(triLerp(dists[14], dists[0], dists[1], dists[2], dists[3], dists[4], dists[5],
		dists[6]) - dists[13]) > error) //center
		return false;
	for (uint i = 0; i < 3; i++) {
		uint a = 1 + i * 2;
		if (fabs(Lerp(dists[a], dists[a + 1]) - ssqrt(getDistSqrKD((corners[a] + corners[a + 1]) * 0.5,
			kdTree, objParams))) > error) // x-Lines
			return false;
		a = (3 + i) % 5;
		if (fabs(Lerp(dists[a], dists[a + 2]) - ssqrt(getDistSqrKD((corners[a] + corners[a + 2]) * 0.5,
			kdTree, objParams))) > error) // y-Lines
			return false;
		if (fabs(Lerp(dists[i], dists[i + 4]) - ssqrt(getDistSqrKD((corners[i] + corners[i + 4]) * 0.5,
			kdTree, objParams))) > error) // z-Lines
			return false;
	}
	if (fabs(biLerp(dists[0], dists[2], dists[4], dists[6]) - ssqrt(getDistSqrKD((corners[0] + corners[6])
		* 0.5, kdTree, objParams))) > error) //opposite yz
		return false;
	if (fabs(biLerp(dists[1], dists[2], dists[5], dists[6]) - ssqrt(getDistSqrKD((corners[1] + corners[6])
		* 0.5, kdTree, objParams))) > error) // opposite xz
		return false;
	if (fabs(biLerp(dists[3], dists[4], dists[5], dists[6]) - ssqrt(getDistSqrKD((corners[3] + corners[6])
		* 0.5, kdTree, objParams))) > error) // opposite xy
		return false;
	return true;
}
inline __forceinline__ __device__ uint CN(const PRITree& tree, uint i, uint index) {
	auto node = tree._nodes[i];
	if (node.level < 0) return 0;
	return node.child + index;
}
inline __forceinline__ __device__ void getDistOutofPRI(
	REAL3& p, const REAL3& center, const REAL half, const REAL scale)
{
	p -= center;
	p *= 1.0 / (scale * (half + half));
	p += 0.5;
	if (p.x < 0.0)		p.x = 0.0;
	else if (p.x > 1.0) p.x = 1.0;
	if (p.y < 0.0)		p.y = 0.0;
	else if (p.y > 1.0) p.y = 1.0;
	if (p.z < 0.0)		p.z = 0.0;
	else if (p.z > 1.0) p.z = 1.0;
}
inline __forceinline__ __device__ REAL getSphereSDF(const REAL3& pos) {
	return sqrt(pos.x * pos.x + (pos.y + 0.01) * (pos.y + 0.01) + pos.z * pos.z) - 0.4;
}
inline __forceinline__ __device__ REAL getDistancePRI(
	const PRITree& tree, REAL3 p,
	REAL3* gradient = nullptr)
{
#if 0
	if (gradient) {
		REAL eps = 1.0e-5;
		*gradient = make_REAL3(
			getSphereSDF(make_REAL3(p.x + eps, p.y, p.z)) - getSphereSDF(make_REAL3(p.x - eps, p.y, p.z)),
			getSphereSDF(make_REAL3(p.x, p.y + eps, p.z)) - getSphereSDF(make_REAL3(p.x, p.y - eps, p.z)),
			getSphereSDF(make_REAL3(p.x, p.y, p.z + eps)) - getSphereSDF(make_REAL3(p.x, p.y, p.z - eps)));
		Normalize(*gradient);
	}
	return getSphereSDF(p);
#else
	const unsigned char NUM_P[8][8] = {
		{ 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 1, 0, 1, 0, 1, 0, 1 },
		{ 0, 0, 2, 2, 0, 0, 2, 2 },
		{ 0, 1, 2, 3, 0, 1, 2, 3 },
		{ 0, 0, 0, 0, 4, 4, 4, 4 },
		{ 0, 1, 0, 1, 4, 5, 4, 5 },
		{ 0, 0, 2, 2, 4, 4, 6, 6 },
		{ 0, 1, 2, 3, 4, 5, 6, 7 } };
	const unsigned char NUM_C[8][8] = {
		{ 0, 1, 2, 3, 4, 5, 6, 7 },
		{ 1, 0, 3, 2, 5, 4, 7, 6 },
		{ 2, 3, 0, 1, 6, 7, 4, 5 },
		{ 3, 2, 1, 0, 7, 6, 5, 4 },
		{ 4, 5, 6, 7, 0, 1, 2, 3 },
		{ 5, 4, 7, 6, 1, 0, 3, 2 },
		{ 6, 7, 4, 5, 2, 3, 0, 1 },
		{ 7, 6, 5, 4, 3, 2, 1, 0 } };

	uint nodes[8];
	uint temp[8];
	REAL dists[8];

	getDistOutofPRI(p, tree._center, tree._half, tree._scale);
	for (int i = 0; i < 8; i++) {
		nodes[i] = temp[i] = i + 1;
		dists[i] = tree._nodes[nodes[i]].dist;
	}
	while (1) {
		uchar px = (p.x >= 0.5);
		uchar py = (p.y >= 0.5) << 1;
		uchar pz = (p.z >= 0.5) << 2;
		uchar nx = px ^ 1;
		uchar ny = py ^ 2;
		uchar nz = pz ^ 4;
		for (int i = 0; i < 8; i++)
			//nodes[i] = temp[D_NUM_P[px + py + pz][i]] ? CN(tree, temp[D_NUM_P[px + py + pz][i]], D_NUM_C[px + py + pz][i]) : 0;
			nodes[i] = temp[NUM_P[px + py + pz][i]] ? CN(tree, temp[NUM_P[px + py + pz][i]], NUM_C[px + py + pz][i]) : 0;

		if (!nodes[0] && !nodes[1] && !nodes[2] && !nodes[3] && !nodes[4] && !nodes[5] && !nodes[6] && !nodes[7])
			break;

		dists[nx + ny + nz] = (nodes[nx + ny + nz] ? tree._nodes[nodes[nx + ny + nz]].dist :
			triLerp(dists[0], dists[1], dists[2], dists[3], dists[4], dists[5], dists[6], dists[7]));

		dists[nx + ny + pz] = (nodes[nx + ny + pz] ? tree._nodes[nodes[nx + ny + pz]].dist :
			biLerp(dists[px + py + pz], dists[nx + py + pz], dists[px + ny + pz], dists[nx + ny + pz]));
		dists[nx + py + nz] = (nodes[nx + py + nz] ? tree._nodes[nodes[nx + py + nz]].dist :
			biLerp(dists[px + py + pz], dists[nx + py + pz], dists[px + py + nz], dists[nx + py + nz]));
		dists[px + ny + nz] = (nodes[px + ny + nz] ? tree._nodes[nodes[px + ny + nz]].dist :
			biLerp(dists[px + py + pz], dists[px + ny + pz], dists[px + py + nz], dists[px + ny + nz]));

		dists[px + ny + pz] = (nodes[px + ny + pz] ? tree._nodes[nodes[px + ny + pz]].dist :
			Lerp(dists[px + ny + pz], dists[px + py + pz]));
		dists[nx + py + pz] = (nodes[nx + py + pz] ? tree._nodes[nodes[nx + py + pz]].dist :
			Lerp(dists[nx + py + pz], dists[px + py + pz]));
		dists[px + py + nz] = (nodes[px + py + nz] ? tree._nodes[nodes[px + py + nz]].dist :
			Lerp(dists[px + py + nz], dists[px + py + pz]));

		for (int i = 0; i < 8; i++)
			temp[i] = nodes[i];

		p.x = (p.x - 0.5 * px) * 2;
		p.y = (p.y - 0.5 * (py >> 1)) * 2;
		p.z = (p.z - 0.5 * (pz >> 2)) * 2;
	}
	/*if (gradient) {
		REAL eps = 1.0e-5;
		*gradient = make_REAL3(
			getDistancePRI(tree, make_REAL3(p.x + eps, p.y, p.z)) -
			getDistancePRI(tree, make_REAL3(p.x - eps, p.y, p.z)),
			getDistancePRI(tree, make_REAL3(p.x, p.y + eps, p.z)) -
			getDistancePRI(tree, make_REAL3(p.x, p.y - eps, p.z)),
			getDistancePRI(tree, make_REAL3(p.x, p.y, p.z + eps)) -
			getDistancePRI(tree, make_REAL3(p.x, p.y, p.z - eps)));
		Normalize(*gradient);
	}*/
	if (gradient) {
		/*gradient->x = biLerp(dists[1], dists[5], dists[3], dists[7], make_REAL2(p.z, p.y))
			- biLerp(dists[0], dists[4], dists[2], dists[6], make_REAL2(p.z, p.y));
		gradient->y = biLerp(dists[2], dists[3], dists[6], dists[7], make_REAL2(p.x, p.z))
			- biLerp(dists[0], dists[1], dists[4], dists[5], make_REAL2(p.x, p.z));
		gradient->z = biLerp(dists[4], dists[5], dists[6], dists[7], make_REAL2(p.x, p.y))
			- biLerp(dists[0], dists[1], dists[2], dists[3], make_REAL2(p.x, p.y));*/
		REAL eps = 1.0e-10;
		gradient->x = triLerp(dists[0], dists[1], dists[2], dists[3], dists[4], dists[5], dists[6], dists[7], make_REAL3(min(p.x + eps, 1.0), p.y, p.z))
			- triLerp(dists[0], dists[1], dists[2], dists[3], dists[4], dists[5], dists[6], dists[7], make_REAL3(max(p.x - eps, 0.0), p.y, p.z));
		gradient->y = triLerp(dists[0], dists[1], dists[2], dists[3], dists[4], dists[5], dists[6], dists[7], make_REAL3(p.x, min(p.y + eps, 1.0), p.z))
			- triLerp(dists[0], dists[1], dists[2], dists[3], dists[4], dists[5], dists[6], dists[7], make_REAL3(p.x, max(p.y - eps, 0.0), p.z));
		gradient->z = triLerp(dists[0], dists[1], dists[2], dists[3], dists[4], dists[5], dists[6], dists[7], make_REAL3(p.x, p.y, min(p.z + eps, 1.0)))
			- triLerp(dists[0], dists[1], dists[2], dists[3], dists[4], dists[5], dists[6], dists[7], make_REAL3(p.x, p.y, max(p.z - eps, 0.0)));
		
		Normalize(*gradient);
	}
	return triLerp(dists[0], dists[1], dists[2], dists[3], dists[4], dists[5], dists[6], dists[7], p);
#endif
}
inline __forceinline__ __device__ REAL getDistancePRItoTri(
	const PRITree& tree, const REAL3* vertices, uint minId,
	REAL* u = nullptr, REAL* v = nullptr, REAL3* norm = nullptr)
{
#if 0
	const REAL3 v01 = vertices[1] - vertices[0];
	const REAL3 v02 = vertices[2] - vertices[0];
	REAL v01Dotv01 = LengthSquared(v01);
	REAL v02Dotv02 = LengthSquared(v02);
	REAL v01Dotv02 = Dot(v01, v02);
	REAL invdet = 1.0 / (v01Dotv01 * v02Dotv02 - v01Dotv02 * v01Dotv02);
	REAL w0, w1, w2;
	w0 = w1 = 1.0 / 3.0;
	w2 = 1.0 - w0 - w1;

	REAL3 p, grad;
	REAL mindist, dist;
	p = vertices[0] * w0 + vertices[1] * w1 + vertices[2] * w2;
	mindist = dist = getDistancePRI(tree, p, center, half, scale, &grad);
	*norm = grad;
	*u = w0;
	*v = w1;
	for (uint i = 0; i < FIND_TRISDFDIST_ITERATION; i++) {
		p -= grad * dist * 0.98;
		{
			REAL v0pDotv01 = Dot(p - vertices[0], v01);
			REAL v0pDotv02 = Dot(p - vertices[0], v02);
			REAL v1pDotv12 = v0pDotv02 - v0pDotv01 - v01Dotv02 + v01Dotv01;
			if (v0pDotv01 < 0.0 && v0pDotv02 < 0.0) {
				w0 = 1.0;
				w1 = 0.0;
			}
			else if (v0pDotv02 + v01Dotv01 < v0pDotv01 + v01Dotv02 && v01Dotv01 < v0pDotv01) {
				w0 = 0.0;
				w1 = 1.0;
			}
			else if (v02Dotv02 < v0pDotv02 && v0pDotv01 + v02Dotv02 < v0pDotv02 + v01Dotv02) {
				w0 = 0.0;
				w1 = 0.0;
			}
			else if (v0pDotv01 * v01Dotv02 > v0pDotv02 * v01Dotv01 && v0pDotv01 >= 0.0 && v01Dotv01 >= v0pDotv01) {
				w1 = v0pDotv01 / v01Dotv01;
				w0 = 1.0 - w1;
			}
			else if ((v0pDotv01 - v01Dotv01) * (v02Dotv02 - v01Dotv02) > (v0pDotv02 - v01Dotv02) * (v01Dotv02 - v01Dotv01) && v0pDotv01 + v02Dotv02 >= v0pDotv02 + v01Dotv02) {
				w0 = 0.0;
				w1 = 1.0 - v1pDotv12 / (v01Dotv01 + v02Dotv02 - v01Dotv02 - v01Dotv02);
			}
			else if (v0pDotv02 * v01Dotv02 > v0pDotv01 * v02Dotv02) {
				w0 = 1.0 - v0pDotv02 / v02Dotv02;
				w1 = 0.0;
			}
			else {
				w1 = (v02Dotv02 * v0pDotv01 - v01Dotv02 * v0pDotv02) * invdet;
				w0 = 1.0 - w1 - (v01Dotv01 * v0pDotv02 - v01Dotv02 * v0pDotv01) * invdet;
			}
			w2 = 1.0 - w0 - w1;
		}
		p = vertices[0] * w0 + vertices[1] * w1 + vertices[2] * w2;
		dist = getDistancePRI(tree, p, center, half, scale, &grad);
		if (fabs(mindist) > fabs(dist)) {
			mindist = dist;
			*norm = grad;
			*u = w0;
			*v = w1;
		}
	}
	return mindist;
#else
	REAL3 xi = vertices[minId];
	REAL3 grad, xs;
	REAL minXco, tmp;
	for (int itr = 0; itr < FIND_TRISDFDIST_ITERATION; itr++) {
		getDistancePRI(tree, xi, &grad);
		minXco = Dot(vertices[minId = 0], grad);
		if (minXco > (tmp = Dot(vertices[1], grad))) {
			minXco = tmp;
			minId = 1;
		}
		if (minXco > Dot(vertices[2], grad)) minId = 2;
		xs = vertices[minId] - xi;
		if (-Dot(xs, grad) < 1.0e-10)
			break;
		xi += xs * 2.0 / (REAL)(itr + 2);
	}
	if (u) {
		REAL3 v20 = vertices[0] - vertices[2];
		REAL3 v21 = vertices[1] - vertices[2];
		REAL3 v2x = xi - vertices[2];
		grad = Cross(v20, v21);
		REAL invn = Length(grad);
		if (invn < 1.0e-20) printf("Error Normal\n");
		invn = 1.0 / invn;
		grad *= invn;
		*u = Dot(Cross(v2x, v21), grad) * invn;
		*v = Dot(Cross(v20, v2x), grad) * invn;
		//printf("%f, %f, (%f, %f, %f)\n", *u, *v, xi.x, xi.y, xi.z);
	}
	return getDistancePRI(tree, xi, norm);
#endif
}
inline __forceinline__ __device__ REAL getPredictDistPRItoTri(
	const PRITree& tree, const REAL3* vertices, const REAL3* predVertices,  uint minId,
	REAL* u, REAL* v, REAL3* norm)
{
	REAL3 xi = vertices[minId];
	REAL3 grad, xs;
	REAL minXco, tmp;
	for (int itr = 0; itr < FIND_TRISDFDIST_ITERATION; itr++) {
		getDistancePRI(tree, xi, &grad);
		minXco = Dot(predVertices[minId = 0], grad);
		if (minXco > (tmp = Dot(predVertices[1], grad))) {
			minXco = tmp;
			minId = 1;
		}
		if (minXco > Dot(predVertices[2], grad)) minId = 2;
		xs = vertices[minId] - xi;
		if (-Dot(xs, grad) < 1.0e-10)
			break;
		xi += xs * 2.0 / (REAL)(itr + 2);
	}
	if (u) {
		REAL3 v20 = vertices[0] - vertices[2];
		REAL3 v21 = vertices[1] - vertices[2];
		REAL3 v2x = xi - vertices[2];
		grad = Cross(v20, v21);
		REAL invn = Length(grad);
		if (invn < 1.0e-20) printf("Error Normal\n");
		invn = 1.0 / invn;
		grad *= invn;
		*u = Dot(Cross(v2x, v21), grad) * invn;
		*v = Dot(Cross(v20, v2x), grad) * invn;
		//printf("%f, %f, (%f, %f, %f)\n", *u, *v, xi.x, xi.y, xi.z);
	}
	return getDistancePRI(tree, xi, norm);
}
inline __forceinline__ __device__ REAL getDistancePRItoEdge(
	const PRITree& tree, const REAL3& v0, const REAL3& v1,
	const REAL3& center, const REAL half, const REAL scale,
	REAL* w = nullptr, REAL3* norm = nullptr)
{
#if 1
	REAL3 xi, grad;
	REAL w0 = 0.0, w1 = 1.0, c = 0.5;
	REAL u, v;
	for (int itr = 0; itr < FIND_TRISDFDIST_ITERATION; itr++) {
		u = w0 + (w1 - w0) * INV_GOLDEN_RATIO;
		v = w1 + (w0 - w1) * INV_GOLDEN_RATIO;
		xi = v0 + (v1 - v0) * c;
		if (getDistancePRI(tree, v0 + (v1 - v0) * u) <
			getDistancePRI(tree, v0 + (v1 - v0) * v))
			w1 = v;
		else w0 = u;
		c = (w0 + w1) * 0.5;
	}
	xi = v0 + (v1 - v0) * c;
	if (w) *w = c;
	return getDistancePRI(tree, xi, norm);
#else
	REAL3 v01 = v1 - v0;
	REAL invlenSqr = LengthSquared(v01);
	if (invlenSqr < 1.0e-10)
		printf("Normalize Error\n");
	invlenSqr = 1.0 / invlenSqr;
	REAL u = 0.5;

	REAL3 p, grad;
	REAL mindist, dist;
	p = v0 + (v1 - v0) * u;
	mindist = dist = getDistancePRI(tree, p, center, half, scale, &grad);
	if (w) {
		*norm = grad;
		*w = u;
	}
	for (uint i = 0; i < FIND_TRISDFDIST_ITERATION; i++) {
		//if (threadIdx.x + blockDim.x * blockIdx.x == 1636)
		//	printf("%d, %f, (%f, %f, %f), (%f, %f, %f), %f\n", i, dist, p.x, p.y, p.z, grad.x, grad.y, grad.z, w);
		p -= grad * dist * 0.98;
		u = Dot(v01, p - v0) * invlenSqr;
		if (u < 0.0)		u = 0.0;
		else if (u > 1.0)	u = 1.0;

		p = v0 + (v1 - v0) * u;
		dist = getDistancePRI(tree, p, center, half, scale, &grad);
		if (fabs(mindist) > fabs(dist)) {
			mindist = dist;
			if (w) {
				*norm = grad;
				*w = u;
			}
		}
	}
	return mindist;
	//printf("%d, %f, (%f, %f, %f), (%f, %f, %f), %f\n", threadIdx.x + blockDim.x * blockIdx.x, dist, p.x, p.y, p.z, norm->x, norm->y, norm->z, *u);
#endif
}

#endif