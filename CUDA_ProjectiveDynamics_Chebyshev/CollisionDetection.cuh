#ifndef __COLLISION_DETECTION_CUH__
#define __COLLISION_DETECTION_CUH__

#pragma once
#include "CollisionManager.cuh"

//-------------------------------------------------------------------------
inline __device__ bool Culling_Proximity(
	bool isFV,
	const REAL3& p0, const REAL3& p1, const REAL3& p2, const REAL3& p3,
	REAL delta)
{
	REAL l0, l1, l2, l3, l4, l5;
	REAL r0, r1, r2, r3, r4, r5;

	l0 = l3 = p0.x;
	l1 = l4 = p0.y;
	l2 = l5 = p0.z;
	r0 = r3 = p3.x;
	r1 = r4 = p3.y;
	r2 = r5 = p3.z;

	if (l0 > p1.x) l0 = p1.x;
	if (l1 > p1.y) l1 = p1.y;
	if (l2 > p1.z) l2 = p1.z;
	if (l3 < p1.x) l3 = p1.x;
	if (l4 < p1.y) l4 = p1.y;
	if (l5 < p1.z) l5 = p1.z;
	if (isFV) {
		if (l0 > p2.x) l0 = p2.x;
		if (l1 > p2.y) l1 = p2.y;
		if (l2 > p2.z) l2 = p2.z;
		if (l3 < p2.x) l3 = p2.x;
		if (l4 < p2.y) l4 = p2.y;
		if (l5 < p2.z) l5 = p2.z;
	}
	else {
		if (r0 < p2.x) r0 = p2.x;
		if (r1 < p2.y) r1 = p2.y;
		if (r2 < p2.z) r2 = p2.z;
		if (r3 > p2.x) r3 = p2.x;
		if (r4 > p2.y) r4 = p2.y;
		if (r5 > p2.z) r5 = p2.z;
	}

	return !(
		l0 - delta > r0 || l1 - delta > r1 || l2 - delta > r2 ||
		l3 + delta < r3 || l4 + delta < r4 || l5 + delta < r5);
}
inline __device__ bool Culling_CCD(
	bool isFV,
	const REAL3& p0, const REAL3& p1, const REAL3& p2, const REAL3& p3,
	const REAL3& q0, const REAL3& q1, const REAL3& q2, const REAL3& q3,
	REAL delta)
{
	REAL l0, l1, l2, l3, l4, l5;
	REAL r0, r1, r2, r3, r4, r5;

	l0 = l3 = p0.x;
	l1 = l4 = p0.y;
	l2 = l5 = p0.z;
	r0 = r3 = p3.x;
	r1 = r4 = p3.y;
	r2 = r5 = p3.z;

	if (l0 > p1.x) l0 = p1.x; if (l0 > q0.x) l0 = q0.x; if (l0 > q1.x) l0 = q1.x;
	if (l1 > p1.y) l1 = p1.y; if (l1 > q0.y) l1 = q0.y; if (l1 > q1.y) l1 = q1.y;
	if (l2 > p1.z) l2 = p1.z; if (l2 > q0.z) l2 = q0.z; if (l2 > q1.z) l2 = q1.z;
	if (l3 < p1.x) l3 = p1.x; if (l3 < q0.x) l3 = q0.x; if (l3 < q1.x) l3 = q1.x;
	if (l4 < p1.y) l4 = p1.y; if (l4 < q0.y) l4 = q0.y; if (l4 < q1.y) l4 = q1.y;
	if (l5 < p1.z) l5 = p1.z; if (l5 < q0.z) l5 = q0.z; if (l5 < q1.z) l5 = q1.z;

	if (r0 < q3.x) r0 = q3.x;
	if (r1 < q3.y) r1 = q3.y;
	if (r2 < q3.z) r2 = q3.z;
	if (r3 > q3.x) r3 = q3.x;
	if (r4 > q3.y) r4 = q3.y;
	if (r5 > q3.z) r5 = q3.z;

	if (isFV) {
		if (l0 > p2.x) l0 = p2.x; if (l0 > q2.x) l0 = q2.x;
		if (l1 > p2.y) l1 = p2.y; if (l1 > q2.y) l1 = q2.y;
		if (l2 > p2.z) l2 = p2.z; if (l2 > q2.z) l2 = q2.z;
		if (l3 < p2.x) l3 = p2.x; if (l3 < q2.x) l3 = q2.x;
		if (l4 < p2.y) l4 = p2.y; if (l4 < q2.y) l4 = q2.y;
		if (l5 < p2.z) l5 = p2.z; if (l5 < q2.z) l5 = q2.z;
	}
	else {
		if (r0 < p2.x) r0 = p2.x; if (r0 < q2.x) r0 = q2.x;
		if (r1 < p2.y) r1 = p2.y; if (r1 < q2.y) r1 = q2.y;
		if (r2 < p2.z) r2 = p2.z; if (r2 < q2.z) r2 = q2.z;
		if (r3 > p2.x) r3 = p2.x; if (r3 > q2.x) r3 = q2.x;
		if (r4 > p2.y) r4 = p2.y; if (r4 > q2.y) r4 = q2.y;
		if (r5 > p2.z) r5 = p2.z; if (r5 > q2.z) r5 = q2.z;
	}

	return !(
		l0 - delta > r0 || l1 - delta > r1 || l2 - delta > r2 ||
		l3 + delta < r3 || l4 + delta < r4 || l5 + delta < r5);
}
inline __device__ bool Culling_CCD(
	const REAL3& p0, const REAL3& p1, const REAL3& p2,
	const REAL3& q0, const REAL3& q1, const REAL3& q2,
	REAL delta)
{
	REAL l0, l1, l2, l3, l4, l5;
	REAL r0, r1, r2, r3, r4, r5;

	l0 = l3 = p0.x;
	l1 = l4 = p0.y;
	l2 = l5 = p0.z;
	r0 = r3 = p2.x;
	r1 = r4 = p2.y;
	r2 = r5 = p2.z;

	if (l0 > p1.x) l0 = p1.x; if (l0 > q0.x) l0 = q0.x; if (l0 > q1.x) l0 = q1.x;
	if (l1 > p1.y) l1 = p1.y; if (l1 > q0.y) l1 = q0.y; if (l1 > q1.y) l1 = q1.y;
	if (l2 > p1.z) l2 = p1.z; if (l2 > q0.z) l2 = q0.z; if (l2 > q1.z) l2 = q1.z;
	if (l3 < p1.x) l3 = p1.x; if (l3 < q0.x) l3 = q0.x; if (l3 < q1.x) l3 = q1.x;
	if (l4 < p1.y) l4 = p1.y; if (l4 < q0.y) l4 = q0.y; if (l4 < q1.y) l4 = q1.y;
	if (l5 < p1.z) l5 = p1.z; if (l5 < q0.z) l5 = q0.z; if (l5 < q1.z) l5 = q1.z;

	if (r0 < q2.x) r0 = q2.x;
	if (r1 < q2.y) r1 = q2.y;
	if (r2 < q2.z) r2 = q2.z;
	if (r3 > q2.x) r3 = q2.x;
	if (r4 > q2.y) r4 = q2.y;
	if (r5 > q2.z) r5 = q2.z;

	return !(
		l0 - delta > r0 || l1 - delta > r1 || l2 - delta > r2 ||
		l3 + delta < r3 || l4 + delta < r4 || l5 + delta < r5);
}
inline __device__ bool Culling_CCD(
	const REAL3& p0, const REAL3& p1,
	const REAL3& q0, const REAL3& q1,
	REAL delta)
{
	REAL l0, l1, l2, l3, l4, l5;
	REAL r0, r1, r2, r3, r4, r5;

	l0 = l3 = p0.x;
	l1 = l4 = p0.y;
	l2 = l5 = p0.z;
	r0 = r3 = p1.x;
	r1 = r4 = p1.y;
	r2 = r5 = p1.z;

	if (l0 > q0.x) l0 = q0.x;
	if (l1 > q0.y) l1 = q0.y;
	if (l2 > q0.z) l2 = q0.z;
	if (l3 < q0.x) l3 = q0.x;
	if (l4 < q0.y) l4 = q0.y;
	if (l5 < q0.z) l5 = q0.z;

	if (r0 < q1.x) r0 = q1.x;
	if (r1 < q1.y) r1 = q1.y;
	if (r2 < q1.z) r2 = q1.z;
	if (r3 > q1.x) r3 = q1.x;
	if (r4 > q1.y) r4 = q1.y;
	if (r5 > q1.z) r5 = q1.z;

	return !(
		l0 - delta > r0 || l1 - delta > r1 || l2 - delta > r2 ||
		l3 + delta < r3 || l4 + delta < r4 || l5 + delta < r5);
}
inline __device__ bool Culling_Index(bool isFV, uint i0, uint i1, uint i2, uint i3) {
	return (isFV && i3 != i0 && i3 != i1 && i3 != i2) || (!isFV && i0 != i2 && i0 != i3 && i1 != i2 && i1 != i3);
}
//-------------------------------------------------------------------------
inline __device__ bool Culling_barycentricRTri(REAL w0, REAL w1, uint RTri) {
	return
		(w0 != 0.0 && w1 != 0.0 && w0 + w1 != 1.0 && w0 != 1.0 && w1 != 1.0 && w0 + w1 != 0.0) ||
		(w0 == 1.0 && w1 == 0.0 && RTriVertex(RTri, 0u)) ||
		(w0 == 0.0 && w1 == 1.0 && RTriVertex(RTri, 1u)) ||
		(w0 == 0.0 && w1 == 0.0 && RTriVertex(RTri, 2u)) ||
		(w0 + w1 == 1.0 && w0 != 0.0 && w0 != 1.0 && RTriEdge(RTri, 0u)) ||
		(w0 == 0.0 && w1 != 0.0 && w1 != 1.0 && RTriEdge(RTri, 1u)) ||
		(w1 == 0.0 && w0 != 0.0 && w0 != 1.0 && RTriEdge(RTri, 2u));
}
inline __device__ bool Culling_barycentricRTri(
	REAL w0, REAL w1, uint lRTri, uint rRTri, uint i, uint i1, uint j, uint j1) 
{
	return 
		((w0 != 0.0 && w0 != 1.0) || (w0 == 1.0 && RTriVertex(lRTri, i)) || (w0 == 0.0 && RTriVertex(lRTri, i1))) &&
		((w1 != 0.0 && w1 != 1.0) || (w1 == 1.0 && RTriVertex(rRTri, j)) || (w1 == 0.0 && RTriVertex(rRTri, j1)));
}
//-------------------------------------------------------------------------
inline __device__ bool isDetected_Proximity(
	bool isFV,
	const REAL3& p0, const REAL3& p1, const REAL3& p2, const REAL3& p3,
	REAL thickness, REAL* w0 = nullptr, REAL* w1 = nullptr)
{
	bool result = false;

	if (Culling_Proximity(isFV, p0, p1, p2, p3, thickness)) {
		REAL dist = getDistanceElements(isFV, p0, p1, p2, p3, thickness, w0, w1);
		//REAL dist = getDistanceElements(isFV, p0, p1, p2, p3, thickness * COL_THICKNESS_RATIO, w0, w1);
		if (dist < thickness) 
			result = true;
	}
	return result;
}
inline __device__ bool isDetected_Proximity(
	bool isFV,
	const REAL3& p0, const REAL3& p1, const REAL3& p2, const REAL3& p3,
	REAL thickness, REAL3& norm, REAL* w0 = nullptr, REAL* w1 = nullptr)
{
	bool result = false;

	if (Culling_Proximity(isFV, p0, p1, p2, p3, thickness)) {
		REAL dist = getDistanceElements(isFV, p0, p1, p2, p3, thickness, norm, w0, w1);
		//REAL dist = getDistanceElements(isFV, p0, p1, p2, p3, thickness * COL_THICKNESS_RATIO, norm, w0, w1);
		if (dist < thickness) {
			result = true;
		}
	}
	return result;
}
//-------------------------------------------------------------------------
inline __device__ bool isDetected_CCD(
	const REAL3& p0, const REAL3& p1, const REAL3& p2,
	const REAL3& q0, const REAL3& q1, const REAL3& q2,
	REAL thickness, REAL* t, REAL* w)
{
	bool isApplied = false;
	REAL delta = thickness * COL_CCD_THICKNESS_DETECT;
	REAL w0;
	if (Culling_CCD(p0, p1, p2, q0, q1, q2, delta)) {
		REAL ts[2], dist;
		REAL3 p0m, p1m, p2m;
		uint num = FineLineCoTime(p0, p1, p2, q0, q1, q2, ts);
		for (uint i = 0; i < num; i++) {
			if (ts[i] < *t) {
				p0m = p0 + (q0 - p0) * ts[i];
				p1m = p1 + (q1 - p1) * ts[i];
				p2m = p2 + (q2 - p2) * ts[i];
				dist = getDistanceEV(p0m, p1m, p2m, &w0);
				if (dist < delta) {
					//printf("!!!!!!!!! EV %f, %f, %f\n", *t, ts[i], w0);
					*t = ts[i];
					*w = w0;
					isApplied = true;
				}
			}
		}
	}
	return isApplied;
}
inline __device__ bool isDetected_CCD(
	const REAL3& p0, const REAL3& p1,
	const REAL3& q0, const REAL3& q1,
	REAL thickness, REAL* t)
{
	bool isApplied = false;
	REAL delta = thickness * COL_CCD_THICKNESS_DETECT;
	if (Culling_CCD(p0, p1, q0, q1, delta)) {
		REAL time, dist;
		REAL3 p0m, p1m;
		uint num = FineVertexCoTime(p0, p1, q0, q1, &time);
		if (num && time < *t) {
			p0m = p0 + (q0 - p0) * time;
			p1m = p1 + (q1 - p1) * time;
			dist = Length(p0m - p1m);
			if (dist < delta) {
				*t = time;
				isApplied = true;
			}
		}
	}
	return isApplied;
}
inline __device__ bool isDetected_CCD(
	bool isFV,
	const REAL3& p0, const REAL3& p1, const REAL3& p2, const REAL3& p3,
	const REAL3& q0, const REAL3& q1, const REAL3& q2, const REAL3& q3,
	REAL thickness, REAL* t, REAL3* norm, REAL* w0 = nullptr, REAL* w1 = nullptr)
{
	bool result = false;
	REAL delta = thickness * COL_CCD_THICKNESS_DETECT;
	*t = 10.0;
	if (Culling_CCD(isFV, p0, p1, p2, p3, q0, q1, q2, q3, delta)) {
		REAL ts[3];
		REAL3 tnorm;
		REAL dist, wa, wb;
		REAL3 p0m, p1m, p2m, p3m;
		uint num = FinePlaneCoTime(p0, p1, p2, p3, q0, q1, q2, q3, ts);
		for (uint i = 0; i < num; i++) {
			if (ts[i] < *t) {
				p0m = p0 + (q0 - p0) * ts[i];
				p1m = p1 + (q1 - p1) * ts[i];
				p2m = p2 + (q2 - p2) * ts[i];
				p3m = p3 + (q3 - p3) * ts[i];
				dist = getDistanceElements(isFV, p0m, p1m, p2m, p3m, 0.0, tnorm, &wa, &wb);
				if (dist < delta) {
					*t = ts[i];
					if (w0) {
						*w0 = wa;
						*w1 = wb;
					}
					*norm = tnorm;
					result = true;
				}
			}
		}
#if 0
		{
			if (isDetected_CCD(p0, p1, p3, q0, q1, q3, thickness, t, &wa)) {
				if (isFV) {
					*w0 = 1.0 - wa;
					*w1 = wa;
				}
				else {
					*w0 = wa;
					*w1 = 0.0;
				}
				result = true;
			}
			if (isFV) {
				if (isDetected_CCD(p1, p2, p3, q1, q2, q3, thickness, t, &wa)) {
					*w0 = 0.0;
					*w1 = 1.0 - wa;
					result = true;
				}
				if (isDetected_CCD(p2, p0, p3, q2, q0, q3, thickness, t, &wa)) {
					*w0 = wa;
					*w1 = 0.0;
					result = true;
				}
			}
			else {
				if (isDetected_CCD(p0, p1, p2, q0, q1, q2, thickness, t, &wa)) {
					*w0 = wa;
					*w1 = 1.0;
					result = true;
				}
				if (isDetected_CCD(p2, p3, p0, q2, q3, q0, thickness, t, &wa)) {
					*w0 = 0.0;
					*w1 = wa;
					result = true;
				}
				if (isDetected_CCD(p2, p3, p1, q2, q3, q1, thickness, t, &wa)) {
					*w0 = 1.0;
					*w1 = wa;
					result = true;
				}
			}
		}
		{
			if (isDetected_CCD(p0, p3, q0, q3, thickness, t)) {
				if (isFV) {
					*w0 = 1.0;
					*w1 = 0.0;
				}
				else {
					*w0 = 0.0;
					*w1 = 1.0;
				}
				result = true;
			}
			if (isDetected_CCD(p1, p3, q1, q3, thickness, t)) {
				if (isFV)	*w0 = 0.0;
				else		*w0 = 1.0;
				*w1 = 1.0;
				result = true;
			}
			if (isFV) {
				if (isDetected_CCD(p2, p3, q2, q3, thickness, t)) {
					*w0 = 0.0;
					*w1 = 0.0;
					result = true;
				}
			}
			else {
				if (isDetected_CCD(p0, p2, q0, q2, thickness, t)) {
					*w0 = 0.0;
					*w1 = 0.0;
					result = true;
				}
				if (isDetected_CCD(p1, p2, q1, q2, thickness, t)) {
					*w0 = 1.0;
					*w1 = 0.0;
					result = true;
				}
			}
		}
#endif
		if (*t > 1.0) {
			delta = thickness * COL_THICKNESS_RATIO;
			dist = getDistanceElements(isFV, q0, q1, q2, q3, 0.0, tnorm, &wa, &wb);
			if (dist < delta) {
				*t = 1.0;
				if (w0) {
					*w0 = wa;
					*w1 = wb;
				}
				*norm = tnorm;
				result = true;
			}
		}
	}
	return result;
}
inline __device__ void getCCDTime_device(
	bool isFV,
	const REAL3& p0, const REAL3& p1, const REAL3& p2, const REAL3& p3,
	const REAL3& q0, const REAL3& q1, const REAL3& q2, const REAL3& q3,
	REAL thickness, REAL* t)
{
#if 0
	REAL minTIme = 0.0;
	REAL maxTIme = 0.98;
	REAL delta = thickness * COL_CCD_THICKNESS_DETECT;
	//REAL colThickness = thickness * COL_THICKNESS_RATIO;
	REAL colThickness = delta * 5.0;
	if (Culling_CCD(isFV, p0, p1, p2, p3, q0, q1, q2, q3, delta)) {
		REAL ts[3];
		REAL wa, wb, dist, ratio;
		REAL3 p0m, p1m, p2m, p3m, tmp;
		uint num = FinePlaneCoTime(p0, p1, p2, p3, q0, q1, q2, q3, ts);
		for (uint i = 0; i < num; i++) {
			if (*t > ts[i] * minTIme) {
				p0m = p0 + (q0 - p0) * ts[i];
				p1m = p1 + (q1 - p1) * ts[i];
				p2m = p2 + (q2 - p2) * ts[i];
				p3m = p3 + (q3 - p3) * ts[i];
				dist = getDistanceElements(isFV, p0m, p1m, p2m, p3m, 0.0, &wa, &wb);
				if (dist < delta) {
					p0m = q0 - p0;
					p1m = q1 - p1;
					p2m = q2 - p2;
					p3m = q3 - p3;
					if (isFV)
						tmp = p0m * wa + p1m * wb + p2m * (1.0 - wa - wb) - p3m;
					else
						tmp = p0m + (p1m - p0m) * wa - p2m - (p3m - p2m) * wb;
					dist = Length(tmp);
					ratio = minTIme;
					if (dist > 0.0) {
						ratio = 1.0 - colThickness / (dist * ts[i]);
						if (ratio < minTIme)		ratio = minTIme;
						else if (ratio > maxTIme)	ratio = maxTIme;
					}
					ts[i] *= ratio;
					if (*t > ts[i])
						*t = ts[i];
				}
			}
		}

		if (*t > minTIme) {
			dist = getDistanceElements(isFV, q0, q1, q2, q3, 0.0, &wa, &wb);
			if (dist < delta) {
				p0m = q0 - p0;
				p1m = q1 - p1;
				p2m = q2 - p2;
				p3m = q3 - p3;
				if (isFV)
					tmp = p0m * wa + p1m * wb + p2m * (1.0 - wa - wb) - p3m;
				else
					tmp = p0m + (p1m - p0m) * wa - p2m - (p3m - p2m) * wb;
				dist = Length(tmp);
				ratio = minTIme;
				if (dist > 0.0) {
					ratio = 1.0 - colThickness / dist;
					if (ratio < minTIme)		ratio = minTIme;
					else if (ratio > maxTIme)	ratio = maxTIme;
				}
				if (*t > ratio)
					*t = ratio;
			}
		}
	}
#else
	REAL delta = thickness * COL_CCD_THICKNESS_DETECT;
	bool result = false;
	if (Culling_CCD(isFV, p0, p1, p2, p3, q0, q1, q2, q3, delta)) {
		REAL ts[3];
		REAL dist, ratio;
		REAL3 p0m, p1m, p2m, p3m, tmp;
		uint num = FinePlaneCoTime(p0, p1, p2, p3, q0, q1, q2, q3, ts);
		for (uint i = 0; i < num; i++) {
			if (ts[i] < *t) {
				p0m = p0 + (q0 - p0) * ts[i];
				p1m = p1 + (q1 - p1) * ts[i];
				p2m = p2 + (q2 - p2) * ts[i];
				p3m = p3 + (q3 - p3) * ts[i];
				dist = getDistanceElements(isFV, p0m, p1m, p2m, p3m, 0.0);
				if (dist < delta) {
					*t = ts[i];
					result = true;
				}
			}
		}
#if 0
		{
			REAL wa, wb;
			if (isDetected_CCD(p0, p1, p3, q0, q1, q3, thickness, t, &wa)) {
			}
			if (isFV) {
				if (isDetected_CCD(p1, p2, p3, q1, q2, q3, thickness, t, &wa)) {
				}
				if (isDetected_CCD(p2, p0, p3, q2, q0, q3, thickness, t, &wa)) {
				}
			}
			else {
				if (isDetected_CCD(p0, p1, p2, q0, q1, q2, thickness, t, &wa)) {
				}
				if (isDetected_CCD(p2, p3, p0, q2, q3, q0, thickness, t, &wa)) {
				}
				if (isDetected_CCD(p2, p3, p1, q2, q3, q1, thickness, t, &wa)) {
				}
			}
		}
		{
			if (isDetected_CCD(p0, p3, q0, q3, thickness, t)) {
			}
			if (isDetected_CCD(p1, p3, q1, q3, thickness, t)) {
			}
			if (isFV) {
				if (isDetected_CCD(p2, p3, q2, q3, thickness, t)) {
				}
			}
			else {
				if (isDetected_CCD(p0, p2, q0, q2, thickness, t)) {
				}
				if (isDetected_CCD(p1, p2, q1, q2, thickness, t)) {
				}
			}
		}
#endif

		if (*t > 1.0) {
			REAL currDist = getDistanceElements(isFV, p0, p1, p2, p3, thickness);
			delta = min(thickness * COL_THICKNESS_RATIO, currDist) * 0.5;
			dist = getDistanceElements(isFV, q0, q1, q2, q3, delta);
			if (dist < delta)
				*t = 1.0;
		}
	}
#endif
}
//-------------------------------------------------------------------------
__device__ void numSelfContactElements_device(
	const uint lRTri, const uint rRTri,
	const uint* ino, const REAL3* pi,
	const uint* jno, const REAL3* pj,
	REAL thickness, uint& num)
{
	uint i, j, i1, j1;
	for (i = 0u; i < 3u; i++) {
		if (RTriVertex(rRTri, i)) {
			if (Culling_Index(true, ino[0], ino[1], ino[2], jno[i]))
				if (Culling_Proximity(true, pi[0], pi[1], pi[2], pj[i], thickness))
				//if (isDetected_Proximity(true, pi[0], pi[1], pi[2], pj[i], thickness, &w0, &w1))
					//if (Culling_barycentricRTri(w0, w1, lRTri))
						num++;
		}
		if (RTriVertex(lRTri, i)) {
			if (Culling_Index(true, jno[0], jno[1], jno[2], ino[i]))
				if (Culling_Proximity(true, pj[0], pj[1], pj[2], pi[i], thickness))
				//if (isDetected_Proximity(true, pj[0], pj[1], pj[2], pi[i], thickness, &w0, &w1))
					//if (Culling_barycentricRTri(w0, w1, rRTri))
					num++;
		}

		i1 = (i + 1u) % 3u;
		if (RTriEdge(lRTri, i)) {
			for (j = 0u; j < 3u; j++) {
				j1 = (j + 1u) % 3u;
				if (RTriEdge(rRTri, j)) {
					if (Culling_Index(false, ino[i], ino[i1], jno[j], jno[j1]))
						if (Culling_Proximity(false, pi[i], pi[i1], pj[j], pj[j1], thickness))
						//if (isDetected_Proximity(false, pi[i], pi[i1], pj[j], pj[j1], thickness, &w0, &w1))
							//if (Culling_barycentricRTri(w0, w1, lRTri, rRTri, i, i1, j, j1))
							num++;
				}
			}
		}
	}
}
__device__ void numObstacleContactElements_device(
	const uint lRTri, const uint rRTri,
	const uint* ino, const REAL3* pi,
	const uint* jno, const REAL3* pj,
	REAL thickness, uint& num)
{
	uint i, j, i1, j1;
	for (i = 0u; i < 3u; i++) {
		if (RTriVertex(rRTri, i)) {
			if (Culling_Proximity(true, pi[0], pi[1], pi[2], pj[i], thickness))
			//if (isDetected_Proximity(true, pi[0], pi[1], pi[2], pj[i], thickness, &w0, &w1))
				//if (Culling_barycentricRTri(w0, w1, lRTri))
					num++;
		}
		if (RTriVertex(lRTri, i)) {
			if (Culling_Proximity(true, pj[0], pj[1], pj[2], pi[i], thickness))
			//if (isDetected_Proximity(true, pj[0], pj[1], pj[2], pi[i], thickness, &w0, &w1))
				//if (Culling_barycentricRTri(w0, w1, rRTri))
					num++;
		}

		i1 = (i + 1u) % 3u;
		if (RTriEdge(lRTri, i)) {
			for (j = 0u; j < 3u; j++) {
				j1 = (j + 1u) % 3u;
				if (RTriEdge(rRTri, j)) {
					if (Culling_Proximity(false, pi[i], pi[i1], pj[j], pj[j1], thickness))
					//if (isDetected_Proximity(false, pi[i], pi[i1], pj[j], pj[j1], thickness, &w0, &w1))
						//if (Culling_barycentricRTri(w0, w1, lRTri, rRTri, i, i1, j, j1))
							num++;
				}
			}
		}
	}
}
__device__ void getSelfContactElements_device(
	const uint lRTri, const uint rRTri,
	const uint* ino, const REAL3* pi,
	const uint* jno, const REAL3* pj,
	REAL thickness, ContactElem* ces, uint& num)
{
	uint i, j, i1, j1;
	REAL3 norm;
	REAL w0, w1;
	num = 0u;
	for (i = 0u; i < 3u; i++) {
		if (RTriVertex(rRTri, i)) {
			if (Culling_Index(true, ino[0], ino[1], ino[2], jno[i]))
				if (isDetected_Proximity(true, pi[0], pi[1], pi[2], pj[i], thickness, norm, &w0, &w1)) 
					if(Culling_barycentricRTri(w0, w1, lRTri))
						makeSelfCE(ces[num++], true, 0.0, 0u, norm, w0, w1, ino[0], ino[1], ino[2], jno[i]);
		}
		if (RTriVertex(lRTri, i)) {
			if (Culling_Index(true, jno[0], jno[1], jno[2], ino[i]))
				if (isDetected_Proximity(true, pj[0], pj[1], pj[2], pi[i], thickness, norm, &w0, &w1))
					if (Culling_barycentricRTri(w0, w1, rRTri))
						makeSelfCE(ces[num++], true, 0.0, 0u, norm, w0, w1, jno[0], jno[1], jno[2], ino[i]);
		}

		i1 = (i + 1u) % 3u;
		if (RTriEdge(lRTri, i)) {
			for (j = 0u; j < 3u; j++) {
				j1 = (j + 1u) % 3u;
				if (RTriEdge(rRTri, j)) {
					if (Culling_Index(false, ino[i], ino[i1], jno[j], jno[j1]))
						if (isDetected_Proximity(false, pi[i], pi[i1], pj[j], pj[j1], thickness, norm, &w0, &w1))
							if (Culling_barycentricRTri(w0, w1, lRTri, rRTri, i, i1, j, j1))
								makeSelfCE(ces[num++], false, 0.0, 0u, norm, w0, w1, ino[i], ino[i1], jno[j], jno[j1]);
				}
			}
		}
	}
}
__device__ void getObstacleContactElements_device(
	const uint lRTri, const uint rRTri,
	const uint* ino, const REAL3* pi,
	const uint* jno, const REAL3* pj,
	REAL thickness, ContactElem* ces, uint& num)
{
	uint i, j, i1, j1;
	REAL3 norm;
	REAL w0, w1;
	num = 0u;
	for (i = 0u; i < 3u; i++) {
		if (RTriVertex(rRTri, i)) {
			if (isDetected_Proximity(true, pi[0], pi[1], pi[2], pj[i], thickness, norm, &w0, &w1))
				if (Culling_barycentricRTri(w0, w1, lRTri))
					makeSelfCE(ces[num++], true, 0.0, 1u, norm, w0, w1, ino[0], ino[1], ino[2], jno[i]);
		}
		if (RTriVertex(lRTri, i)) {
			if (isDetected_Proximity(true, pj[0], pj[1], pj[2], pi[i], thickness, norm, &w0, &w1))
				if (Culling_barycentricRTri(w0, w1, rRTri))
					makeSelfCE(ces[num++], true, 0.0, 2u, norm, w0, w1, jno[0], jno[1], jno[2], ino[i]);
		}


		i1 = (i + 1u) % 3u;
		if (RTriEdge(lRTri, i)) {
			for (j = 0u; j < 3u; j++) {
				j1 = (j + 1u) % 3u;
				if (RTriEdge(rRTri, j)) {
					if (isDetected_Proximity(false, pi[i], pi[i1], pj[j], pj[j1], thickness, norm, &w0, &w1))
						if (Culling_barycentricRTri(w0, w1, lRTri, rRTri, i, i1, j, j1))
							makeSelfCE(ces[num++], false, 0.0, 1u, norm, w0, w1, ino[i], ino[i1], jno[j], jno[j1]);
				}
			}
		}
	}
}
//-------------------------------------------------------------------------
__global__ void getNumLastBvtts_kernel(
	BVHParam clothBvh,
	uint* lastBvttIds, uint lastBvhSize)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= lastBvhSize)
		return;
	if (id == 0u)
		lastBvttIds[0] = 0u;

	BVHNode node, comp;
	uint icomp, ino;

	uint path, level, lastPath;

	bool loop = true;
	bool goback = false;

	path = id;
	level = clothBvh._maxLevel - 1u;
	ino = getBVHIndex(path, level);
	getBVHNode(node, clothBvh, ino);
	{
		uint half = lastBvhSize >> 1u;
		lastPath = (id + half);
		if (id >= half)
			lastPath--;
		if (lastPath >= lastBvhSize) {
			goback = true;
			lastPath -= lastBvhSize;
		}
	}

	if ((path & 1u) == 0u)
		path |= 1u;
	else {
		if (path + 1u >> level) {
			if (goback) {
				path = 0u;
				level = 1u;
				goback = false;
			}
			else path = 0xffffffff;
		}
		else {
			do {
				level--;
				path >>= 1u;
			} while (path & 1u);
			path |= 1u;
		}
	}
	if (!goback && path > (lastPath >> clothBvh._maxLevel - 1u - level))
		loop = false;

	uint num = 0u;
	while (loop) {
		icomp = getBVHIndex(path, level);
		getBVHAABB(comp._aabb, clothBvh, icomp);
		bool isIntersect = intersect(node._aabb, comp._aabb);
		if (isIntersect && level < clothBvh._maxLevel - 1u) {
			level++;
			path <<= 1u;
		}
		else {
			if (isIntersect)
				num++;

			if ((path & 1u) == 0u)
				path |= 1u;
			else {
				if (path + 1u >> level) {
					if (goback) {
						path = 0u;
						level = 1u;
						goback = false;
					}
					else path = 0xffffffff;
				}
				else {
					do {
						level--;
						path >>= 1u;
					} while (path & 1u);
					path |= 1u;
				}
			}
			if (!goback && path > (lastPath >> clothBvh._maxLevel - 1u - level))
				loop = false;
		}
	}

	lastBvttIds[id + 1u] = num;
}
__global__ void getNumLastBvtts_kernel(
	BVHParam clothBvh, BVHParam obsBvh,
	uint* lastBvttIds, uint lastBvhSize)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= lastBvhSize)
		return;
	if (id == 0u)
		lastBvttIds[0] = 0u;

	BVHNode node, comp;
	uint icomp, ino;

	uint path, level, lastPath;

	bool loop = true;

	path = level = 0u;

	ino = getBVHIndex(id, clothBvh._maxLevel - 1u);
	getBVHNode(node, clothBvh, ino);

	uint num = 0u;
	while (loop) {
		icomp = getBVHIndex(path, level);
		getBVHAABB(comp._aabb, obsBvh, icomp);
		bool isIntersect = intersect(node._aabb, comp._aabb);
		if (isIntersect && level < obsBvh._maxLevel - 1u) {
			level++;
			path <<= 1u;
		}
		else {
			if (isIntersect)
				num++;

			if ((path & 1u) == 0u)
				path |= 1u;
			else {
				if (path + 1u >> level)
					loop = false;
				else {
					do {
						level--;
						path >>= 1u;
					} while (path & 1u);
					path |= 1u;
				}
			}
		}
	}

	lastBvttIds[id + 1u] = num;
}
__global__ void getLastBvtts_kernel(
	BVHParam clothBvh,
	uint2* lastBvtts, uint* lastBvttIds, uint lastBvhSize)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= lastBvhSize)
		return;

	BVHNode node, comp;
	uint icomp, ino;

	uint path, level, lastPath;

	bool loop = true;
	bool goback = false;

	path = id;
	level = clothBvh._maxLevel - 1u;
	ino = getBVHIndex(path, level);
	getBVHNode(node, clothBvh, ino);
	{
		uint half = lastBvhSize >> 1u;
		lastPath = (id + half);
		if (id >= half)
			lastPath--;
		if (lastPath >= lastBvhSize) {
			goback = true;
			lastPath -= lastBvhSize;
		}
	}

	if ((path & 1u) == 0u)
		path |= 1u;
	else {
		if (path + 1u >> level) {
			if (goback) {
				path = 0u;
				level = 1u;
				goback = false;
			}
			else path = 0xffffffff;
		}
		else {
			do {
				level--;
				path >>= 1u;
			} while (path & 1u);
			path |= 1u;
		}
	}
	if (!goback && path > (lastPath >> clothBvh._maxLevel - 1u - level))
		loop = false;

	uint2 bvtt;
	bvtt.x = ino;
	ino = lastBvttIds[id];
	while (loop) {
		icomp = getBVHIndex(path, level);
		getBVHAABB(comp._aabb, clothBvh, icomp);
		bool isIntersect = intersect(node._aabb, comp._aabb);
		if (isIntersect && level < clothBvh._maxLevel - 1u) {
			level++;
			path <<= 1u;
		}
		else {
			if (isIntersect) {
				bvtt.y = icomp;
				lastBvtts[ino++] = bvtt;
			}

			if ((path & 1u) == 0u)
				path |= 1u;
			else {
				if (path + 1u >> level) {
					if (goback) {
						path = 0u;
						level = 1u;
						goback = false;
					}
					else path = 0xffffffff;
				}
				else {
					do {
						level--;
						path >>= 1u;
					} while (path & 1u);
					path |= 1u;
				}
			}
			if (!goback && path > (lastPath >> clothBvh._maxLevel - 1u - level))
				loop = false;
		}
	}
}
__global__ void getLastBvtts_kernel(
	BVHParam clothBvh, BVHParam obsBvh,
	uint2* lastBvtts, uint* lastBvttIds, uint lastBvhSize)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= lastBvhSize)
		return;

	BVHNode node, comp;
	uint icomp, ino;

	uint path, level, lastPath;

	bool loop = true;

	path = level = 0u;

	ino = getBVHIndex(id, clothBvh._maxLevel - 1u);
	getBVHNode(node, clothBvh, ino);

	uint2 bvtt;
	bvtt.x = ino;
	ino = lastBvttIds[id];
	while (loop) {
		icomp = getBVHIndex(path, level);
		getBVHAABB(comp._aabb, obsBvh, icomp);
		bool isIntersect = intersect(node._aabb, comp._aabb);
		if (isIntersect && level < obsBvh._maxLevel - 1u) {
			level++;
			path <<= 1u;
		}
		else {
			if (isIntersect) {
				bvtt.y = icomp;
				lastBvtts[ino++] = bvtt;
			}

			if ((path & 1u) == 0u)
				path |= 1u;
			else {
				if (path + 1u >> level)
					loop = false;
				else {
					do {
						level--;
						path >>= 1u;
					} while (path & 1u);
					path |= 1u;
				}
			}
		}
	}
}
//-------------------------------------------------------------------------
__global__ void getNumSelfContactElements_kernel(
	ContactElemParam ceParam,
	ObjParam clothParam, BVHParam clothBvh, RTriParam clothRTri,
	const REAL thickness)
{
	__shared__ uint s_bvtts[2][BVTT_SHARED_SIZE];
	__shared__ uint s_bvttNum;
	__shared__ uint s_endNum;
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	BVHNode node, comp;
	uint icomp, ino;
	if (threadIdx.x == 0u) {
		ino = blockDim.x * (blockIdx.x + 1u);
		if (ino > clothParam._numFaces)
			s_endNum = ino - clothParam._numFaces;
		else s_endNum = 0u;
		s_bvttNum = 0u;
	}
	__syncthreads();

	uint ileaf = clothBvh._size - clothParam._numFaces;
	uint path, level, lastPath;

	bool isLoop = true;
	bool goback = false;
	bool isIntersected;

	if (id >= clothParam._numFaces)
		isLoop = false;
	else {
		path = id;
		level = clothBvh._maxLevel;
		if (id >= clothBvh._pivot) {
			path = id - clothBvh._pivot + (clothBvh._pivot >> 1u);
			level--;
		}
		ino = getBVHIndex(path, level);
		getBVHNode(node, clothBvh, ino);
		{
			uint half = clothBvh._numFaces >> 1u;
			lastPath = (id + half);
			if (!(clothBvh._numFaces & 1u) && id >= half)
				lastPath--;
			if (lastPath >= clothBvh._numFaces) {
				goback = true;
				lastPath -= clothBvh._numFaces;
			}

			if (lastPath >= clothBvh._pivot)
				lastPath += lastPath - clothBvh._pivot;
		}

		if ((path & 1u) == 0u)
			path |= 1u;
		else {
			if (path + 1u >> level) {
				if (goback) {
					path = 0u;
					level = 1u;
					goback = false;
				}
				else path = 0xffffffff;
			}
			else {
				do {
					level--;
					path >>= 1u;
				} while (path & 1u);
				path |= 1u;
			}
		}
		if (!goback && path > (lastPath >> clothBvh._maxLevel - level)) {
			atomicAdd(&s_endNum, 1u);
			isLoop = false;
		}
	}

	uint lRTri, rRTri;
	uint inos[3], jnos[3];
	REAL3 pis[3], pjs[3];
	uint num = 0u;

	do {
		__syncthreads();
		if (isLoop) {
			icomp = getBVHIndex(path, level);
			getBVHAABB(comp._aabb, clothBvh, icomp);
			isIntersected = intersect(node._aabb, comp._aabb);
			if (isIntersected && icomp < ileaf) {
				level++;
				path <<= 1u;
			}
			else {
				if (isIntersected) {
					ino = atomicAdd(&s_bvttNum, 1u);
					s_bvtts[0][ino] = node._face;
					s_bvtts[1][ino] = clothBvh._faces[icomp - ileaf];
				}

				if ((path & 1u) == 0u)
					path |= 1u;
				else {
					if (path + 1u >> level) {
						if (goback) {
							path = 0u;
							level = 1u;
							goback = false;
						}
						else path = 0xffffffff;
					}
					else {
						do {
							level--;
							path >>= 1u;
						} while (path & 1u);
						path |= 1u;
					}
				}
				if (!goback && path > (lastPath >> clothBvh._maxLevel - level)) {
					atomicAdd(&s_endNum, 1u);
					isLoop = false;
				}
			}
		}
		__syncthreads();
		if (s_endNum >= blockDim.x || s_bvttNum >= BVTT_SHARED_SIZE - blockDim.x) {
			for (ino = threadIdx.x; ino < s_bvttNum; ino += blockDim.x) {
				lRTri = clothRTri._info[s_bvtts[0][ino]];
				rRTri = clothRTri._info[s_bvtts[1][ino]];
				getMeshElements_device(clothParam, s_bvtts[0][ino], s_bvtts[1][ino], inos, jnos, pis, pjs);
				numSelfContactElements_device(
					lRTri, rRTri,
					inos, pis,
					jnos, pjs,
					thickness, num);
			}
			__syncthreads();
			if (threadIdx.x == 0u)
				s_bvttNum = 0u;
		}
	} while (s_endNum < blockDim.x);

	s_bvtts[0][threadIdx.x] = num;
	for (ino = blockDim.x >> 1u; ino > 32u; ino >>= 1u) {
		__syncthreads();
		if (threadIdx.x < ino)
			s_bvtts[0][threadIdx.x] += s_bvtts[0][threadIdx.x + ino];
	}
	__syncthreads();
	if (threadIdx.x < 32u) {
		warpAdd(s_bvtts[0], threadIdx.x);
		if (threadIdx.x == 0u)
			atomicAdd(ceParam.d_tmp, s_bvtts[0][0]);
	}
}
__global__ void getNumObstacleContactElements_kernel(
	ContactElemParam ceParam,
	ObjParam clothParam, BVHParam clothBvh, RTriParam clothRTri,
	ObjParam obsParam, BVHParam obsBvh, RTriParam obsRTri,
	const REAL thickness)
{
	__shared__ uint s_bvtts[2][BVTT_SHARED_SIZE];
	__shared__ uint s_bvttNum;
	__shared__ uint s_endNum;

	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	BVHNode node, comp;
	uint icomp, ino;

	if (threadIdx.x == 0u) {
		ino = blockDim.x * (blockIdx.x + 1u);
		if (ino > clothParam._numFaces)
			s_endNum = ino - clothParam._numFaces;
		else s_endNum = 0u;
		s_bvttNum = 0u;
	}

	uint clothileaf = clothBvh._size - clothParam._numFaces;
	uint obsileaf = obsBvh._size - obsParam._numFaces;

	uint path = 0;
	uint level = 0;

	bool isLoop = true, isIntersected;

	if (id >= clothParam._numFaces)
		isLoop = false;
	else {
		ino = clothileaf + id;
		getBVHNode(node, clothBvh, ino);
	}

	uint lRTri, rRTri;
	uint inos[3], jnos[3];
	REAL3 pis[3], pjs[3];
	uint num = 0u;

	do {
		__syncthreads();
		if (isLoop) {
			icomp = getBVHIndex(path, level);
			getBVHAABB(comp._aabb, obsBvh, icomp);
			isIntersected = intersect(node._aabb, comp._aabb);
			if (isIntersected && icomp < obsileaf) {
				level++;
				path <<= 1u;
			}
			else {
				if (isIntersected) {
					comp._face = obsBvh._faces[icomp - obsileaf];
					ino = atomicAdd(&s_bvttNum, 1u);
					s_bvtts[0][ino] = node._face;
					s_bvtts[1][ino] = comp._face;
				}

				if ((path & 1u) == 0u)
					path |= 1u;
				else {
					if (path + 1u >> level) {
						isLoop = false;
						atomicAdd(&s_endNum, 1u);
					}
					else {
						do {
							level--;
							path >>= 1u;
						} while (path & 1u);
						path |= 1u;
					}
				}
			}
		}
		__syncthreads();
		if (s_endNum >= blockDim.x || s_bvttNum >= BVTT_SHARED_SIZE - blockDim.x) {
			for (ino = threadIdx.x; ino < s_bvttNum; ino += blockDim.x) {
				lRTri = clothRTri._info[s_bvtts[0][ino]];
				rRTri = obsRTri._info[s_bvtts[1][ino]];
				getMeshElements_device(clothParam, obsParam, s_bvtts[0][ino], s_bvtts[1][ino], inos, jnos, pis, pjs);
				numObstacleContactElements_device(
					lRTri, rRTri,
					inos, pis,
					jnos, pjs,
					thickness, num);
			}
			__syncthreads();
			if (threadIdx.x == 0u)
				s_bvttNum = 0u;
		}
	} while (s_endNum < blockDim.x);

	s_bvtts[0][threadIdx.x] = num;
	for (ino = blockDim.x >> 1u; ino > 32u; ino >>= 1u) {
		__syncthreads();
		if (threadIdx.x < ino)
			s_bvtts[0][threadIdx.x] += s_bvtts[0][threadIdx.x + ino];
	}
	__syncthreads();
	if (threadIdx.x < 32u) {
		warpAdd(s_bvtts[0], threadIdx.x);
		if (threadIdx.x == 0u)
			atomicAdd(ceParam.d_tmp, s_bvtts[0][0]);
	}
}
__global__ void getSelfContactElements_kernel(
	ContactElemParam ceParam,
	ObjParam clothParam, BVHParam clothBvh, RTriParam clothRTri,
	const REAL thickness)
{
	__shared__ uint s_bvtts[2][BVTT_SHARED_SIZE];
	__shared__ uint s_bvttNum;
	__shared__ uint s_endNum;
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	BVHNode node, comp;
	uint icomp, ino;
	if (threadIdx.x == 0u) {
		ino = blockDim.x * (blockIdx.x + 1u);
		if (ino > clothParam._numFaces)
			s_endNum = ino - clothParam._numFaces;
		else s_endNum = 0u;
		s_bvttNum = 0u;
	}
	__syncthreads();

	uint ileaf = clothBvh._size - clothParam._numFaces;
	uint path, level, lastPath;

	bool isLoop = true;
	bool goback = false;
	bool isIntersected;

	if (id >= clothParam._numFaces)
		isLoop = false;
	else {
		path = id;
		level = clothBvh._maxLevel;
		if (id >= clothBvh._pivot) {
			path = id - clothBvh._pivot + (clothBvh._pivot >> 1u);
			level--;
		}
		ino = getBVHIndex(path, level);
		getBVHNode(node, clothBvh, ino);
		{
			uint half = clothBvh._numFaces >> 1u;
			lastPath = (id + half);
			if (!(clothBvh._numFaces & 1u) && id >= half)
				lastPath--;
			if (lastPath >= clothBvh._numFaces) {
				goback = true;
				lastPath -= clothBvh._numFaces;
			}

			if (lastPath >= clothBvh._pivot)
				lastPath += lastPath - clothBvh._pivot;
		}

		if ((path & 1u) == 0u)
			path |= 1u;
		else {
			if (path + 1u >> level) {
				if (goback) {
					path = 0u;
					level = 1u;
					goback = false;
				}
				else path = 0xffffffff;
			}
			else {
				do {
					level--;
					path >>= 1u;
				} while (path & 1u);
				path |= 1u;
			}
		}
		if (!goback && path > (lastPath >> clothBvh._maxLevel - level)) {
			atomicAdd(&s_endNum, 1u);
			isLoop = false;
		}
	}

	uint lRTri, rRTri;
	uint inos[3], jnos[3];
	REAL3 pis[3], pjs[3];

	ContactElem ces[15];
	uint ceSize;

	do {
		__syncthreads();
		if (isLoop) {
			icomp = getBVHIndex(path, level);
			getBVHAABB(comp._aabb, clothBvh, icomp);
			isIntersected = intersect(node._aabb, comp._aabb);
			if (isIntersected && icomp < ileaf) {
				level++;
				path <<= 1u;
			}
			else {
				if (isIntersected) {
					ino = atomicAdd(&s_bvttNum, 1u);
					s_bvtts[0][ino] = node._face;
					s_bvtts[1][ino] = clothBvh._faces[icomp - ileaf];
				}

				if ((path & 1u) == 0u)
					path |= 1u;
				else {
					if (path + 1u >> level) {
						if (goback) {
							path = 0u;
							level = 1u;
							goback = false;
						}
						else path = 0xffffffff;
					}
					else {
						do {
							level--;
							path >>= 1u;
						} while (path & 1u);
						path |= 1u;
					}
				}
				if (!goback && path > (lastPath >> clothBvh._maxLevel - level)) {
					atomicAdd(&s_endNum, 1u);
					isLoop = false;
				}
			}
		}
		__syncthreads();
		if (s_endNum >= blockDim.x || s_bvttNum >= BVTT_SHARED_SIZE - blockDim.x) {
			for (ino = threadIdx.x; ino < s_bvttNum; ino += blockDim.x) {
				lRTri = clothRTri._info[s_bvtts[0][ino]];
				rRTri = clothRTri._info[s_bvtts[1][ino]];
				getMeshElements_device(clothParam, s_bvtts[0][ino], s_bvtts[1][ino], inos, jnos, pis, pjs);

				getSelfContactElements_device(
					lRTri, rRTri,
					inos, pis,
					jnos, pjs,
					thickness, ces, ceSize);
				if (ceSize) {
					addCE(ceParam, ces, ceSize);
#ifdef CHECK_DETECTION
					clothBvh._isDetecteds[s_bvtts[0][ino]] = 1u;
					clothBvh._isDetecteds[s_bvtts[1][ino]] = 1u;
#endif
				}
			}
			__syncthreads();
			if (threadIdx.x == 0u)
				s_bvttNum = 0u;
		}
	} while (s_endNum < blockDim.x);
}
__global__ void getObstacleContactElements_kernel(
	ContactElemParam ceParam,
	ObjParam clothParam, BVHParam clothBvh, RTriParam clothRTri,
	ObjParam obsParam, BVHParam obsBvh, RTriParam obsRTri,
	const REAL thickness)
{
	__shared__ uint s_bvtts[2][BVTT_SHARED_SIZE];
	__shared__ uint s_bvttNum;
	__shared__ uint s_endNum;

	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	BVHNode node, comp;
	uint RTriA, RTriB, icomp, ino;

	if (threadIdx.x == 0u) {
		ino = blockDim.x * (blockIdx.x + 1u);
		if (ino > clothParam._numFaces)
			s_endNum = ino - clothParam._numFaces;
		else s_endNum = 0u;
		s_bvttNum = 0u;
	}

	uint clothileaf = clothBvh._size - clothParam._numFaces;
	uint obsileaf = obsBvh._size - obsParam._numFaces;

	uint path = 0;
	uint level = 0;

	bool isLoop = true, isIntersected;

	if (id >= clothParam._numFaces)
		isLoop = false;
	else {
		ino = clothileaf + id;

		getBVHNode(node, clothBvh, ino);
	}

	uint lRTri, rRTri;
	uint inos[3], jnos[3];
	REAL3 pis[3], pjs[3];

	ContactElem ces[15];
	uint ceSize;

	do {
		__syncthreads();
		if (isLoop) {
			icomp = getBVHIndex(path, level);
			getBVHAABB(comp._aabb, obsBvh, icomp);
			isIntersected = intersect(node._aabb, comp._aabb);
			if (isIntersected && icomp < obsileaf) {
				level++;
				path <<= 1u;
			}
			else {
				if (isIntersected) {
					comp._face = obsBvh._faces[icomp - obsileaf];
					ino = atomicAdd(&s_bvttNum, 1u);
					s_bvtts[0][ino] = node._face;
					s_bvtts[1][ino] = comp._face;
				}

				if ((path & 1u) == 0u)
					path |= 1u;
				else {
					if (path + 1u >> level) {
						isLoop = false;
						atomicAdd(&s_endNum, 1u);
					}
					else {
						do {
							level--;
							path >>= 1u;
						} while (path & 1u);
						path |= 1u;
					}
				}
			}
		}
		__syncthreads();
		if (s_endNum >= blockDim.x || s_bvttNum >= BVTT_SHARED_SIZE - blockDim.x) {
			for (ino = threadIdx.x; ino < s_bvttNum; ino += blockDim.x) {
				lRTri = clothRTri._info[s_bvtts[0][ino]];
				rRTri = obsRTri._info[s_bvtts[1][ino]];
				getMeshElements_device(clothParam, obsParam, s_bvtts[0][ino], s_bvtts[1][ino], inos, jnos, pis, pjs);
				getObstacleContactElements_device(
					lRTri, rRTri,
					inos, pis,
					jnos, pjs,
					thickness, ces, ceSize);
				if (ceSize) {
					addCE(ceParam, ces, ceSize);
#ifdef CHECK_DETECTION
					clothBvh._isDetecteds[s_bvtts[0][ino]] = 1u;
					obsBvh._isDetecteds[s_bvtts[1][ino]] = 1u;
#endif
				}
			}
			__syncthreads();
			if (threadIdx.x == 0u)
				s_bvttNum = 0u;
		}
	} while (s_endNum < blockDim.x);
}
//-------------------------------------------------------------------------
__global__ void getNumSelfContactElements_LastBvtt_kernel(
	ObjParam clothParam, BVHParam clothBvh, RTriParam clothRTri,
	uint2* lastBvtts, uint lastBvttSize,
	uint* ceSize,
	const REAL thickness)
{
	extern __shared__ uint s_nums[];
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= lastBvttSize) {
		s_nums[threadIdx.x] = 0u;
		return;
	}

	uint2 bvtt = lastBvtts[id];
	uint2 ifaces[4];
	uint fnum, ino;
	getLastBvttIfaces(clothBvh, bvtt.x, bvtt.y, ifaces, fnum);

	uint lRTri, rRTri;
	uint inos[3], jnos[3];
	REAL3 pis[3], pjs[3];

	uint numCols = 0u;
	for (ino = 0u; ino < fnum; ino++) {
		lRTri = clothRTri._info[ifaces[ino].x];
		rRTri = clothRTri._info[ifaces[ino].y];
		getMeshElements_device(clothParam, ifaces[ino].x, ifaces[ino].y, inos, jnos, pis, pjs);
		numSelfContactElements_device(
			lRTri, rRTri,
			inos, pis,
			jnos, pjs,
			thickness, numCols);
	}
	s_nums[threadIdx.x] = numCols;

	for (ino = blockDim.x >> 1u; ino > 32u; ino >>= 1u) {
		__syncthreads();
		if (threadIdx.x < ino)
			s_nums[threadIdx.x] += s_nums[threadIdx.x + ino];
	}
	__syncthreads();
	if (threadIdx.x < 32u) {
		warpAdd(s_nums, threadIdx.x);
		if (threadIdx.x == 0u)
			atomicAdd(ceSize, s_nums[0]);
	}
}
__global__ void getNumObstacleContactElements_LastBvtt_kernel(
	ObjParam clothParam, BVHParam clothBvh, RTriParam clothRTri,
	ObjParam obsParam, BVHParam obsBvh, RTriParam obsRTri,
	uint2* lastBvtts, uint lastBvttSize,
	uint* ceSize,
	const REAL thickness)
{
	extern __shared__ uint s_nums[];
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= lastBvttSize) {
		s_nums[threadIdx.x] = 0u;
		return;
	}

	uint2 bvtt = lastBvtts[id];
	uint2 ifaces[4];
	uint fnum, ino;
	getLastBvttIfaces(clothBvh, obsBvh, bvtt.x, bvtt.y, ifaces, fnum);

	uint lRTri, rRTri;
	uint inos[3], jnos[3];
	REAL3 pis[3], pjs[3];

	uint numCols = 0u;
	for (ino = 0u; ino < fnum; ino++) {
		lRTri = clothRTri._info[ifaces[ino].x];
		rRTri = obsRTri._info[ifaces[ino].y];
		getMeshElements_device(clothParam, obsParam, ifaces[ino].x, ifaces[ino].y, inos, jnos, pis, pjs);
		numObstacleContactElements_device(
			lRTri, rRTri,
			inos, pis,
			jnos, pjs,
			thickness, numCols);
	}
	s_nums[threadIdx.x] = numCols;

	for (ino = blockDim.x >> 1u; ino > 32u; ino >>= 1u) {
		__syncthreads();
		if (threadIdx.x < ino)
			s_nums[threadIdx.x] += s_nums[threadIdx.x + ino];
	}
	__syncthreads();
	if (threadIdx.x < 32u) {
		warpAdd(s_nums, threadIdx.x);
		if (threadIdx.x == 0u)
			atomicAdd(ceSize, s_nums[0]);
	}
}
__global__ void getSelfContactElements_LastBvtt_kernel(
	ObjParam clothParam, BVHParam clothBvh, RTriParam clothRTri,
	uint2* lastBvtts, uint lastBvttSize,
	ContactElemParam ceParam,
	const REAL thickness)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= lastBvttSize)
		return;

	uint2 bvtt = lastBvtts[id];
	uint2 ifaces[4];
	uint fnum, ino;
	getLastBvttIfaces(clothBvh, bvtt.x, bvtt.y, ifaces, fnum);

	uint lRTri, rRTri;
	uint inos[3], jnos[3];
	REAL3 pis[3], pjs[3];

	ContactElem ces[15];
	uint ceSize;
	for (ino = 0u; ino < fnum; ino++) {
		lRTri = clothRTri._info[ifaces[ino].x];
		rRTri = clothRTri._info[ifaces[ino].y];
		getMeshElements_device(clothParam, ifaces[ino].x, ifaces[ino].y, inos, jnos, pis, pjs);
		getSelfContactElements_device(
			lRTri, rRTri,
			inos, pis,
			jnos, pjs,
			thickness, ces, ceSize);
		if (ceSize) {
			addCE(ceParam, ces, ceSize);
#ifdef CHECK_DETECTION
			clothBvh._isDetecteds[ifaces[ino].x] = 1u;
			clothBvh._isDetecteds[ifaces[ino].y] = 1u;
#endif
		}
	}
}
__global__ void getObstacleContactElements_LastBvtt_kernel(
	ObjParam clothParam, BVHParam clothBvh, RTriParam clothRTri,
	ObjParam obsParam, BVHParam obsBvh, RTriParam obsRTri,
	uint2* lastBvtts, uint lastBvttSize,
	ContactElemParam ceParam,
	const REAL thickness)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= lastBvttSize)
		return;

	uint2 bvtt = lastBvtts[id];
	uint2 ifaces[4];
	uint fnum, ino;
	getLastBvttIfaces(clothBvh, obsBvh, bvtt.x, bvtt.y, ifaces, fnum);

	uint lRTri, rRTri;
	uint inos[3], jnos[3];
	REAL3 pis[3], pjs[3];

	ContactElem ces[15];
	uint ceSize;
	for (ino = 0u; ino < fnum; ino++) {
		lRTri = clothRTri._info[ifaces[ino].x];
		rRTri = obsRTri._info[ifaces[ino].y];
		getMeshElements_device(clothParam, obsParam, ifaces[ino].x, ifaces[ino].y, inos, jnos, pis, pjs);
		getObstacleContactElements_device(
			lRTri, rRTri,
			inos, pis,
			jnos, pjs,
			thickness, ces, ceSize);
		if (ceSize) {
			addCE(ceParam, ces, ceSize);
#ifdef CHECK_DETECTION
			clothBvh._isDetecteds[ifaces[ino].x] = 1u;
			obsBvh._isDetecteds[ifaces[ino].y] = 1u;
#endif
		}
	}
}
//-------------------------------------------------------------------------
__device__ void getSelfCCDtime_device(
	const uint lRTri, const uint rRTri,
	const uint* ino, const REAL3* pi, const REAL3* qi,
	const uint* jno, const REAL3* pj, const REAL3* qj,
	REAL thickness, REAL* minTime)
{
	uint i, j, i1, j1;
	for (i = 0u; i < 3u; i++) {
		if (RTriVertex(rRTri, i)) {
			if (Culling_Index(true, ino[0], ino[1], ino[2], jno[i]))
				getCCDTime_device(true, pi[0], pi[1], pi[2], pj[i], qi[0], qi[1], qi[2], qj[i], thickness, minTime);
		}
		if (RTriVertex(lRTri, i)) {
			if (Culling_Index(true, jno[0], jno[1], jno[2], ino[i]))
				getCCDTime_device(true, pj[0], pj[1], pj[2], pi[i], qj[0], qj[1], qj[2], qi[i], thickness, minTime);
		}

		i1 = (i + 1u) % 3u;
		if (RTriEdge(lRTri, i)) {
			for (j = 0u; j < 3u; j++) {
				j1 = (j + 1u) % 3u;
				if (RTriEdge(rRTri, j)) {
					if (Culling_Index(false, ino[i], ino[i1], jno[j], jno[j1]))
						getCCDTime_device(false, pi[i], pi[i1], pj[j], pj[j1], qi[i], qi[i1], qj[j], qj[j1], thickness, minTime);
				}
			}
		}
	}
}
__device__ void getObstacleCCDtime_device(
	const uint lRTri, const uint rRTri,
	const uint* ino, const REAL3* pi, const REAL3* qi,
	const uint* jno, const REAL3* pj, const REAL3* qj,
	REAL thickness, REAL* minTime)
{
	uint i, j, i1, j1;
	for (i = 0u; i < 3u; i++) {
		if (RTriVertex(rRTri, i))
			getCCDTime_device(true, pi[0], pi[1], pi[2], pj[i], qi[0], qi[1], qi[2], qj[i], thickness, minTime);
		if (RTriVertex(lRTri, i))
			getCCDTime_device(true, pj[0], pj[1], pj[2], pi[i], qj[0], qj[1], qj[2], qi[i], thickness, minTime);

		i1 = (i + 1u) % 3u;
		if (RTriEdge(lRTri, i)) {
			for (j = 0u; j < 3u; j++) {
				j1 = (j + 1u) % 3u;
				if (RTriEdge(rRTri, j))
					getCCDTime_device(false, pi[i], pi[i1], pj[j], pj[j1], qi[i], qi[i1], qj[j], qj[j1], thickness, minTime);
			}
		}
	}
}
__global__ void getSelfCCDtime_kernel(
	ObjParam clothParam, BVHParam clothBvh, RTriParam clothRTri,
	const REAL thickness, const REAL dt,
	REAL* minTime)
{
	__shared__ uint s_bvtts[2][BVTT_SHARED_SIZE];
	__shared__ uint s_bvttNum;
	__shared__ uint s_endNum;
	__shared__ REAL s_minTimes[BLOCKSIZE];
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	BVHNode node, comp;
	uint icomp, ino;
	if (threadIdx.x == 0u) {
		ino = blockDim.x * (blockIdx.x + 1u);
		if (ino > clothParam._numFaces)
			s_endNum = ino - clothParam._numFaces;
		else s_endNum = 0u;
		s_bvttNum = 0u;
	}
	__syncthreads();

	uint ileaf = clothBvh._size - clothParam._numFaces;
	uint path, level, lastPath;

	bool isLoop = true;
	bool goback = false;
	bool isIntersected;

	if (id >= clothParam._numFaces)
		isLoop = false;
	else {
		path = id;
		level = clothBvh._maxLevel;
		if (id >= clothBvh._pivot) {
			path = id - clothBvh._pivot + (clothBvh._pivot >> 1u);
			level--;
		}
		ino = getBVHIndex(path, level);
		getBVHNode(node, clothBvh, ino);
		{
			uint half = clothBvh._numFaces >> 1u;
			lastPath = (id + half);
			if (!(clothBvh._numFaces & 1u) && id >= half)
				lastPath--;
			if (lastPath >= clothBvh._numFaces) {
				goback = true;
				lastPath -= clothBvh._numFaces;
			}

			if (lastPath >= clothBvh._pivot)
				lastPath += lastPath - clothBvh._pivot;
		}

		if ((path & 1u) == 0u)
			path |= 1u;
		else {
			if (path + 1u >> level) {
				if (goback) {
					path = 0u;
					level = 1u;
					goback = false;
				}
				else path = 0xffffffff;
			}
			else {
				do {
					level--;
					path >>= 1u;
				} while (path & 1u);
				path |= 1u;
			}
		}
		if (!goback && path > (lastPath >> clothBvh._maxLevel - level)) {
			atomicAdd(&s_endNum, 1u);
			isLoop = false;
		}
	}

	uint lRTri, rRTri;
	uint inos[3], jnos[3];
	REAL3 pis[3], pjs[3];
	REAL3 qis[3], qjs[3];
	REAL t = REAL_MAX;

	do {
		__syncthreads();
		if (isLoop) {
			icomp = getBVHIndex(path, level);
			getBVHAABB(comp._aabb, clothBvh, icomp);
			isIntersected = intersect(node._aabb, comp._aabb);
			if (isIntersected && icomp < ileaf) {
				level++;
				path <<= 1u;
			}
			else {
				if (isIntersected) {
					ino = atomicAdd(&s_bvttNum, 1u);
					s_bvtts[0][ino] = node._face;
					s_bvtts[1][ino] = clothBvh._faces[icomp - ileaf];
				}

				if ((path & 1u) == 0u)
					path |= 1u;
				else {
					if (path + 1u >> level) {
						if (goback) {
							path = 0u;
							level = 1u;
							goback = false;
						}
						else path = 0xffffffff;
					}
					else {
						do {
							level--;
							path >>= 1u;
						} while (path & 1u);
						path |= 1u;
					}
				}
				if (!goback && path > (lastPath >> clothBvh._maxLevel - level)) {
					atomicAdd(&s_endNum, 1u);
					isLoop = false;
				}
			}
		}
		__syncthreads();
		if (s_endNum >= blockDim.x || s_bvttNum >= BVTT_SHARED_SIZE - blockDim.x) {
			for (ino = threadIdx.x; ino < s_bvttNum; ino += blockDim.x) {
				lRTri = clothRTri._info[s_bvtts[0][ino]];
				rRTri = clothRTri._info[s_bvtts[1][ino]];
				getMeshElements_device(clothParam, s_bvtts[0][ino], s_bvtts[1][ino], inos, jnos, pis, pjs, qis, qjs, dt);
				getSelfCCDtime_device(
					lRTri, rRTri,
					inos, pis, qis,
					jnos, pjs, qjs,
					thickness, &t);
			}
			__syncthreads();
			if (threadIdx.x == 0u)
				s_bvttNum = 0u;
		}
	} while (s_endNum < blockDim.x);

	s_minTimes[threadIdx.x] = t;
	for (ino = blockDim.x >> 1u; ino > 32u; ino >>= 1u) {
		__syncthreads();
		if (threadIdx.x < ino)
			if (s_minTimes[threadIdx.x] > s_minTimes[threadIdx.x + ino])
				s_minTimes[threadIdx.x] = s_minTimes[threadIdx.x + ino];
	}
	__syncthreads();
	if (threadIdx.x < 32u) {
		warpMin(s_minTimes, threadIdx.x);
		if (threadIdx.x == 0u) {
			atomicMin_REAL(minTime, s_minTimes[0]);
		}
	}
}
__global__ void getObstacleCCDtime_kernel(
	ObjParam clothParam, BVHParam clothBvh, RTriParam clothRTri,
	ObjParam obsParam, BVHParam obsBvh, RTriParam obsRTri,
	const REAL thickness, const REAL dt,
	REAL* minTime)
{
	__shared__ uint s_bvtts[2][BVTT_SHARED_SIZE];
	__shared__ uint s_bvttNum;
	__shared__ uint s_endNum;
	__shared__ REAL s_minTimes[BLOCKSIZE];

	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	BVHNode node, comp;
	uint RTriA, RTriB, icomp, ino;

	if (threadIdx.x == 0u) {
		ino = blockDim.x * (blockIdx.x + 1u);
		if (ino > clothParam._numFaces)
			s_endNum = ino - clothParam._numFaces;
		else s_endNum = 0u;
		s_bvttNum = 0u;
	}

	uint clothileaf = clothBvh._size - clothParam._numFaces;
	uint obsileaf = obsBvh._size - obsParam._numFaces;

	uint path = 0;
	uint level = 0;

	bool isLoop = true, isIntersected;

	if (id >= clothParam._numFaces)
		isLoop = false;
	else {
		ino = clothileaf + id;

		getBVHNode(node, clothBvh, ino);
	}

	uint lRTri, rRTri;
	uint inos[3], jnos[3];
	REAL3 pis[3], pjs[3];
	REAL3 qis[3], qjs[3];
	REAL t = REAL_MAX;

	do {
		__syncthreads();
		if (isLoop) {
			icomp = getBVHIndex(path, level);
			getBVHAABB(comp._aabb, obsBvh, icomp);
			isIntersected = intersect(node._aabb, comp._aabb);
			if (isIntersected && icomp < obsileaf) {
				level++;
				path <<= 1u;
			}
			else {
				if (isIntersected) {
					comp._face = obsBvh._faces[icomp - obsileaf];
					ino = atomicAdd(&s_bvttNum, 1u);
					s_bvtts[0][ino] = node._face;
					s_bvtts[1][ino] = comp._face;
				}

				if ((path & 1u) == 0u)
					path |= 1u;
				else {
					if (path + 1u >> level) {
						isLoop = false;
						atomicAdd(&s_endNum, 1u);
					}
					else {
						do {
							level--;
							path >>= 1u;
						} while (path & 1u);
						path |= 1u;
					}
				}
			}
		}
		__syncthreads();
		if (s_endNum >= blockDim.x || s_bvttNum >= BVTT_SHARED_SIZE - blockDim.x) {
			for (ino = threadIdx.x; ino < s_bvttNum; ino += blockDim.x) {
				lRTri = clothRTri._info[s_bvtts[0][ino]];
				rRTri = obsRTri._info[s_bvtts[1][ino]];
				getMeshElements_device(clothParam, obsParam, s_bvtts[0][ino], s_bvtts[1][ino], inos, jnos, pis, pjs, qis, qjs, dt);
				getObstacleCCDtime_device(
					lRTri, rRTri,
					inos, pis, qis,
					jnos, pjs, qjs,
					thickness, &t);
			}
			__syncthreads();
			if (threadIdx.x == 0u)
				s_bvttNum = 0u;
		}
	} while (s_endNum < blockDim.x);

	s_minTimes[threadIdx.x] = t;
	for (ino = blockDim.x >> 1u; ino > 32u; ino >>= 1u) {
		__syncthreads();
		if (threadIdx.x < ino)
			if (s_minTimes[threadIdx.x] > s_minTimes[threadIdx.x + ino])
				s_minTimes[threadIdx.x] = s_minTimes[threadIdx.x + ino];
	}
	__syncthreads();
	if (threadIdx.x < 32u) {
		warpMin(s_minTimes, threadIdx.x);
		if (threadIdx.x == 0u)
			atomicMin_REAL(minTime, s_minTimes[0]);
	}
}
//-------------------------------------------------------------------------
__global__ void getClothCollisionElementsSDF_T_kernel(
	ContactElemSDFParam ceParam, 
	ObjParam clothParam, PRITree obsPRITree)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= clothParam._numFaces)
		return;

	ContactElemSDF ce;
	uint ino[3];
	ino[0] = clothParam._fs[id * 3 + 0];
	ino[1] = clothParam._fs[id * 3 + 1];
	ino[2] = clothParam._fs[id * 3 + 2];
	REAL3 ps[3];
	ps[0] = make_double3(clothParam._ns[ino[0] * 3 + 0], clothParam._ns[ino[0] * 3 + 1], clothParam._ns[ino[0] * 3 + 2]);
	ps[1] = make_double3(clothParam._ns[ino[1] * 3 + 0], clothParam._ns[ino[1] * 3 + 1], clothParam._ns[ino[1] * 3 + 2]);
	ps[2] = make_double3(clothParam._ns[ino[2] * 3 + 0], clothParam._ns[ino[2] * 3 + 1], clothParam._ns[ino[2] * 3 + 2]);

	AABB aabb;
	aabb._min = obsPRITree._center - obsPRITree._half;
	aabb._max = obsPRITree._center + obsPRITree._half;
	if (intersect(aabb, ps[0], 0.0) &&
		intersect(aabb, ps[1], 0.0) &&
		intersect(aabb, ps[2], 0.0)) 
	{
		REAL dists[3];
		for (uint i = 0; i < 3; i++)
			dists[i] = getDistancePRI(obsPRITree, ps[i]);

		uint minId = 0, num = 0;
		if (dists[minId] > dists[1]) minId = 1;
		if (dists[minId] > dists[2]) minId = 2;

		ce._dist = getDistancePRItoTri(obsPRITree, ps, minId, ce._w + 0u, ce._w + 1u, &ce._norm);
	}
	else {
		ce._dist = REAL_MAX;
		ce._norm = make_REAL3(0.0);
	}
	ceParam._felems[id] = ce;
}
__global__ void getClothCollisionElementsSDF_V_kernel(
	ContactElemSDFParam ceParam,
	ObjParam clothParam, PRITree obsPRITree)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= clothParam._numNodes)
		return;

	ContactElemSDF ce;
	REAL3 p;
	p = make_double3(clothParam._ns[id * 3 + 0], clothParam._ns[id * 3 + 1], clothParam._ns[id * 3 + 2]);

	AABB aabb;
	aabb._min = obsPRITree._center - obsPRITree._half;
	aabb._max = obsPRITree._center + obsPRITree._half;

	if (intersect(aabb, p, 0.0))
		ce._dist = getDistancePRI(obsPRITree, p, &ce._norm);
	else {
		ce._dist = REAL_MAX;
		ce._norm = make_REAL3(0.0);
	}

	ceParam._nelems[id] = ce;
}
__global__ void getSDFCCDtime_T_kernel(
	ObjParam clothParam, PRITree obsPRITree,
	const REAL thickness, const REAL dt, 
	REAL* minTime)
{
	extern __shared__ REAL s_minTimes[];
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	REAL t = *minTime;
	if (t > 1.0 && id < clothParam._numFaces) {
		uint ino[3];
		ino[0] = clothParam._fs[id * 3 + 0];
		ino[1] = clothParam._fs[id * 3 + 1];
		ino[2] = clothParam._fs[id * 3 + 2];
		REAL3 ps[3], vs[3];
		ps[0] = make_double3(clothParam._ns[ino[0] * 3 + 0], clothParam._ns[ino[0] * 3 + 1], clothParam._ns[ino[0] * 3 + 2]);
		ps[1] = make_double3(clothParam._ns[ino[1] * 3 + 0], clothParam._ns[ino[1] * 3 + 1], clothParam._ns[ino[1] * 3 + 2]);
		ps[2] = make_double3(clothParam._ns[ino[2] * 3 + 0], clothParam._ns[ino[2] * 3 + 1], clothParam._ns[ino[2] * 3 + 2]);
		vs[0] = make_double3(clothParam._vs[ino[0] * 3 + 0], clothParam._vs[ino[0] * 3 + 1], clothParam._vs[ino[0] * 3 + 2]);
		vs[1] = make_double3(clothParam._vs[ino[1] * 3 + 0], clothParam._vs[ino[1] * 3 + 1], clothParam._vs[ino[1] * 3 + 2]);
		vs[2] = make_double3(clothParam._vs[ino[2] * 3 + 0], clothParam._vs[ino[2] * 3 + 1], clothParam._vs[ino[2] * 3 + 2]);
		ps[0] += vs[0] * dt;
		ps[1] += vs[1] * dt;
		ps[2] += vs[2] * dt;

		AABB aabb;
		aabb._min = obsPRITree._center - obsPRITree._half;
		aabb._max = obsPRITree._center + obsPRITree._half;
		if (intersect(aabb, ps[0], 0.0) &&
			intersect(aabb, ps[1], 0.0) &&
			intersect(aabb, ps[2], 0.0))
		{
			REAL dists[3];
			for (uint i = 0; i < 3; i++)
				dists[i] = getDistancePRI(obsPRITree, ps[i]);

			uint minId = 0, num = 0;
			if (dists[minId] > dists[1]) minId = 1;
			if (dists[minId] > dists[2]) minId = 2;

			REAL dist = getDistancePRItoTri(obsPRITree, ps, minId);
			//printf("%f %f, %f\n", dist, thickness * COL_THICKNESS_RATIO * 0.5, t);
			if (dist < thickness* COL_THICKNESS_RATIO * 0.5)
				*minTime = 1.0;
		}
	}
}
__global__ void getSDFCCDtime_V_kernel(
	ObjParam clothParam, PRITree obsPRITree,
	const REAL thickness, const REAL dt,
	REAL* minTime)
{
	extern __shared__ REAL s_minTimes[];
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	REAL t = *minTime;
	if (t > 1.0 && id < clothParam._numNodes) {
		REAL3 p, v;
		p = make_double3(clothParam._ns[id * 3 + 0], clothParam._ns[id * 3 + 1], clothParam._ns[id * 3 + 2]);
		v = make_double3(clothParam._vs[id * 3 + 0], clothParam._vs[id * 3 + 1], clothParam._vs[id * 3 + 2]);
		p += v * dt;

		AABB aabb;
		aabb._min = obsPRITree._center - obsPRITree._half;
		aabb._max = obsPRITree._center + obsPRITree._half;

		if (intersect(aabb, p, 0.0)) {
			REAL dist = getDistancePRI(obsPRITree, p);
			//printf("%f %f, %f\n", dist, thickness * COL_THICKNESS_RATIO * 0.5, t);
			if (dist < thickness * COL_THICKNESS_RATIO * 0.5)
				*minTime = 1.0;
		}
	}
}

#endif