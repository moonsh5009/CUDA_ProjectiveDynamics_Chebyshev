#ifndef __COLLISION_MANAGER_CUH__
#define __COLLISION_MANAGER_CUH__

#pragma once
#include "CollisionSolver.h"
#include "PrimalTree.cuh"
#include "../include/CUDA_Custom/DeviceManager.cuh"

//----------------------------------------------
#define SOLVER_EPS						0.0
#define DISTANCE_EPS					1.0e-50
#define W_EPS							0.0
//----------------------------------------------
#define COL_CCD_THICKNESS_DETECT		1.0e-6
#define COL_THICKNESS_RATIO				0.25
//----------------------------------------------
#define CUBIC_ITERATOR					30
#define CUBIC_TOLERANCE					1.0e-20
//----------------------------------------------
#define BVTT_SHARED_SIZE				2048
#define BVTT_SIZE						10000
//-------------------------------------------------------------------------
inline __host__ __device__ __forceinline__
void PrintCE(const ContactElem& ce) {
	printf("%d, %d: %d, %d, %d, %d, %f\n", (int)ce._isObs, (int)ce._isFV, ce._i[0], ce._i[1], ce._i[2], ce._i[3], ce._info);
}
inline __device__ __forceinline__
void makeSelfCE(
	ContactElem& ce, 
	bool isFV, REAL info, uchar isObs, const REAL3& norm,
	REAL w0, REAL w1,
	uint i0, uint i1, uint i2, uint i3)
{
	ce._isFV = isFV;
	ce._info = info;
	ce._isObs = isObs;
	ce._norm = norm;
	ce._w[0] = w0;
	ce._w[1] = w1;

	ce._i[0] = i0;  ce._i[1] = i1;  ce._i[2] = i2;  ce._i[3] = i3;
	//if (isFV) {
	//	if (i0 < i1 && i0 < i2 && i1 < i2) {
	//		ce._i[0] = i0;  ce._i[1] = i1;  ce._i[2] = i2;  ce._i[3] = i3;
	//	}
	//	else if (i0 < i1 && i0 < i2 && i2 < i1) {
	//		ce._i[0] = i0;  ce._i[1] = i2;  ce._i[2] = i1;  ce._i[3] = i3;
	//	}
	//	else if (i1 < i0 && i1 < i2 && i0 < i2) {
	//		ce._i[0] = i1;  ce._i[1] = i0;  ce._i[2] = i2;  ce._i[3] = i3;
	//	}
	//	else if (i1 < i0 && i1 < i2 && i2 < i0) {
	//		ce._i[0] = i1;  ce._i[1] = i2;  ce._i[2] = i0;  ce._i[3] = i3;
	//	}
	//	else if (i2 < i0 && i2 < i1 && i0 < i1) {
	//		ce._i[0] = i2;  ce._i[1] = i0;  ce._i[2] = i1;  ce._i[3] = i3;
	//	}
	//	else if (i2 < i0 && i2 < i1 && i1 < i0) {
	//		ce._i[0] = i2;  ce._i[1] = i1;  ce._i[2] = i0;  ce._i[3] = i3;
	//	}
	//	else {
	//		printf("Error: tri %d, %d, %d, %d\n", i0, i1, i2, i3);
	//		//assert(0);
	//	}
	//}
	//else {
	//	if (!isObs) {
	//		if (i0 < i1 && i0 < i2 && i0 < i3 && i2 < i3) {
	//			ce._i[0] = i0;  ce._i[1] = i1;  ce._i[2] = i2;  ce._i[3] = i3;
	//		}
	//		else if (i0 < i1 && i0 < i2 && i0 < i3 && i3 < i2) {
	//			ce._i[0] = i0;  ce._i[1] = i1;  ce._i[2] = i3;  ce._i[3] = i2;
	//		}
	//		else if (i1 < i0 && i1 < i2 && i1 < i3 && i2 < i3) {
	//			ce._i[0] = i1;  ce._i[1] = i0;  ce._i[2] = i2;  ce._i[3] = i3;
	//		}
	//		else if (i1 < i0 && i1 < i2 && i1 < i3 && i3 < i2) {
	//			ce._i[0] = i1;  ce._i[1] = i0;  ce._i[2] = i3;  ce._i[3] = i2;
	//		}
	//		else if (i2 < i0 && i2 < i1 && i2 < i3 && i0 < i1) {
	//			ce._i[0] = i2;  ce._i[1] = i3;  ce._i[2] = i0;  ce._i[3] = i1;
	//		}
	//		else if (i2 < i0 && i2 < i1 && i2 < i3 && i1 < i0) {
	//			ce._i[0] = i2;  ce._i[1] = i3;  ce._i[2] = i1;  ce._i[3] = i0;
	//		}
	//		else if (i3 < i0 && i3 < i1 && i3 < i2 && i0 < i1) {
	//			ce._i[0] = i3;  ce._i[1] = i2;  ce._i[2] = i0;  ce._i[3] = i1;
	//		}
	//		else if (i3 < i0 && i3 < i1 && i3 < i2 && i1 < i0) {
	//			ce._i[0] = i3;  ce._i[1] = i2;  ce._i[2] = i1;  ce._i[3] = i0;
	//		}
	//	}
	//	else {
	//		if (i0 < i1) {
	//			ce._i[0] = i0;  ce._i[1] = i1;
	//		}
	//		else {
	//			ce._i[0] = i1;  ce._i[1] = i0;
	//		}
	//		if (i2 < i3) {
	//			ce._i[2] = i2;  ce._i[3] = i3;
	//		}
	//		else {
	//			ce._i[2] = i3;  ce._i[3] = i2;
	//		}
	//	}
	//}
}
inline __device__ __forceinline__
void addCE(const ContactElemParam& ceParams, const ContactElem* ces, const uint num) {
	uint ind = atomicAdd(ceParams.d_tmp, num);
	for (uint i = 0u; i < num; i++) {
		ceParams._elems[i + ind] = ces[i];
		//PrintCE(ces[i]);
	}
}
//-------------------------------------------------------------------------
inline __device__ __forceinline__ bool cmpSolver(REAL a, REAL b) {
	a -= b;
	return a >= -1.0e-10 && a <= 1.0e-10;
}
inline __device__ REAL QuadraticEval(REAL a, REAL b, REAL c, REAL x) {
	return (a * x + b) * x + c;
}
inline __device__ REAL CubicEval(REAL a, REAL b, REAL c, REAL d, REAL x) {
	return ((a * x + b) * x + c) * x + d;
}
inline __device__ REAL CubicIterator(REAL a, REAL b, REAL c, REAL d, REAL r0, REAL r1, REAL x0, REAL x1) {
	REAL x, result;
	//if (x0 >= -SOLVER_EPS && x0 <= SOLVER_EPS)		result = r0;
	//else if (x1 >= -SOLVER_EPS && x1 <= SOLVER_EPS) result = r1;
	if (x0 == 0.0)		result = r0;
	else if (x1 == 0.0) result = r1;
	else {
		result = 0.5 * (r0 + r1);
		uint itr;
		for (itr = 0u; itr < CUBIC_ITERATOR; itr++) {
			x = CubicEval(a, b, c, d, result);
			//if (x >= -CUBIC_TOLERANCE && x <= CUBIC_TOLERANCE)
			//	break;
			if (x == 0.0)
				break;

			if (x0 * x < 0.0)	r1 = result;
			else				r0 = result;

			if (result == 0.5 * (r0 + r1))
				break;
			result = 0.5 * (r0 + r1);
		}
		/*x = CubicEval(a, b, c, d, result);
		if (x < -CUBIC_TOLERANCE || x > CUBIC_TOLERANCE) {
			printf("%.15f\n", x);
			result = REAL_MAX;
		}*/
	}
	return result;
}
inline __device__ uint QuadraticSolver(REAL a, REAL b, REAL c, REAL* times)
{
	uint num = 0;
	REAL t;
	if (a >= -SOLVER_EPS && a <= SOLVER_EPS) {
		if (b < -SOLVER_EPS || b > SOLVER_EPS) {
			t = -c / b;
			if (t >= 0.0 && t <= 1.0)
				times[num++] = t;
		}
	}
	else {
		c = b * b - 4.0 * a * c;
		if (c >= 0.0) {
			c = sqrt(c);
			if (c < 0.0)
				c = 0.0;
			a = 1.0 / (a + a);
			c *= a;
			a *= -b;

			t = a + c;
			if (t >= 0.0 && t <= 1.0)
				times[num++] = t;
			if (c != 0.0) {
				t = a - c;
				if (t >= 0.0 && t <= 1.0)
					times[num++] = t;
			}
		}
		/*if (c > 0.0) {
			if (b < 0.0) {
				REAL cx = -b / (a + a);
				REAL cy = QuadraticEval(a, b, c, cx);
				if (cy <= 0.0) {
					if (cy == 0.0) {
						if (cx >= 0.0 && cx <= 1.0)
							times[num++] = cx;
					}
					else {
						REAL det = b * b - 4.0 * a * c;
						a = 1.0 / (a + a);
						det = sqrt(det);
						t = (-b - det) * a;
						if (t >= 0.0 && t <= 1.0)
							times[num++] = t;
						t = (-b + det) * a;
						if (t >= 0.0 && t <= 1.0)
							times[num++] = t;
					}
				}
			}
		}
		else if (c < 0.0) {
			REAL r1 = QuadraticEval(a, b, c, 1.0);
			if (r1 >= 0.0) {
				if (r1 == 0.0)
					times[num++] = 1.0;
				else {
					t = (-b + sqrt(b * b - 4.0 * a * c)) / (a + a);
					if (t >= 0.0 && t <= 1.0)
						times[num++] = t;
				}
			}
		}
		else {
			times[num++] = 0.0;
			t = -b / a;
			if (t >= 0.0 && t <= 1.0)
				times[num++] = t;
		}*/
	}
	return num;
}
inline __device__ uint CubicSolver(REAL a, REAL b, REAL c, REAL d, REAL* times) {
	uint num = 0u;
	if (a >= -SOLVER_EPS && a <= SOLVER_EPS)
		num = QuadraticSolver(b, c, d, times);
	else {
		if (a < 0.0)
		{
			a = -a; b = -b; c = -c; d = -d;
		}
		uint csNum = 0u;
		uchar2 cs[3];

		REAL rs[4];
		REAL fs[4];
		rs[0] = 0.0; rs[1] = 1.0;
		fs[0] = CubicEval(a, b, c, d, rs[0]);
		fs[1] = CubicEval(a, b, c, d, rs[1]);
		REAL det = b * b - 3.0 * a * c;
		if (det >= 0.0)
		{
			rs[2] = (-b - sqrt(det)) / (3.0 * a);
			rs[3] = (-b + sqrt(det)) / (3.0 * a);
			fs[2] = CubicEval(a, b, c, d, rs[2]);
			fs[3] = CubicEval(a, b, c, d, rs[3]);

			if (rs[2] >= 0.0 && rs[3] <= 1.0) {
				if (fs[0] * fs[2] <= 0.0)
					cs[csNum++] = make_uchar2(0u, 2u);
				if (fs[2] * fs[3] <= 0.0)
					cs[csNum++] = make_uchar2(2u, 3u);
				if (fs[3] * fs[1] <= 0.0)
					cs[csNum++] = make_uchar2(3u, 1u);
			}
			else if (rs[2] >= 0.0 && rs[2] <= 1.0) {
				if (fs[0] * fs[2] <= 0.0)
					cs[csNum++] = make_uchar2(0u, 2u);
				if (fs[2] * fs[1] <= 0.0)
					cs[csNum++] = make_uchar2(2u, 1u);
			}
			else if (rs[3] >= 0.0 && rs[3] <= 1.0) {
				if (fs[0] * fs[3] <= 0.0)
					cs[csNum++] = make_uchar2(0u, 3u);
				if (fs[3] * fs[1] <= 0.0)
					cs[csNum++] = make_uchar2(3u, 1u);
			}
		}
		if (!csNum && fs[0] * fs[1] <= 0.0)
			cs[csNum++] = make_uchar2(0u, 1u);

		REAL eval;
		for (uint i = 0u; i < csNum; i++) {
			//if (fs[cs[i].x] < -SOLVER_EPS || fs[cs[i].x] > SOLVER_EPS || fs[cs[i].y] < -SOLVER_EPS || fs[cs[i].y] > SOLVER_EPS) {
				eval = CubicIterator(a, b, c, d, rs[cs[i].x], rs[cs[i].y], fs[cs[i].x], fs[cs[i].y]);
				if (eval >= 0.0 && eval <= 1.0)
					times[num++] = eval;
			//}
		}
	}
	return num;
}
//-------------------------------------------------------------------------
inline __device__ uint FinePlaneCoTime(
	const REAL3& a0, const REAL3& b0, const REAL3& c0, const REAL3& d0,
	const REAL3& a1, const REAL3& b1, const REAL3& c1, const REAL3& d1,
	REAL* times)
{
	REAL3 v01_0 = b0 - a0; REAL3 v01_1 = b1 - a1 - v01_0;
	REAL3 v02_0 = c0 - a0; REAL3 v02_1 = c1 - a1 - v02_0;
	REAL3 v0p_0 = d0 - a0; REAL3 v0p_1 = d1 - a1 - v0p_0;
	REAL3 cross0 = Cross(v01_1, v02_1);
	REAL3 cross1 = Cross(v01_0, v02_1) + Cross(v01_1, v02_0);
	REAL3 cross2 = Cross(v01_0, v02_0);

	REAL a = Dot(cross0, v0p_1);
	REAL b = Dot(cross0, v0p_0) + Dot(cross1, v0p_1);
	REAL c = Dot(cross1, v0p_0) + Dot(cross2, v0p_1);
	REAL d = Dot(cross2, v0p_0);

	uint num = CubicSolver(a, b, c, d, times);
	return num;
}
inline __device__ uint FineLineCoTime(
	const REAL3& a0, const REAL3& b0, const REAL3& c0,
	const REAL3& a1, const REAL3& b1, const REAL3& c1,
	REAL* times)
{
	REAL3 v01_0 = b0 - a0; REAL3 v01_1 = b1 - a1 - v01_0;
	REAL3 v02_0 = c0 - a0; REAL3 v02_1 = c1 - a1 - v02_0;

	REAL3 cross0 = Cross(v01_1, v02_1);
	REAL3 cross1 = Cross(v01_0, v02_1) + Cross(v01_1, v02_0);
	REAL3 cross2 = Cross(v01_0, v02_0);

	uint num = 0u;
	REAL buffer[2], e;
	uint qnum, i;

	REAL eps = 1.0e-5;
	qnum = QuadraticSolver(cross0.x, cross1.x, cross2.x, buffer);
	for (i = 0u; i < qnum; i++) {
		if (buffer[i] >= 0.0 && buffer[i] <= 1.0) {
			e = QuadraticEval(cross0.y, cross1.y, cross2.y, buffer[i]);
			if (e >= -eps && e < eps) {
				e = QuadraticEval(cross0.z, cross1.z, cross2.z, buffer[i]);
				if (e >= -eps && e < eps) {
					if (num == 0u)
						times[num++] = buffer[i];
					else if (num == 1u) {
						if (!cmpSolver(times[0], buffer[i]))
							times[num++] = buffer[i];
					}
					else {
						if (!cmpSolver(times[0], buffer[i]) && buffer[i] < times[0]) {
							times[1] = times[0];
							times[0] = buffer[i];
						}
						else if (!cmpSolver(times[1], buffer[i]) && buffer[i] < times[1])
							times[1] = buffer[i];
					}
				}
			}
		}
	}
	qnum = QuadraticSolver(cross0.y, cross1.y, cross2.y, buffer);
	for (i = 0u; i < qnum; i++) {
		if (buffer[i] >= 0.0 && buffer[i] <= 1.0) {
			e = QuadraticEval(cross0.x, cross1.x, cross2.x, buffer[i]);
			if (e >= -eps && e < eps) {
				e = QuadraticEval(cross0.z, cross1.z, cross2.z, buffer[i]);
				if (e >= -eps && e < eps) {
					if (num == 0u)
						times[num++] = buffer[i];
					else if (num == 1u) {
						if (!cmpSolver(times[0], buffer[i]))
							times[num++] = buffer[i];
					}
					else {
						if (!cmpSolver(times[0], buffer[i]) && buffer[i] < times[0]) {
							times[1] = times[0];
							times[0] = buffer[i];
						}
						else if (!cmpSolver(times[1], buffer[i]) && buffer[i] < times[1])
							times[1] = buffer[i];
					}
				}
			}
		}
	}
	qnum = QuadraticSolver(cross0.z, cross1.z, cross2.z, buffer);
	for (i = 0u; i < qnum; i++) {
		if (buffer[i] >= 0.0 && buffer[i] <= 1.0) {
			e = QuadraticEval(cross0.y, cross1.y, cross2.y, buffer[i]);
			if (e >= -eps && e < eps) {
				e = QuadraticEval(cross0.x, cross1.x, cross2.x, buffer[i]);
				if (e >= -eps && e < eps) {
					if (num == 0u)
						times[num++] = buffer[i];
					else if (num == 1u) {
						if (!cmpSolver(times[0], buffer[i]))
							times[num++] = buffer[i];
					}
					else {
						if (!cmpSolver(times[0], buffer[i]) && buffer[i] < times[0]) {
							times[1] = times[0];
							times[0] = buffer[i];
						}
						else if (!cmpSolver(times[1], buffer[i]) && buffer[i] < times[1])
							times[1] = buffer[i];
					}
				}
			}
		}
	}
	//if (num) printf("!!!!!!!!!!!! %d (%f %f)\n", num, times[0], times[1]);

	return num;
}
inline __device__ uint FineVertexCoTime(
	const REAL3& a0, const REAL3& b0,
	const REAL3& a1, const REAL3& b1,
	REAL* time)
{
	REAL3 v01_0 = b0 - a0; REAL3 v01_1 = b1 - a1 - v01_0;

	uint num = 0u;
	REAL t = 0.0;
	if (v01_1.x < -SOLVER_EPS || v01_1.x > SOLVER_EPS) {
		t += -v01_0.x / v01_1.x;
		num++;
	}
	if (v01_1.y < -SOLVER_EPS || v01_1.y > SOLVER_EPS) {
		t += -v01_0.y / v01_1.y;
		num++;
	}
	if (v01_1.z < -SOLVER_EPS || v01_1.z > SOLVER_EPS) {
		t += -v01_0.z / v01_1.z;
		num++;
	}
	if (num) {
		t /= (REAL)num;
		if (t >= 0.0 && t <= 1.0) {
			*time = t;
			num = 1u;
		}
		else num = 0u;
	}
	/*REAL t = REAL_MAX;
	if (v01_1.x < -SOLVER_EPS || v01_1.x > SOLVER_EPS)
		t = min(-v01_0.x / v01_1.x, t);
	if (v01_1.y < -SOLVER_EPS || v01_1.y > SOLVER_EPS)
		t = min(-v01_0.y / v01_1.y, t);
	if (v01_1.z < -SOLVER_EPS || v01_1.z > SOLVER_EPS)
		t = min(-v01_0.z / v01_1.z, t);

	if (t >= 0.0 && t <= 1.0) {
		*time = t;
		num = 1u;
	}*/
	return num;
}
//-------------------------------------------------------------------------
inline __device__ REAL getDistanceEV(
	const REAL3& v0, const REAL3& v1, const REAL3& p,
	REAL* w = nullptr)
{
	REAL result = REAL_MAX;
	REAL3 norm = v1 - v0;
	REAL lnorm = LengthSquared(norm);
	if (lnorm) {
		norm *= 1.0 / sqrt(lnorm);
		REAL t1 = Dot(p - v0, norm);
		REAL t2 = Dot(p - v1, norm);
		t2 = t1 - t2;

		REAL w0;
		if (t2) {
			w0 = t1 / t2;
			//printf("aiosdvnlan %f %f %f\n", t1, t2, w0);
			if (w0 >= 0.0 && w0 <= 1.0) {
				if (w) *w = w0;
				result = LengthSquared(p - v0) - t1 * t1;
				if (result > 0.0)	result = sqrt(result);
				else				result = 0.0;
			}
		}
	}
	return result;
}
inline __device__ REAL getDistanceEV2(
	const REAL3& v0, const REAL3& v1, const REAL3& p,
	REAL* w = nullptr)
{
	REAL result = REAL_MAX;
	REAL v01 = LengthSquared(v1 - v0);
	REAL v0p = Dot(p - v0, v1 - v0);
	REAL w0;
	if (v0p <= 0.0)				w0 = 0.0;
	else if (v0p >= v01)		w0 = 1.0;
	else						w0 = v0p / v01;

	if (w) *w = w0;

	REAL3 norm = p - v0 - (v1 - v0) * w0;
	result = Length(norm);
	return result;
}
inline __device__ REAL getDistanceTV(
	const REAL3& v0, const REAL3& v1, const REAL3& v2, const REAL3& p,
	REAL delta, REAL* wa = nullptr, REAL* wb = nullptr)
{
#if 1
	REAL result = REAL_MAX;
	REAL w0, w1;
	REAL3 v20 = v0 - v2;
	REAL3 v21 = v1 - v2;
	REAL t0 = Dot(v20, v20);
	REAL t1 = Dot(v21, v21);
	REAL t2 = Dot(v20, v21);
	REAL t3 = Dot(v20, p - v2);
	REAL t4 = Dot(v21, p - v2);
	REAL det = t0 * t1 - t2 * t2;
	if (fabs(det) > 1.0e-20) {
		REAL invdet = 1.0 / det;
		w0 = (+t1 * t3 - t2 * t4) * invdet;
		if (w0 >= 0.0 - W_EPS && w0 <= 1.0 + W_EPS) {
			w1 = (-t2 * t3 + t0 * t4) * invdet;
			if (w1 >= 0.0 - W_EPS && w1 <= 1.0 + W_EPS) {
				const REAL w2 = 1.0 - w0 - w1;
				if (w2 >= 0.0 - W_EPS && w2 <= 1.0 + W_EPS) {
					REAL3 pw = v0 * w0 + v1 * w1 + v2 * w2;
					if (wa) {
						*wa = w0;
						*wb = w1;
					}
					result = Length(pw - p);
				}
			}
		}
	}
	return result;
#else
	REAL result = REAL_MAX;
	REAL3 v01 = v1 - v0;
	REAL3 v02 = v2 - v0;
	REAL3 v0p = p - v0;

	REAL3 norm = Cross(v01, v02);
	REAL nl2 = LengthSquared(norm);

	REAL3 tmp;
	tmp = Cross(v0p, v02);
	REAL w_0p02 = Dot(tmp, norm);
	tmp = Cross(v01, v0p);
	REAL w_010p = Dot(tmp, norm);
	REAL ba, bb, bc;

	if (nl2 > DISTANCE_EPS && w_0p02 >= 0.0 && w_010p >= 0.0 && nl2 - w_0p02 - w_010p >= 0.0)
	{
		bb = w_0p02 / nl2;
		bc = w_010p / nl2;
		ba = 1.0 - bb - bc;
	}
	else
	{
		REAL distance, v;
		if (nl2 - w_0p02 - w_010p < 0.0 && ((distance = getDistanceEV2(v1, v2, p, &v)) < result))
		{
			result = distance;
			ba = 0.0;
			bb = 1.0 - v;
			bc = v;
		}
		if (w_0p02 < 0 && ((distance = getDistanceEV2(v0, v2, p, &v)) < result))
		{
			result = distance;
			ba = 1.0 - v;
			bb = 0.0;
			bc = v;
		}
		if (w_010p < 0 && ((distance = getDistanceEV2(v0, v1, p, &v)) < result))
		{
			result = distance;
			ba = 1.0 - v;
			bb = v;
			bc = 0.0;
		}
		if (result >= delta)
			return REAL_MAX;
	}
	if (wa) {
		*wa = ba;
		*wb = bb;
	}
	
	/*if (nl2 > DISTANCE_EPS) {
		norm *= 1.0 / sqrt(nl2);
		result = Dot(p - v0, norm);
		if (result < 0.0)
			result = -result;
	}
	else {
		norm = p - v0 * ba - v1 * bb - v2 * bc;
		result = Length(norm);
	}*/
	norm = p - v0 * ba - v1 * bb - v2 * bc;
	result = Length(norm);

	return result;
#endif
}
inline __device__ REAL getDistanceEE(
	const REAL3& pa, const REAL3& pb, const REAL3& pc, const REAL3& pd,
	REAL delta, REAL* wa = nullptr, REAL* wb = nullptr)
{
#if 1
	REAL result = REAL_MAX;
	REAL w0, w1;
	REAL3 vp = pb - pa;
	REAL3 vq = pd - pc;
	REAL t0 = Dot(vp, vp);
	REAL t1 = Dot(vq, vq);
	REAL t2 = Dot(vp, vq);
	REAL det = t0 * t1 - t2 * t2;
	if (fabs(det) < 1.0e-50) {
		REAL lp0 = Dot(pa, vp);
		REAL lp1 = Dot(pb, vp);
		REAL lq0 = Dot(pc, vp);
		REAL lq1 = Dot(pd, vp);
		REAL p_min = (lp0 < lp1) ? lp0 : lp1;
		REAL p_max = (lp0 > lp1) ? lp0 : lp1;
		REAL q_min = (lq0 < lq1) ? lq0 : lq1;
		REAL q_max = (lq0 > lq1) ? lq0 : lq1;
		REAL lm;
		if (p_max < q_min)		lm = (p_max + q_min) * 0.5;
		else if (p_min > q_max) lm = (q_max + p_min) * 0.5;
		else if (p_max < q_max)
			if (p_min < q_min)	lm = (p_max + q_min) * 0.5;
			else				lm = (p_max + p_min) * 0.5;
		else
			if (p_min < q_min)	lm = (q_max + q_min) * 0.5;
			else				lm = (q_max + p_min) * 0.5;
		w0 = (lm - lp0) / (lp1 - lp0);
		if (w0 >= 0.0 - W_EPS && w0 <= 1.0 + W_EPS) {
			w1 = (lm - lq0) / (lq1 - lq0);
			if (w1 >= 0.0 - W_EPS && w1 <= 1.0 + W_EPS) {
				Normalize(vp);
				REAL3 ppc = pa - pc;
				REAL3 vert = ppc - vp * Dot(ppc, vp);
				if (wa) { *wa = w0; *wb = w1; }
				result = Length(vert);
			}
		}
	}
	else {
		REAL t3 = Dot(vp, pc - pa);
		REAL t4 = Dot(vq, pc - pa);
		REAL invdet = 1.0 / det;
		w0 = (+t1 * t3 - t2 * t4) * invdet;
		if (w0 >= 0.0 - W_EPS && w0 <= 1.0 + W_EPS) {
			w1 = (+t2 * t3 - t0 * t4) * invdet;
			if (w1 >= 0.0 - W_EPS && w1 <= 1.0 + W_EPS) {
				if (wa) { *wa = w0; *wb = w1; }
				result = Length(pa + vp * w0 - pc - vq * w1);
			}
		}
	}
	return result;
#else
	REAL result = REAL_MAX;
	REAL3 vab = pb - pa;
	REAL3 vcd = pd - pc;
	REAL3 vac = pc - pa;

	REAL3 norm;
	norm = Cross(vab, vcd);
	REAL nl2 = LengthSquared(norm);

	REAL3 tmp;
	tmp = Cross(vac, vab);
	REAL w_acab = Dot(tmp, norm);
	tmp = Cross(vac, vcd);
	REAL w_accd = Dot(tmp, norm);
	REAL r, s;

	if (nl2 > DISTANCE_EPS && w_acab >= 0.0 && w_acab <= nl2 && w_accd >= 0.0 && w_accd <= nl2)
	{
		r = w_accd / nl2;
		s = w_acab / nl2;
	}
	else
	{
		REAL distance, v;

		if (w_accd < 0.0 && ((distance = getDistanceEV2(pc, pd, pa, &v)) < result))
		{
			result = distance;
			s = v;
			r = 0.0;
		}
		if (w_accd > nl2 && ((distance = getDistanceEV2(pc, pd, pb, &v)) < result))
		{
			result = distance;
			s = v;
			r = 1.0;
		}
		if (w_acab < 0.0 && ((distance = getDistanceEV2(pa, pb, pc, &v)) < result))
		{
			result = distance;
			r = v;
			s = 0.0;
		}
		if (w_acab > nl2 && ((distance = getDistanceEV2(pa, pb, pd, &v)) < result))
		{
			result = distance;
			r = v;
			s = 1.0;
		}

		if (result >= delta)
			return REAL_MAX;
	}
	if (wa) {
		*wa = r;
		*wb = s;
	}
	if (nl2 > DISTANCE_EPS) {
		norm *= 1.0 / sqrt(nl2);
		result = Dot(pa - pc, norm);
		if (result < 0.0)
			result = -result;
	}
	else {
		norm = pa + (pb - pa) * r - pc - (pd - pc) * s;
		result = Length(norm);
	}
	return result;
#endif
}
inline __device__ REAL getDistanceElements(
	bool isFV,
	const REAL3& p0, const REAL3& p1, const REAL3& p2, const REAL3& p3,
	REAL delta, REAL* wa = nullptr, REAL* wb = nullptr)
{
	REAL result;
	if (isFV)	result = getDistanceTV(p0, p1, p2, p3, delta, wa, wb);
	else		result = getDistanceEE(p0, p1, p2, p3, delta, wa, wb);
	/*if (wa) {
		if (*wa < 0.0) *wa = 0.0;
		if (*wa > 1.0) *wa = 1.0;
	}
	if (wb) {
		if (*wb < 0.0) *wb = 0.0;
		if (*wb > 1.0) *wb = 1.0;
	}*/
	return result;
}
inline __device__ REAL getDistanceTV(
	const REAL3& v0, const REAL3& v1, const REAL3& v2, const REAL3& p,
	REAL delta, REAL3& norm, REAL* wa = nullptr, REAL* wb = nullptr)
{
#if 1
	REAL result = REAL_MAX;
	REAL w0, w1;
	REAL3 v20 = v0 - v2;
	REAL3 v21 = v1 - v2;
	REAL t0 = Dot(v20, v20);
	REAL t1 = Dot(v21, v21);
	REAL t2 = Dot(v20, v21);
	REAL t3 = Dot(v20, p - v2);
	REAL t4 = Dot(v21, p - v2);
	REAL det = t0 * t1 - t2 * t2;
	if (fabs(det) > 1.0e-20) {
		REAL invdet = 1.0 / det;
		w0 = (+t1 * t3 - t2 * t4) * invdet;
		if (w0 >= 0.0 - W_EPS && w0 <= 1.0 + W_EPS) {
			w1 = (-t2 * t3 + t0 * t4) * invdet;
			if (w1 >= 0.0 - W_EPS && w1 <= 1.0 + W_EPS) {
				const REAL w2 = 1.0 - w0 - w1;
				if (w2 >= 0.0 - W_EPS && w2 <= 1.0 + W_EPS) {
					if (wa) {
						*wa = w0;
						*wb = w1;
					}
					REAL3 pc = v0 * w0 + v1 * w1 + v2 * w2;
					norm = p - pc;
					result = Length(norm);
					if (result > 0.0)
						norm *= 1.0 / result;
				}
			}
		}
	}
	return result;
#else
	REAL result = REAL_MAX;
	REAL3 v01 = v1 - v0;
	REAL3 v02 = v2 - v0;
	REAL3 v0p = p - v0;

	norm = Cross(v01, v02);
	REAL nl2 = LengthSquared(norm);

	REAL3 tmp;
	tmp = Cross(v0p, v02);
	REAL w_0p02 = Dot(tmp, norm);
	tmp = Cross(v01, v0p);
	REAL w_010p = Dot(tmp, norm);
	REAL ba, bb, bc;

	if (nl2 > DISTANCE_EPS && w_0p02 >= 0.0 && w_010p >= 0.0 && nl2 - w_0p02 - w_010p >= 0.0)
	{
		bb = w_0p02 / nl2;
		bc = w_010p / nl2;
		ba = 1.0 - bb - bc;
	}
	else
	{
		REAL distance, v;
		if (nl2 - w_0p02 - w_010p < 0.0 && ((distance = getDistanceEV2(v1, v2, p, &v)) < result))
		{
			result = distance;
			ba = 0.0;
			bb = 1.0 - v;
			bc = v;
		}
		if (w_0p02 < 0.0 && ((distance = getDistanceEV2(v0, v2, p, &v)) < result))
		{
			result = distance;
			ba = 1.0 - v;
			bb = 0.0;
			bc = v;
		}
		if (w_010p < 0.0 && ((distance = getDistanceEV2(v0, v1, p, &v)) < result))
		{
			result = distance;
			ba = 1.0 - v;
			bb = v;
			bc = 0.0;
		}
		if (result >= delta)
			return REAL_MAX;
	}
	if (wa) {
		*wa = ba;
		*wb = bb;
	}

	
	/*if (nl2 > DISTANCE_EPS) {
		norm *= 1.0 / sqrt(nl2);
		result = Dot(p - v0, norm);
		if (result < 0.0) {
			result = -result;
			norm.x = -norm.x;
			norm.y = -norm.y;
			norm.z = -norm.z;
		}
	}
	else {
		norm = p - v0 * ba - v1 * bb - v2 * bc;
		result = Length(norm);
		if (result > 0.0)
			norm *= 1.0 / result;
	}*/
	norm = p - v0 * ba - v1 * bb - v2 * bc;
	result = Length(norm);
	if (result > 0.0)
		norm *= 1.0 / result;

	return result;
#endif
}
inline __device__ REAL getDistanceEE(
	const REAL3& pa, const REAL3& pb, const REAL3& pc, const REAL3& pd,
	REAL delta, REAL3& norm, REAL* wa = nullptr, REAL* wb = nullptr)
{
#if 1
	REAL result = REAL_MAX;
	REAL w0, w1;
	REAL3 vp = pb - pa;
	REAL3 vq = pd - pc;
	REAL t0 = Dot(vp, vp);
	REAL t1 = Dot(vq, vq);
	REAL t2 = Dot(vp, vq);
	REAL det = t0 * t1 - t2 * t2;
	if (fabs(det) < 1.0e-50) {
		REAL lp0 = Dot(pa, vp);
		REAL lp1 = Dot(pb, vp);
		REAL lq0 = Dot(pc, vp);
		REAL lq1 = Dot(pd, vp);
		REAL p_min = (lp0 < lp1) ? lp0 : lp1;
		REAL p_max = (lp0 > lp1) ? lp0 : lp1;
		REAL q_min = (lq0 < lq1) ? lq0 : lq1;
		REAL q_max = (lq0 > lq1) ? lq0 : lq1;
		REAL lm;
		if (p_max < q_min)		lm = (p_max + q_min) * 0.5;
		else if (p_min > q_max) lm = (q_max + p_min) * 0.5;
		else if (p_max < q_max)
			if (p_min < q_min)	lm = (p_max + q_min) * 0.5;
			else				lm = (p_max + p_min) * 0.5;
		else
			if (p_min < q_min)	lm = (q_max + q_min) * 0.5;
			else				lm = (q_max + p_min) * 0.5;
		w0 = (lm - lp0) / (lp1 - lp0);
		if (w0 >= 0.0 - W_EPS && w0 <= 1.0 + W_EPS) {
			w1 = (lm - lq0) / (lq1 - lq0);
			if (w1 >= 0.0 - W_EPS && w1 <= 1.0 + W_EPS) {
				if (wa) { *wa = w0; *wb = w1; }
				Normalize(vp);
				REAL3 ppc = pa - pc;
				norm = vp * Dot(ppc, vp) - ppc;
				result = Length(norm);
				if (result > 0.0)
					norm *= 1.0 / result;
			}
		}
	}
	else {
		REAL t3 = Dot(vp, pc - pa);
		REAL t4 = Dot(vq, pc - pa);
		REAL invdet = 1.0 / det;
		w0 = (+t1 * t3 - t2 * t4) * invdet;
		if (w0 >= 0.0 - W_EPS && w0 <= 1.0 + W_EPS) {
			w1 = (+t2 * t3 - t0 * t4) * invdet;
			if (w1 >= 0.0 - W_EPS && w1 <= 1.0 + W_EPS) {
				if (wa) { *wa = w0; *wb = w1; }
				norm = pc + vq * w1 - pa - vp * w0;
				result = Length(norm);
				if (result > 0.0)
					norm *= 1.0 / result;
			}
		}
	}
	return result;
#else
	REAL result = REAL_MAX;
	REAL3 vab = pb - pa;
	REAL3 vcd = pd - pc;
	REAL3 vac = pc - pa;

	norm = Cross(vab, vcd);
	REAL nl2 = LengthSquared(norm);

	REAL3 tmp;
	tmp = Cross(vac, vab);
	REAL w_acab = Dot(tmp, norm);
	tmp = Cross(vac, vcd);
	REAL w_accd = Dot(tmp, norm);
	REAL r, s;

	if (nl2 > DISTANCE_EPS && w_acab >= 0.0 && w_acab <= nl2 && w_accd >= 0.0 && w_accd <= nl2)
	{
		r = w_accd / nl2;
		s = w_acab / nl2;
	}
	else
	{
		REAL distance, v;

		if (w_accd < 0.0 && ((distance = getDistanceEV2(pc, pd, pa, &v)) < result))
		{
			result = distance;
			s = v;
			r = 0.0;
		}
		if (w_accd > nl2 && ((distance = getDistanceEV2(pc, pd, pb, &v)) < result))
		{
			result = distance;
			s = v;
			r = 1.0;
		}
		if (w_acab < 0.0 && ((distance = getDistanceEV2(pa, pb, pc, &v)) < result))
		{
			result = distance;
			r = v;
			s = 0.0;
		}
		if (w_acab > nl2 && ((distance = getDistanceEV2(pa, pb, pd, &v)) < result))
		{
			result = distance;
			r = v;
			s = 1.0;
		}

		if (result >= delta)
			return REAL_MAX;
	}
	if (wa) {
		*wa = r;
		*wb = s;
	}
	if (nl2 > DISTANCE_EPS) {
		norm *= 1.0 / sqrt(nl2);
		result = Dot(pa - pc, norm);
		if (result < 0.0) {
			result = -result;
			norm.x = -norm.x;
			norm.y = -norm.y;
			norm.z = -norm.z;
		}
	}
	else {
		norm = pa + (pb - pa) * r - pc - (pd - pc) * s;
		result = Length(norm);
		if (result > 0.0)
			norm *= 1.0 / result;
	}
	return result;
#endif
}
inline __device__ REAL getDistanceElements(
	bool isFV,
	const REAL3& p0, const REAL3& p1, const REAL3& p2, const REAL3& p3,
	REAL delta, REAL3& norm, REAL* wa = nullptr, REAL* wb = nullptr)
{
	REAL result;
	if (isFV)	result = getDistanceTV(p0, p1, p2, p3, delta, norm, wa, wb);
	else		result = getDistanceEE(p0, p1, p2, p3, delta, norm, wa, wb);
	/*if (wa) {
		if (*wa < 0.0) *wa = 0.0;
		if (*wa > 1.0) *wa = 1.0;
	}
	if (wb) {
		if (*wb < 0.0) *wb = 0.0;
		if (*wb > 1.0) *wb = 1.0;
	}*/
	return result;
}
//-------------------------------------------------------------------------
__device__ void getMeshElements_device(
	const ObjParam& param,
	uint f0, uint f1,
	uint* inos, uint* jnos,
	REAL3* pis, REAL3* pjs)
{
	getIfromParam(param._fs, f0, inos);
	getIfromParam(param._fs, f1, jnos);
	getXfromParam(param._ns, inos, pis);
	getXfromParam(param._ns, jnos, pjs);
}
__device__ void getMeshElements_device(
	const ObjParam& param,
	uint f0, uint f1,
	uint* inos, uint* jnos,
	REAL3* pis, REAL3* pjs,
	REAL3* vis, REAL3* vjs,
	REAL* invMis, REAL* invMjs)
{
	getIfromParam(param._fs, f0, inos);
	getIfromParam(param._fs, f1, jnos);
	getXfromParam(param._ns, inos, pis);
	getXfromParam(param._ns, jnos, pjs);
	getXfromParam(param._vs, inos, vis);
	getXfromParam(param._vs, jnos, vjs);
	invMis[0] = param._invMs[inos[0]];
	invMis[1] = param._invMs[inos[1]];
	invMis[2] = param._invMs[inos[2]];
	invMjs[0] = param._invMs[jnos[0]];
	invMjs[1] = param._invMs[jnos[1]];
	invMjs[2] = param._invMs[jnos[2]];
}
__device__ void getMeshElements_device(
	const ObjParam& param,
	uint f0, uint f1,
	uint* inos, uint* jnos,
	REAL3* pis, REAL3* pjs,
	REAL3* qis, REAL3* qjs,
	REAL dt)
{
	getIfromParam(param._fs, f0, inos);
	getIfromParam(param._fs, f1, jnos);
	getXfromParam(param._ns, inos, pis);
	getXfromParam(param._ns, jnos, pjs);
	getXfromParam(param._vs, inos, qis);
	getXfromParam(param._vs, jnos, qjs);
	qis[0] = pis[0] + qis[0] * dt;
	qis[1] = pis[1] + qis[1] * dt;
	qis[2] = pis[2] + qis[2] * dt;
	qjs[0] = pjs[0] + qjs[0] * dt;
	qjs[1] = pjs[1] + qjs[1] * dt;
	qjs[2] = pjs[2] + qjs[2] * dt;
}
__device__ void getMeshElements_device(
	const ObjParam& clothParam,
	const ObjParam& obsParam,
	uint f0, uint f1,
	uint* inos, uint* jnos,
	REAL3* pis, REAL3* pjs)
{
	getIfromParam(clothParam._fs, f0, inos);
	getIfromParam(obsParam._fs, f1, jnos);
	getXfromParam(clothParam._ns, inos, pis);
	getXfromParam(obsParam._ns, jnos, pjs);
}
__device__ void getMeshElements_device(
	const ObjParam& clothParam,
	const ObjParam& obsParam,
	uint f0, uint f1,
	uint* inos, uint* jnos,
	REAL3* pis, REAL3* pjs,
	REAL3* vis, REAL3* vjs,
	REAL* invMis, REAL* invMjs)
{
	getIfromParam(clothParam._fs, f0, inos);
	getIfromParam(obsParam._fs, f1, jnos);
	getXfromParam(clothParam._ns, inos, pis);
	getXfromParam(obsParam._ns, jnos, pjs);
	getXfromParam(clothParam._vs, inos, vis);
	getXfromParam(obsParam._vs, jnos, vjs);

	invMis[0] = clothParam._invMs[inos[0]];
	invMis[1] = clothParam._invMs[inos[1]];
	invMis[2] = clothParam._invMs[inos[2]];
	invMjs[0] = 0.0;
	invMjs[1] = 0.0;
	invMjs[2] = 0.0;
}
__device__ void getMeshElements_device(
	const ObjParam& clothParam,
	const ObjParam& obsParam,
	uint f0, uint f1,
	uint* inos, uint* jnos,
	REAL3* pis, REAL3* pjs,
	REAL3* qis, REAL3* qjs,
	REAL dt)
{
	getIfromParam(clothParam._fs, f0, inos);
	getIfromParam(obsParam._fs, f1, jnos);
	getXfromParam(clothParam._ns, inos, pis);
	getXfromParam(obsParam._ns, jnos, pjs);
	getXfromParam(clothParam._vs, inos, qis);
	getXfromParam(obsParam._vs, jnos, qjs);
	qis[0] = pis[0] + qis[0] * dt;
	qis[1] = pis[1] + qis[1] * dt;
	qis[2] = pis[2] + qis[2] * dt;
	qjs[0] = pjs[0] + qjs[0] * dt;
	qjs[1] = pjs[1] + qjs[1] * dt;
	qjs[2] = pjs[2] + qjs[2] * dt;
}
//-------------------------------------------------------------------------
__device__ void getLastBvttIfaces(
	BVHParam& bvh,
	uint nodeL, uint nodeR,
	uint2* ifaces, uint& num)
{
	uint ileaf = bvh._size - bvh._numFaces;
	bool isLeafL = nodeL >= ileaf;
	bool isLeafR = nodeR >= ileaf;
	if (nodeL == nodeR)
		printf("asdfasdfasdfasdf\n");

	AABB aabbs[4];
	uint ifBuffer[4];
	uint numL, numR;

	if (isLeafL) {
		ifBuffer[0] = nodeL;
		numL = 1u;
	}
	else {
		ifBuffer[0] = (nodeL << 1u) + 1u;
		ifBuffer[1] = ifBuffer[0] + 1u;
		numL = 2u;
	}
	if (isLeafR) {
		ifBuffer[2] = nodeR;
		numR = 3u;
	}
	else {
		ifBuffer[2] = (nodeR << 1u) + 1u;
		ifBuffer[3] = ifBuffer[2] + 1u;
		numR = 4u;
	}

	uint i, j;
	for (i = 0u; i < numL; i++) {
		getBVHAABB(aabbs[i], bvh, ifBuffer[i]);
		ifBuffer[i] = bvh._faces[ifBuffer[i] - ileaf];
	}
	for (j = 2u; j < numR; j++) {
		getBVHAABB(aabbs[j], bvh, ifBuffer[j]);
		ifBuffer[j] = bvh._faces[ifBuffer[j] - ileaf];
	}

	num = 0u;
	for (i = 0u; i < numL; i++) {
		for (j = 2u; j < numR; j++) {
			if (intersect(aabbs[i], aabbs[j])) {
				ifaces[num].x = ifBuffer[i];
				ifaces[num++].y = ifBuffer[j];
			}
		}
	}
}
__device__ void getLastBvttIfaces(
	BVHParam& clothBvh, BVHParam& obsBvh,
	uint nodeL, uint nodeR,
	uint2* ifaces, uint& num)
{
	uint clothLeaf = clothBvh._size - clothBvh._numFaces;
	uint obsLeaf = obsBvh._size - obsBvh._numFaces;
	bool isLeafL = nodeL >= clothLeaf;
	bool isLeafR = nodeR >= obsLeaf;

	AABB aabbs[4];
	uint ifBuffer[4];
	uint numL, numR;

	if (isLeafL) {
		ifBuffer[0] = nodeL;
		numL = 1u;
	}
	else {
		ifBuffer[0] = (nodeL << 1u) + 1u;
		ifBuffer[1] = ifBuffer[0] + 1u;
		numL = 2u;
	}
	if (isLeafR) {
		ifBuffer[2] = nodeR;
		numR = 3u;
	}
	else {
		ifBuffer[2] = (nodeR << 1u) + 1u;
		ifBuffer[3] = ifBuffer[2] + 1u;
		numR = 4u;
	}

	uint i, j;
	for (i = 0u; i < numL; i++) {
		getBVHAABB(aabbs[i], clothBvh, ifBuffer[i]);
		ifBuffer[i] = clothBvh._faces[ifBuffer[i] - clothLeaf];
	}
	for (j = 2u; j < numR; j++) {
		getBVHAABB(aabbs[j], obsBvh, ifBuffer[j]);
		ifBuffer[j] = obsBvh._faces[ifBuffer[j] - obsLeaf];
	}

	num = 0u;
	for (i = 0u; i < numL; i++) {
		for (j = 2u; j < numR; j++) {
			if (intersect(aabbs[i], aabbs[j])) {
				ifaces[num].x = ifBuffer[i];
				ifaces[num++].y = ifBuffer[j];
			}
		}
	}
}
//-------------------------------------------------------------------------
inline __device__ void CalcInvMat3(REAL* ainv, const REAL* a)
{
	const REAL det =
		+a[0] * a[4] * a[8] + a[3] * a[7] * a[2] + a[6] * a[1] * a[5]
		- a[0] * a[7] * a[5] - a[6] * a[4] * a[2] - a[3] * a[1] * a[8];
	const REAL inv_det = 1.0 / det;

	ainv[0] = inv_det * (a[4] * a[8] - a[5] * a[7]);
	ainv[1] = inv_det * (a[2] * a[7] - a[1] * a[8]);
	ainv[2] = inv_det * (a[1] * a[5] - a[2] * a[4]);

	ainv[3] = inv_det * (a[5] * a[6] - a[3] * a[8]);
	ainv[4] = inv_det * (a[0] * a[8] - a[2] * a[6]);
	ainv[5] = inv_det * (a[2] * a[3] - a[0] * a[5]);

	ainv[6] = inv_det * (a[3] * a[7] - a[4] * a[6]);
	ainv[7] = inv_det * (a[1] * a[6] - a[0] * a[7]);
	ainv[8] = inv_det * (a[0] * a[4] - a[1] * a[3]);
}
//-------------------------------------------------------------------------

#endif