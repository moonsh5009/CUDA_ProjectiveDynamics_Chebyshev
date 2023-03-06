#ifndef __COLLISION_RESPONSE_CUH__
#define __COLLISION_RESPONSE_CUH__

#pragma once
#include "CollisionManager.cuh"

//-------------------------------------------------------------------------
__device__ bool resolveClothBoundaryCollision_device(
	uint i,
	const ObjParam clothParam, const AABB boundary,
	REAL* impulses, REAL* infos, 
	const REAL friction, const REAL delta, const REAL dt)
{
	bool apply = false;
	REAL invM = clothParam._invMs[i];
	if (invM > 0.0) {
		REAL minb[3], maxb[3], p[3], vel[3], vel0[3];
		minb[0] = boundary._min.x; minb[1] = boundary._min.y; minb[2] = boundary._min.z;
		maxb[0] = boundary._max.x; maxb[1] = boundary._max.y; maxb[2] = boundary._max.z;

		p[0] = clothParam._ns[i * 3u + 0u];
		p[1] = clothParam._ns[i * 3u + 1u];
		p[2] = clothParam._ns[i * 3u + 2u];
		vel[0] = clothParam._vs[i * 3u + 0u];
		vel[1] = clothParam._vs[i * 3u + 1u];
		vel[2] = clothParam._vs[i * 3u + 2u];
		vel0[0] = vel[0];
		vel0[1] = vel[1];
		vel0[2] = vel[2];

		REAL N[3], dist, newDist;
		N[0] = N[1] = N[2] = 0.0;
		for (uint j = 0u; j < 3u; j++) {
			N[j] = 1.0;
			dist = p[j] - minb[j] - delta;
			newDist = dist + vel[j] * dt;
			if (newDist < 0.0) {
				REAL A_newVelN = vel[j] - newDist / dt;
				REAL3 relVelT = make_REAL3(vel[0] - N[0] * vel[j], vel[1] - N[1] * vel[j], vel[2] - N[2] * vel[j]);
				REAL lrelVelT = Length(relVelT);
				REAL3 newRelVelT = make_REAL3(0.0);
				if (lrelVelT) {
					newRelVelT = relVelT * max(0.0, 1.0 - friction * (-newDist / dt) / Length(relVelT));
				}
				vel[0] = (N[0] * A_newVelN + newRelVelT.x);
				vel[1] = (N[1] * A_newVelN + newRelVelT.y);
				vel[2] = (N[2] * A_newVelN + newRelVelT.z);
				apply = true;
			}
			N[j] = -1.0;
			dist = maxb[j] - p[j] - delta;
			newDist = dist - vel[j] * dt;
			if (newDist < 0.0) {
				REAL A_newVelN = -vel[j] - newDist / dt;
				REAL3 relVelT = make_REAL3(vel[0] + N[0] * vel[j], vel[1] + N[1] * vel[j], vel[2] + N[2] * vel[j]);
				REAL lrelVelT = Length(relVelT);
				REAL3 newRelVelT = make_REAL3(0.0);
				if (lrelVelT) {
					newRelVelT = relVelT * max(0.0, 1.0 - friction * (-newDist / dt) / Length(relVelT));
				}
				vel[0] = (N[0] * A_newVelN + newRelVelT.x);
				vel[1] = (N[1] * A_newVelN + newRelVelT.y);
				vel[2] = (N[2] * A_newVelN + newRelVelT.z);
				apply = true;
			}
			N[j] = 0.0;
		}
		if (apply) {
			atomicAdd_REAL(impulses + i * 3u + 0u, vel[0] - vel0[0]);
			atomicAdd_REAL(impulses + i * 3u + 1u, vel[1] - vel0[1]);
			atomicAdd_REAL(impulses + i * 3u + 2u, vel[2] - vel0[2]);
			atomicAdd_REAL(infos + i, 1.0);
		}
	}
	return apply;
}
__device__ bool resolveClothCollisionSDF_T_device(
	uint* inos, const REAL3* ps, const REAL3* vs, REAL* invMs,
	REAL dist, REAL w0, REAL w1, const REAL3& norm,
	REAL* impulses, REAL* infos,
	const REAL thickness, const REAL friction, const REAL dt)
{
	bool result = false;
	REAL w2 = 1.0 - w0 - w1;

	REAL iP0 = invMs[0] * w0 * w0;
	REAL iP1 = invMs[1] * w1 * w1;
	REAL iP2 = invMs[2] * w2 * w2;
	REAL iPt = iP0 + iP1 + iP2;
	if (iPt > 0.0) {
		REAL3 relV = vs[0] * w0 + vs[1] * w1 + vs[2] * w2;
		REAL relVN = Dot(relV, norm);
		if (dist < 0.0) dist = 0.0;

		REAL imp = (thickness * COL_THICKNESS_RATIO - dist) / dt - relVN;
		if (imp > 0.0) {
			REAL3 impulse = imp * norm;
			REAL3 relVT = relV - relVN * norm;
			REAL lrelVT = Length(relVT);
			if (lrelVT)
				impulse -= min(friction * imp / lrelVT, 1.0) * relVT;
			impulse *= 1.0 / iPt;

			REAL3 tmp;
			REAL imp_v;
			imp_v = w0 * invMs[0];
			if (imp_v > 0.0) {
				tmp = imp_v * impulse;
				atomicAdd_REAL(impulses + inos[0] * 3u + 0u, tmp.x);
				atomicAdd_REAL(impulses + inos[0] * 3u + 1u, tmp.y);
				atomicAdd_REAL(impulses + inos[0] * 3u + 2u, tmp.z);
				atomicAdd_REAL(infos + inos[0], 1.0);
			}
			imp_v = w1 * invMs[1];
			if (imp_v > 0.0) {
				tmp = imp_v * impulse;
				atomicAdd_REAL(impulses + inos[1] * 3u + 0u, tmp.x);
				atomicAdd_REAL(impulses + inos[1] * 3u + 1u, tmp.y);
				atomicAdd_REAL(impulses + inos[1] * 3u + 2u, tmp.z);
				atomicAdd_REAL(infos + inos[1], 1.0);
			}
			imp_v = w2 * invMs[2];
			if (imp_v > 0.0) {
				tmp = imp_v * impulse;
				atomicAdd_REAL(impulses + inos[2] * 3u + 0u, tmp.x);
				atomicAdd_REAL(impulses + inos[2] * 3u + 1u, tmp.y);
				atomicAdd_REAL(impulses + inos[2] * 3u + 2u, tmp.z);
				atomicAdd_REAL(infos + inos[2], 1.0);
			}
			result = true;
		}
	}
	return result;
}
__device__ bool resolveClothCollisionSDF_V_device(
	uint ino, const REAL3& p, const REAL3& v, REAL invM,
	REAL dist, const REAL3& norm,
	REAL* impulses, REAL* infos,
	const REAL thickness, const REAL friction, const REAL dt)
{
	bool result = false;
	if (invM > 0.0) {
		REAL relVN = Dot(v, norm);
		if (dist < 0.0) dist = 0.0;

		REAL imp = (thickness * COL_THICKNESS_RATIO - dist) / dt - relVN;
		if (imp > 0.0) {
			REAL3 impulse = imp * norm;
			REAL3 relVT = v - relVN * norm;
			REAL lrelVT = Length(relVT);
			if (lrelVT)
				impulse -= min(friction * imp / lrelVT, 1.0) * relVT;

			atomicAdd_REAL(impulses + ino * 3u + 0u, impulse.x);
			atomicAdd_REAL(impulses + ino * 3u + 1u, impulse.y);
			atomicAdd_REAL(impulses + ino * 3u + 2u, impulse.z);
			atomicAdd_REAL(infos + ino, 1.0);
			result = true;
		}
	}
	return result;
}
__device__ bool resolveClothCollisionProximity_device(
	bool isFV,
	uint i0, uint i1, uint i2, uint i3,
	const REAL3& p0, const REAL3& p1, const REAL3& p2, const REAL3& p3,
	const REAL3& v0, const REAL3& v1, const REAL3& v2, const REAL3& v3,
	REAL invM0, REAL invM1, REAL invM2, REAL invM3,
	REAL w0, REAL w1, const REAL3& norm,
	REAL* impulses, REAL* infos,
	REAL friction, REAL thickness, REAL dt)
{
	bool result = false;

	REAL3 q0, q1, q2, q3;
	REAL dist, w2, w3;

	if (isFV) {
		w2 = 1.0 - w0 - w1;
		w3 = 1.0;
	}
	else {
		w3 = w1; w1 = w0;
		w0 = 1.0 - w1;
		w2 = 1.0 - w3;
	}
	
	REAL iP0 = invM0 * w0 * w0;
	REAL iP1 = invM1 * w1 * w1;
	REAL iP2 = invM2 * w2 * w2;
	REAL iP3 = invM3 * w3 * w3;
	REAL iPt = iP0 + iP1 + iP2 + iP3;
	if (iPt > 0.0) {
		REAL3 relV;
		REAL relVN;
		if (isFV)
			relV = v3 - v0 * w0 - v1 * w1 - v2 * w2;
		else
			relV = v2 + (v3 - v2) * w3 - v0 - (v1 - v0) * w1;

		dist = Dot(p3 - p0, norm);
		/*if (isFV)
			dist = Dot(p3 - p0 * w0 - p1 * w1 - p2 * w2, norm);
		else
			dist = Dot(p2 + (p3 - p2) * w3 - p0 - (p1 - p0) * w1, norm);*/
		relVN = Dot(relV, norm);

		REAL imp = (thickness * COL_THICKNESS_RATIO - dist) / dt - relVN;
		if (imp > 0.0) {
			REAL3 impulse = imp * norm;
			REAL3 relVT = relV - relVN * norm;
			REAL lrelVT = Length(relVT);
			if (lrelVT)
				impulse -= min(friction * imp / lrelVT, 1.0) * relVT;
			impulse *= 1.0 / iPt;

			REAL imp_v;
			imp_v = w0 * invM0;
			if (imp_v > 0.0) {
				q0 = -imp_v * impulse;
				atomicAdd_REAL(impulses + i0 * 3u + 0u, q0.x);
				atomicAdd_REAL(impulses + i0 * 3u + 1u, q0.y);
				atomicAdd_REAL(impulses + i0 * 3u + 2u, q0.z);
				atomicAdd_REAL(infos + i0, 1.0);
			}
			imp_v = w1 * invM1;
			if (imp_v > 0.0) {
				q1 = -imp_v * impulse;
				atomicAdd_REAL(impulses + i1 * 3u + 0u, q1.x);
				atomicAdd_REAL(impulses + i1 * 3u + 1u, q1.y);
				atomicAdd_REAL(impulses + i1 * 3u + 2u, q1.z);
				atomicAdd_REAL(infos + i1, 1.0);
			}
			imp_v = w2 * invM2;
			if (imp_v > 0.0) {
				if (isFV) imp_v = -imp_v;
				q2 = imp_v * impulse;
				atomicAdd_REAL(impulses + i2 * 3u + 0u, q2.x);
				atomicAdd_REAL(impulses + i2 * 3u + 1u, q2.y);
				atomicAdd_REAL(impulses + i2 * 3u + 2u, q2.z);
				atomicAdd_REAL(infos + i2, 1.0);
			}
			imp_v = w3 * invM3;
			if (imp_v > 0.0) {
				q3 = imp_v * impulse;
				atomicAdd_REAL(impulses + i3 * 3u + 0u, q3.x);
				atomicAdd_REAL(impulses + i3 * 3u + 1u, q3.y);
				atomicAdd_REAL(impulses + i3 * 3u + 2u, q3.z);
				atomicAdd_REAL(infos + i3, 1.0);
			}

			result = true;
		}
	}
	return result;
}
__device__ bool resolveClothCollisionCCD_device(
	bool isFV,
	uint i0, uint i1, uint i2, uint i3,
	const REAL3& p0, const REAL3& p1, const REAL3& p2, const REAL3& p3,
	const REAL3& v0, const REAL3& v1, const REAL3& v2, const REAL3& v3,
	REAL invM0, REAL invM1, REAL invM2, REAL invM3,
	REAL t, REAL3& norm, REAL w0, REAL w1,
	REAL* impulses, REAL* infos,
	REAL thickness, REAL dt)
{
	bool result = false;

	REAL w2, w3;
	REAL dist;

	if (isFV) {
		w2 = 1.0 - w0 - w1;
		w3 = 1.0;
	}
	else {
		w3 = w1; w1 = w0;
		w0 = 1.0 - w1;
		w2 = 1.0 - w3;
	}

	REAL iP0 = invM0 * w0 * w0;
	REAL iP1 = invM1 * w1 * w1;
	REAL iP2 = invM2 * w2 * w2;
	REAL iP3 = invM3 * w3 * w3;
	REAL iPt = iP0 + iP1 + iP2 + iP3;
	if (iPt > 0.0) {
		if (t < 1.0) {
			if (isFV)
				norm = p3 - p0 * w0 - p1 * w1 - p2 * w2;
			else
				norm = p2 + (p3 - p2) * w3 - p0 - (p1 - p0) * w1;
			dist = Length(norm);
			if (dist > 1.0e-40) {
				norm *= 1.0 / dist;

				REAL3 relV;
				REAL relVN;
				if (isFV)
					relV = v3 - v0 * w0 - v1 * w1 - v2 * w2;
				else
					relV = v2 + (v3 - v2) * w3 - v0 - (v1 - v0) * w1;
				relVN = Dot(relV, norm);
				/*if (relVN > 0.0) {
					norm.x = -norm.x;
					norm.y = -norm.y;
					norm.z = -norm.z;
					relVN = -relVN;
				}*/

				REAL imp = thickness * COL_THICKNESS_RATIO / dt - relVN * (1.0 - t);

				if (imp > 0.0) {
					REAL3 impulse = imp / iPt * norm;

					REAL3 tmp;
					REAL imp_v;
					imp_v = w0 * invM0;
					if (imp_v > 0.0) {
						tmp = -imp_v * impulse;
						atomicAdd_REAL(impulses + i0 * 3u + 0u, tmp.x);
						atomicAdd_REAL(impulses + i0 * 3u + 1u, tmp.y);
						atomicAdd_REAL(impulses + i0 * 3u + 2u, tmp.z);
						atomicAdd_REAL(infos + i0, 1.0);
					}
					imp_v = w1 * invM1;
					if (imp_v > 0.0) {
						tmp = -imp_v * impulse;
						atomicAdd_REAL(impulses + i1 * 3u + 0u, tmp.x);
						atomicAdd_REAL(impulses + i1 * 3u + 1u, tmp.y);
						atomicAdd_REAL(impulses + i1 * 3u + 2u, tmp.z);
						atomicAdd_REAL(infos + i1, 1.0);
					}
					imp_v = w2 * invM2;
					if (imp_v > 0.0) {
						if (isFV) imp_v = -imp_v;
						tmp = imp_v * impulse;
						atomicAdd_REAL(impulses + i2 * 3u + 0u, tmp.x);
						atomicAdd_REAL(impulses + i2 * 3u + 1u, tmp.y);
						atomicAdd_REAL(impulses + i2 * 3u + 2u, tmp.z);
						atomicAdd_REAL(infos + i2, 1.0);
					}
					imp_v = w3 * invM3;
					if (imp_v > 0.0) {
						tmp = imp_v * impulse;
						atomicAdd_REAL(impulses + i3 * 3u + 0u, tmp.x);
						atomicAdd_REAL(impulses + i3 * 3u + 1u, tmp.y);
						atomicAdd_REAL(impulses + i3 * 3u + 2u, tmp.z);
						atomicAdd_REAL(infos + i3, 1.0);
					}

					result = true;
				}
			}
		}
		else {
			REAL3 q0 = p0 + v0 * dt;
			REAL3 q1 = p1 + v1 * dt;
			REAL3 q2 = p2 + v2 * dt;
			REAL3 q3 = p3 + v3 * dt;
			if (isDetected_Proximity(isFV, q0, q1, q2, q3, thickness * COL_THICKNESS_RATIO, &w0, &w1)) {
				if (isFV)
					norm = q3 - q0 * w0 - q1 * w1 - q2 * w2;
				else
					norm = q2 + (q3 - q2) * w3 - q0 - (q1 - q0) * w1;
				dist = Length(norm);
				if (dist > 1.0e-40) {
					REAL3 relV;
					REAL relVN;
					if (isFV)
						relV = v3 - v0 * w0 - v1 * w1 - v2 * w2;
					else
						relV = v2 + (v3 - v2) * w3 - v0 - (v1 - v0) * w1;
					relVN = Dot(relV, norm);

					REAL imp = (thickness * COL_THICKNESS_RATIO - dist) / dt;

					if (imp > 0.0) {
						REAL3 impulse = imp / iPt * norm;

						REAL3 tmp;
						REAL imp_v;
						imp_v = w0 * invM0;
						if (imp_v > 0.0) {
							tmp = -imp_v * impulse;
							atomicAdd_REAL(impulses + i0 * 3u + 0u, tmp.x);
							atomicAdd_REAL(impulses + i0 * 3u + 1u, tmp.y);
							atomicAdd_REAL(impulses + i0 * 3u + 2u, tmp.z);
							atomicAdd_REAL(infos + i0, 1.0);
						}
						imp_v = w1 * invM1;
						if (imp_v > 0.0) {
							tmp = -imp_v * impulse;
							atomicAdd_REAL(impulses + i1 * 3u + 0u, tmp.x);
							atomicAdd_REAL(impulses + i1 * 3u + 1u, tmp.y);
							atomicAdd_REAL(impulses + i1 * 3u + 2u, tmp.z);
							atomicAdd_REAL(infos + i1, 1.0);
						}
						imp_v = w2 * invM2;
						if (imp_v > 0.0) {
							if (isFV) imp_v = -imp_v;
							tmp = imp_v * impulse;
							atomicAdd_REAL(impulses + i2 * 3u + 0u, tmp.x);
							atomicAdd_REAL(impulses + i2 * 3u + 1u, tmp.y);
							atomicAdd_REAL(impulses + i2 * 3u + 2u, tmp.z);
							atomicAdd_REAL(infos + i2, 1.0);
						}
						imp_v = w3 * invM3;
						if (imp_v > 0.0) {
							tmp = imp_v * impulse;
							atomicAdd_REAL(impulses + i3 * 3u + 0u, tmp.x);
							atomicAdd_REAL(impulses + i3 * 3u + 1u, tmp.y);
							atomicAdd_REAL(impulses + i3 * 3u + 2u, tmp.z);
							atomicAdd_REAL(infos + i3, 1.0);
						}

						result = true;
					}
				}
			}
		}
	}


	return result;
}
__device__ bool ClothDetectedProximity_device(
	bool isFV,
	uint i0, uint i1, uint i2, uint i3,
	const REAL3& p0, const REAL3& p1, const REAL3& p2, const REAL3& p3,
	const REAL3& v0, const REAL3& v1, const REAL3& v2, const REAL3& v3,
	REAL w0, REAL w1, const REAL3& norm,
	REAL thickness, REAL dt)
{
	bool result = false;
#if 1
	REAL dist, w2, w3;

	if (isFV) {
		w2 = 1.0 - w0 - w1;
		w3 = 1.0;
	}
	else {
		w3 = w1; w1 = w0;
		w0 = 1.0 - w1;
		w2 = 1.0 - w3;
	}

	REAL3 relV;
	REAL relVN;
	if (isFV)
		relV = v3 - v0 * w0 - v1 * w1 - v2 * w2;
	else
		relV = v2 + (v3 - v2) * w3 - v0 - (v1 - v0) * w1;

	dist = Dot(p3 - p0, norm);
	relVN = Dot(relV, norm);

	REAL imp = (min(thickness * COL_THICKNESS_RATIO, dist) * 0.1 - dist) / dt - relVN;
	if (imp > 0.0)
		result = true;
#else
	REAL t;
	result =
		isDetected_CCD(isFV, p0, p1, p2, p3, p0 + v0 * dt, p1 + v1 * dt, p2 + v2 * dt, p3 + v3 * dt, thickness, &t);
#endif

	return result;
}
//-------------------------------------------------------------------------
__global__ void compClothBoundaryCollisionImpulse_kernel(
	const ObjParam clothParam, const AABB boundary,
	REAL* impulses, REAL* infos,
	const REAL friction, const REAL delta, const REAL dt)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= clothParam._numNodes)
		return;

	resolveClothBoundaryCollision_device(id, clothParam, boundary, impulses, infos, friction, delta, dt);
}
__global__ void compClothCollisionSDFImpulse_T_kernel(
	ContactElemSDFParam ceParam, ObjParam clothParam,
	REAL* impulses, REAL* infos,
	const REAL friction, const REAL thickness, const REAL dt)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= clothParam._numFaces)
		return;

	ContactElemSDF ce = ceParam._felems[id];
	if (ce._dist == REAL_MAX)
		return;

	uint ino[3];
	ino[0] = clothParam._fs[id * 3 + 0];
	ino[1] = clothParam._fs[id * 3 + 1];
	ino[2] = clothParam._fs[id * 3 + 2];
	REAL3 ps[3], vs[3], qs[3];
	REAL invMs[3];
	ps[0] = make_double3(clothParam._ns[ino[0] * 3 + 0], clothParam._ns[ino[0] * 3 + 1], clothParam._ns[ino[0] * 3 + 2]);
	ps[1] = make_double3(clothParam._ns[ino[1] * 3 + 0], clothParam._ns[ino[1] * 3 + 1], clothParam._ns[ino[1] * 3 + 2]);
	ps[2] = make_double3(clothParam._ns[ino[2] * 3 + 0], clothParam._ns[ino[2] * 3 + 1], clothParam._ns[ino[2] * 3 + 2]);
	vs[0] = make_double3(clothParam._vs[ino[0] * 3 + 0], clothParam._vs[ino[0] * 3 + 1], clothParam._vs[ino[0] * 3 + 2]);
	vs[1] = make_double3(clothParam._vs[ino[1] * 3 + 0], clothParam._vs[ino[1] * 3 + 1], clothParam._vs[ino[1] * 3 + 2]);
	vs[2] = make_double3(clothParam._vs[ino[2] * 3 + 0], clothParam._vs[ino[2] * 3 + 1], clothParam._vs[ino[2] * 3 + 2]);
	invMs[0] = clothParam._invMs[ino[0]];
	invMs[1] = clothParam._invMs[ino[1]];
	invMs[2] = clothParam._invMs[ino[2]];

	resolveClothCollisionSDF_T_device(
		ino, ps, vs, invMs,
		ce._dist, ce._w[0], ce._w[1], ce._norm,
		impulses, infos,
		thickness, friction, dt);
}
__global__ void compClothCollisionSDFImpulse_V_kernel(
	ContactElemSDFParam ceParam, ObjParam clothParam,
	REAL* impulses, REAL* infos,
	const REAL friction, const REAL thickness, const REAL dt)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= clothParam._numNodes)
		return;

	ContactElemSDF ce = ceParam._nelems[id];
	if (ce._dist == REAL_MAX)
		return;

	REAL3 p, v;
	REAL invM;
	p = make_double3(clothParam._ns[id * 3 + 0], clothParam._ns[id * 3 + 1], clothParam._ns[id * 3 + 2]);
	v = make_double3(clothParam._vs[id * 3 + 0], clothParam._vs[id * 3 + 1], clothParam._vs[id * 3 + 2]);
	invM = clothParam._invMs[id];

	resolveClothCollisionSDF_V_device(
		id, p, v, invM,
		ce._dist, ce._norm,
		impulses, infos,
		thickness, friction, dt);
}
//-------------------------------------------------------------------------
__global__ void resetCE_kernel(
	ContactElemParam ceParam)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= ceParam._size)
		return;

	ContactElem ce = ceParam._elems[id];
	ce._info = 0.0;
	ceParam._elems[id] = ce;
}
__global__ void compClothDetected_CE_kernel(
	ContactElemParam ceParam,
	ObjParam clothParam, ObjParam obsParam,
	const REAL thickness, const REAL dt,
	bool* isApplied)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= ceParam._size)
		return;

	ContactElem ce = ceParam._elems[id];

	uint ino;
	REAL3 ps[4], vs[4];
	REAL invMs[4];
	ObjParam* param;
	for (uint i = 0u; i < 4u; i++) {
		if (!ce._isObs)
			param = &clothParam;
		else if (ce._isObs == 1u) {
			if (ce._isFV) {
				if (i < 3u)
					param = &clothParam;
				else
					param = &obsParam;
			}
			else {
				if (i < 2u)
					param = &clothParam;
				else
					param = &obsParam;
			}
		}
		else {
			if (ce._isFV) {
				if (i < 3u)
					param = &obsParam;
				else
					param = &clothParam;
			}
			else {
				if (i < 2u)
					param = &obsParam;
				else
					param = &clothParam;
			}
		}
		ino = ce._i[i] * 3u;
		ps[i].x = param->_ns[ino + 0u];
		ps[i].y = param->_ns[ino + 1u];
		ps[i].z = param->_ns[ino + 2u];
		vs[i].x = param->_vs[ino + 0u];
		vs[i].y = param->_vs[ino + 1u];
		vs[i].z = param->_vs[ino + 2u];
	}

	if (ClothDetectedProximity_device(
		ce._isFV, ce._i[0], ce._i[1], ce._i[2], ce._i[3],
		ps[0], ps[1], ps[2], ps[3],
		vs[0], vs[1], vs[2], vs[3],
		ce._w[0], ce._w[1], ce._norm, thickness, dt))
	{
		ce._info = 1.0;
		ceParam._elems[id] = ce;
		*isApplied = true;
	}
}
__global__ void compClothCollisionImpulse_CE_kernel(
	ContactElemParam ceParam,
	ObjParam clothParam, ObjParam obsParam,
	REAL* impulses, REAL* infos,
	const REAL friction, const REAL thickness, const REAL dt)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= ceParam._size)
		return;

	ContactElem ce = ceParam._elems[id];

	uint ino;
	REAL3 ps[4], vs[4];
	REAL invMs[4];
	ObjParam* param;
	for (uint i = 0u; i < 4u; i++) {
		if (!ce._isObs)
			param = &clothParam;
		else if (ce._isObs == 1u) {
			if (ce._isFV) {
				if (i < 3u)
					param = &clothParam;
				else
					param = &obsParam;
			}
			else {
				if (i < 2u)
					param = &clothParam;
				else
					param = &obsParam;
			}
		}
		else {
			if (ce._isFV) {
				if (i < 3u)
					param = &obsParam;
				else
					param = &clothParam;
			}
			else {
				if (i < 2u)
					param = &obsParam;
				else
					param = &clothParam;
			}
		}
		ino = ce._i[i] * 3u;
		ps[i].x = param->_ns[ino + 0u];
		ps[i].y = param->_ns[ino + 1u];
		ps[i].z = param->_ns[ino + 2u];
		vs[i].x = param->_vs[ino + 0u];
		vs[i].y = param->_vs[ino + 1u];
		vs[i].z = param->_vs[ino + 2u];
		if (param == &clothParam)
			invMs[i] = param->_invMs[ce._i[i]];
		else
			invMs[i] = 0.0;
	}

	resolveClothCollisionProximity_device(
		ce._isFV, ce._i[0], ce._i[1], ce._i[2], ce._i[3],
		ps[0], ps[1], ps[2], ps[3],
		vs[0], vs[1], vs[2], vs[3],
		invMs[0], invMs[1], invMs[2], invMs[3],
		ce._w[0], ce._w[1], ce._norm,
		impulses, infos, friction, thickness, dt);
}
//-------------------------------------------------------------------------
__device__ void compSelfCollisionCCDImpulse_device(
	const uint lRTri, const uint rRTri,
	const uint* ino, const REAL3* pi, const REAL3* vi, const REAL* invMi,
	const uint* jno, const REAL3* pj, const REAL3* vj, const REAL* invMj,
	REAL* impulses, REAL* infos, 
	REAL thickness, REAL dt, bool& isApplied)
{
	uint i, j, i1, j1;
	REAL3 norm;
	REAL t, w0, w1;
	REAL3 qi[3], qj[3];
	for (i = 0u; i < 3u; i++) {
		qi[i] = pi[i] + vi[i] * dt;
		qj[i] = pj[i] + vj[i] * dt;
	}
	for (i = 0u; i < 3u; i++) {
		if (RTriVertex(rRTri, i)) {
			if (Culling_Index(true, ino[0], ino[1], ino[2], jno[i]))
				if (isDetected_CCD(true, pi[0], pi[1], pi[2], pj[i], qi[0], qi[1], qi[2], qj[i], thickness, &t, &norm, &w0, &w1))
					if (Culling_barycentricRTri(w0, w1, lRTri)) {
						resolveClothCollisionCCD_device(
							true, ino[0], ino[1], ino[2], jno[i],
							pi[0], pi[1], pi[2], pj[i],
							vi[0], vi[1], vi[2], vj[i],
							invMi[0], invMi[1], invMi[2], invMj[i],
							t, norm, w0, w1, impulses, infos, thickness, dt);
						isApplied = true;
					}
		}
		if (RTriVertex(lRTri, i)) {
			if (Culling_Index(true, jno[0], jno[1], jno[2], ino[i]))
				if (isDetected_CCD(true, pj[0], pj[1], pj[2], pi[i], qj[0], qj[1], qj[2], qi[i], thickness, &t, &norm, &w0, &w1))
					if (Culling_barycentricRTri(w0, w1, rRTri)) {
						resolveClothCollisionCCD_device(
							true, jno[0], jno[1], jno[2], ino[i],
							pj[0], pj[1], pj[2], pi[i],
							vj[0], vj[1], vj[2], vi[i],
							invMj[0], invMj[1], invMj[2], invMi[i],
							t, norm, w0, w1, impulses, infos, thickness, dt);
						isApplied = true;
					}
		}

		i1 = (i + 1u) % 3u;
		if (RTriEdge(lRTri, i)) {
			for (j = 0u; j < 3u; j++) {
				j1 = (j + 1u) % 3u;
				if (RTriEdge(rRTri, j)) {
					if (Culling_Index(false, ino[i], ino[i1], jno[j], jno[j1]))
						if (isDetected_CCD(false, pi[i], pi[i1], pj[j], pj[j1], qi[i], qi[i1], qj[j], qj[j1], thickness, &t, &norm, &w0, &w1))
							if (Culling_barycentricRTri(w0, w1, lRTri, rRTri, i, i1, j, j1)){
								resolveClothCollisionCCD_device(
									true, ino[i], ino[i1], jno[j], jno[j1],
									pi[i], pi[i1], pj[j], pj[j1], 
									vi[i], vi[i1], vj[j], vj[j1],
									invMi[i], invMi[i1], invMj[j], invMj[j1],
									t, norm, w0, w1, impulses, infos, thickness, dt);
								isApplied = true;
							}
				}
			}
		}
	}
}
__device__ void compObstacleCollisionCCDImpulse_device(
	const uint lRTri, const uint rRTri,
	const uint* ino, const REAL3* pi, const REAL3* vi, const REAL* invMi,
	const uint* jno, const REAL3* pj, const REAL3* vj, const REAL* invMj,
	REAL* impulses, REAL* infos,
	REAL thickness, REAL dt)
{
	uint i, j, i1, j1;
	REAL3 norm;
	REAL t, w0, w1;
	REAL3 qi[3], qj[3];
	for (i = 0u; i < 3u; i++) {
		qi[i] = pi[i] + vi[i] * dt;
		qj[i] = pj[i] + vj[i] * dt;
	}
	for (i = 0u; i < 3u; i++) {
		if (RTriVertex(rRTri, i)) {
			if (isDetected_CCD(true, pi[0], pi[1], pi[2], pj[i], qi[0], qi[1], qi[2], qj[i], thickness, &t, &norm, &w0, &w1))
				if (Culling_barycentricRTri(w0, w1, lRTri))
					resolveClothCollisionCCD_device(
						true, ino[0], ino[1], ino[2], jno[i],
						pi[0], pi[1], pi[2], pj[i],
						vi[0], vi[1], vi[2], vj[i],
						invMi[0], invMi[1], invMi[2], invMj[i],
						t, norm, w0, w1, impulses, infos, thickness, dt);
		}
		if (RTriVertex(lRTri, i)) {
			if (isDetected_CCD(true, pj[0], pj[1], pj[2], pi[i], qj[0], qj[1], qj[2], qi[i], thickness, &t, &norm, &w0, &w1))
				if (Culling_barycentricRTri(w0, w1, rRTri))
					resolveClothCollisionCCD_device(
						true, jno[0], jno[1], jno[2], ino[i],
						pj[0], pj[1], pj[2], pi[i],
						vj[0], vj[1], vj[2], vi[i],
						invMj[0], invMj[1], invMj[2], invMi[i],
						t, norm, w0, w1, impulses, infos, thickness, dt);
		}

		i1 = (i + 1u) % 3u;
		if (RTriEdge(lRTri, i)) {
			for (j = 0u; j < 3u; j++) {
				j1 = (j + 1u) % 3u;
				if (RTriEdge(rRTri, j)) {
					if (isDetected_CCD(false, pi[i], pi[i1], pj[j], pj[j1], qi[i], qi[i1], qj[j], qj[j1], thickness, &t, &norm, &w0, &w1))
						if (Culling_barycentricRTri(w0, w1, lRTri, rRTri, i, i1, j, j1))
							resolveClothCollisionCCD_device(
								true, ino[i], ino[i1], jno[j], jno[j1],
								pi[i], pi[i1], pj[j], pj[j1],
								vi[i], vi[i1], vj[j], vj[j1],
								invMi[i], invMi[i1], invMj[j], invMj[j1],
								t, norm, w0, w1, impulses, infos, thickness, dt);
				}
			}
		}
	}
}
__global__ void compSelfCollisionCCDImpulse_LastBvtt_kernel(
	ObjParam clothParam, BVHParam clothBvh, RTriParam clothRTri,
	uint2* lastBvtts, uint lastBvttSize,
	REAL* impulses, REAL* infos,
	const REAL thickness, const REAL dt)
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
	REAL3 pis[3], pjs[3], vis[3], vjs[3];
	REAL invMis[3], invMjs[3];

	ContactElem ces[15];
	uint ceSize;
	for (ino = 0u; ino < fnum; ino++) {
		lRTri = clothRTri._info[ifaces[ino].x];
		rRTri = clothRTri._info[ifaces[ino].y];
		getMeshElements_device(clothParam, ifaces[ino].x, ifaces[ino].y, inos, jnos, pis, pjs, vis, vjs, invMis, invMjs);
		bool chk = false;
		compSelfCollisionCCDImpulse_device(
			lRTri, rRTri,
			inos, pis, vis, invMis,
			jnos, pjs, vjs, invMjs,
			impulses, infos, thickness, dt, chk);
		/*if (chk) {
			clothBvh._isDetecteds[ifaces[ino].x] = true;
			clothBvh._isDetecteds[ifaces[ino].y] = true;
		}*/
	}
}
__global__ void compObstacleCollisionCCDImpulse_LastBvtt_kernel(
	ObjParam clothParam, BVHParam clothBvh, RTriParam clothRTri,
	ObjParam obsParam, BVHParam obsBvh, RTriParam obsRTri,
	uint2* lastBvtts, uint lastBvttSize,
	REAL* impulses, REAL* infos,
	const REAL thickness, const REAL dt)
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
	REAL3 pis[3], pjs[3], vis[3], vjs[3];
	REAL invMis[3], invMjs[3];

	ContactElem ces[15];
	uint ceSize;
	for (ino = 0u; ino < fnum; ino++) {
		lRTri = clothRTri._info[ifaces[ino].x];
		rRTri = obsRTri._info[ifaces[ino].y];
		getMeshElements_device(clothParam, obsParam, ifaces[ino].x, ifaces[ino].y, inos, jnos, pis, pjs, vis, vjs, invMis, invMjs);
		compObstacleCollisionCCDImpulse_device(
			lRTri, rRTri,
			inos, pis, vis, invMis,
			jnos, pjs, vjs, invMjs,
			impulses, infos, thickness, dt);
	}
}
//-------------------------------------------------------------------------
__global__ void applyClothCollision_kernel(
	ObjParam clothParam,
	REAL* impulses, REAL* infos,
	REAL thickness, REAL dt, REAL omg, bool* isApplied)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= clothParam._numNodes)
		return;

	REAL num = infos[id];
	if (num > 0.0) {
		uint ino = id * 3u;
		REAL3 impulse;
		impulse.x = impulses[ino + 0u];
		impulse.y = impulses[ino + 1u];
		impulse.z = impulses[ino + 2u];
		impulse *= 1.0 / num;

		/*REAL3 v, prevV;
		prevV.x = prevVs[ino + 0u];
		prevV.y = prevVs[ino + 1u];
		prevV.z = prevVs[ino + 2u];
		v.x = clothParam._vs[ino + 0u];
		v.y = clothParam._vs[ino + 1u];
		v.z = clothParam._vs[ino + 2u];
		prevVs[ino + 0u] = v.x;
		prevVs[ino + 1u] = v.y;
		prevVs[ino + 2u] = v.z;*/

		//if (Length(impulse) * dt > thickness * COL_THICKNESS_RATIO * 1.0e-4) {
			*isApplied = true;

			//v = (v + impulse - prevV) * omg + prevV;

			REAL3 v;
			v.x = clothParam._vs[ino + 0u];
			v.y = clothParam._vs[ino + 1u];
			v.z = clothParam._vs[ino + 2u];
			v += impulse;

			clothParam._vs[ino + 0u] = v.x;
			clothParam._vs[ino + 1u] = v.y;
			clothParam._vs[ino + 2u] = v.z;
		//}
	}
}
//-------------------------------------------------------------------------
__global__ void jacobiProject0_kernel(ObjParam clothParam, REAL* Zs, REAL* newNs, REAL invdt2)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= clothParam._numNodes)
		return;

	uint ino = id * 3u;
	REAL m = clothParam._ms[id];
	REAL3 n0;
	n0.x = Zs[ino + 0u];
	n0.y = Zs[ino + 1u];
	n0.z = Zs[ino + 2u];

	n0 *= m * invdt2;

	newNs[ino + 0u] = n0.x;
	newNs[ino + 1u] = n0.y;
	newNs[ino + 2u] = n0.z;
}
__global__ void jacobiProject1_kernel(REAL* Xs, REAL* newNs, uint* inos, REAL* ws, REAL* restLengths, uint numSprings) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numSprings)
		return;

	uint ino = id << 1u;
	uint ino0 = inos[ino + 0u];
	uint ino1 = inos[ino + 1u];

	REAL w = ws[id];

	ino0 *= 3u; ino1 *= 3u;
	REAL3 n0, n1;
	n0.x = Xs[ino0 + 0u]; n0.y = Xs[ino0 + 1u]; n0.z = Xs[ino0 + 2u];
	n1.x = Xs[ino1 + 0u]; n1.y = Xs[ino1 + 1u]; n1.z = Xs[ino1 + 2u];

	REAL3 error0 = make_REAL3(0.0);
	REAL3 error1 = make_REAL3(0.0);

	error0 += w * n1;
	error1 += w * n0;

	REAL3 d = n0 - n1;
	REAL restLength = restLengths[id];
	REAL length = Length(d);
	if (length) {
		//REAL3 newL = w * restLength / length;
		d *= w * restLength / length;
		error0 += d;
		error1 -= d;
	}

	atomicAdd_REAL(newNs + ino0 + 0u, error0.x);
	atomicAdd_REAL(newNs + ino0 + 1u, error0.y);
	atomicAdd_REAL(newNs + ino0 + 2u, error0.z);
	atomicAdd_REAL(newNs + ino1 + 0u, error1.x);
	atomicAdd_REAL(newNs + ino1 + 1u, error1.y);
	atomicAdd_REAL(newNs + ino1 + 2u, error1.z);
}
__global__ void jacobiProject2_kernel(
	ObjParam clothParam, REAL* Xs, REAL* prevNs, REAL* Bs, REAL* newNs, REAL invdt2,
	REAL underRelax, REAL omega, REAL* maxError)
{
	extern __shared__ REAL s_maxError[];
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	uint ino;

	s_maxError[threadIdx.x] = 0.0;
	if (id < clothParam._numNodes) {
		REAL m = clothParam._ms[id];
		REAL b = Bs[id];
		if (m > 0.0) {
			ino = id * 3u;

			REAL3 n, prevN, newN;
			n.x = Xs[ino + 0u];
			n.y = Xs[ino + 1u];
			n.z = Xs[ino + 2u];

			if (b > 0.0) {
				prevN.x = prevNs[ino + 0u];
				prevN.y = prevNs[ino + 1u];
				prevN.z = prevNs[ino + 2u];
				newN.x = newNs[ino + 0u];
				newN.y = newNs[ino + 1u];
				newN.z = newNs[ino + 2u];

				newN *= 1.0 / (b + m * invdt2);
				newN = omega * (underRelax * (newN - n) + n - prevN) + prevN;

				Xs[ino + 0u] = newN.x;
				Xs[ino + 1u] = newN.y;
				Xs[ino + 2u] = newN.z;

				s_maxError[threadIdx.x] = Length(newN - n);

				REAL3 n0;
				n0.x = clothParam._ns[ino + 0u];
				n0.y = clothParam._ns[ino + 1u];
				n0.z = clothParam._ns[ino + 2u];
				clothParam._vs[ino + 0u] = (newN.x - n0.x) * sqrt(invdt2);
				clothParam._vs[ino + 1u] = (newN.y - n0.y) * sqrt(invdt2);
				clothParam._vs[ino + 2u] = (newN.z - n0.z) * sqrt(invdt2);
			}
			prevNs[ino + 0u] = n.x;
			prevNs[ino + 1u] = n.y;
			prevNs[ino + 2u] = n.z;
		}
	}
	for (ino = blockDim.x >> 1u; ino > 32u; ino >>= 1u) {
		__syncthreads();
		if (threadIdx.x < ino)
			if (s_maxError[threadIdx.x] < s_maxError[threadIdx.x + ino])
				s_maxError[threadIdx.x] = s_maxError[threadIdx.x + ino];
	}
	__syncthreads();
	if (threadIdx.x < 32u) {
		warpMax(s_maxError, threadIdx.x);
		if (threadIdx.x == 0u)
			atomicMax_REAL(maxError, s_maxError[0]);
	}
}
__global__ void updatePosition_kernel(
	ObjParam objParam, REAL dt)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < objParam._numNodes) {
		uint ino = id * 3u;
		REAL3 n, v;
		n.x = objParam._ns[ino + 0u];
		n.y = objParam._ns[ino + 1u];
		n.z = objParam._ns[ino + 2u];
		v.x = objParam._vs[ino + 0u];
		v.y = objParam._vs[ino + 1u];
		v.z = objParam._vs[ino + 2u];

		n += v * dt;

		objParam._ns[ino + 0u] = n.x;
		objParam._ns[ino + 1u] = n.y;
		objParam._ns[ino + 2u] = n.z;
	}
}
__global__ void updateVelocity_kernel(
	ObjParam objParam, REAL invdt)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < objParam._numNodes) {
		uint ino = id * 3u;
		REAL3 n0, n, v;
		n0.x = objParam._n0s[ino + 0u];
		n0.y = objParam._n0s[ino + 1u];
		n0.z = objParam._n0s[ino + 2u];
		n.x = objParam._ns[ino + 0u];
		n.y = objParam._ns[ino + 1u];
		n.z = objParam._ns[ino + 2u];

		v = (n - n0) * invdt;

		objParam._vs[ino + 0u] = v.x;
		objParam._vs[ino + 1u] = v.y;
		objParam._vs[ino + 2u] = v.z;
		objParam._ns[ino + 0u] = n0.x;
		objParam._ns[ino + 1u] = n0.y;
		objParam._ns[ino + 2u] = n0.z;
	}
}
//-------------------------------------------------------------------------
__global__ void ApplyClothRigidImpactZone_kernel(
	ObjParam clothParam, ObjParam obsParam,
	RIZoneParam riz, const REAL dt)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= riz._size)
		return;

	uint istart = riz._zones[id];
	uint iend = riz._zones[id + 1];
	uint i;
	uint2 ino;

	REAL3 gc = make_REAL3(0.0);
	REAL3 av = make_REAL3(0.0);
	REAL3 p, v, q;
	REAL tmp, invM, mass;
	tmp = 0.0;
	for (i = istart; i < iend; i++) {
		ino = riz._ids[i];
		if (!ino.y) {
			mass = clothParam._ms[ino.x]; ino.x *= 3u;
			p.x = clothParam._ns[ino.x + 0u]; p.y = clothParam._ns[ino.x + 1u]; p.z = clothParam._ns[ino.x + 2u];
			v.x = clothParam._vs[ino.x + 0u]; v.y = clothParam._vs[ino.x + 1u]; v.z = clothParam._vs[ino.x + 2u];
		}
		else {
			mass = 0.0; ino.x *= 3u;
			p.x = obsParam._ns[ino.x + 0u]; p.y = obsParam._ns[ino.x + 1u]; p.z = obsParam._ns[ino.x + 2u];
			v.x = obsParam._vs[ino.x + 0u]; v.y = obsParam._vs[ino.x + 1u]; v.z = obsParam._vs[ino.x + 2u];
		}
		if (mass == 0.0) mass = 1.0e+10;
		//mass = 1.0;

		gc += p * mass;
		av += v * mass;
		tmp += mass;
	}
	tmp = 1.0 / tmp;
	gc *= tmp;
	av *= tmp;

	REAL3 L = make_REAL3(0.0);
	REAL I[9] = { 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0 };
	for (i = istart; i < iend; i++) {
		ino = riz._ids[i];
		if (!ino.y) {
			mass = clothParam._ms[ino.x]; ino.x *= 3u;
			p.x = clothParam._ns[ino.x + 0u]; p.y = clothParam._ns[ino.x + 1u]; p.z = clothParam._ns[ino.x + 2u];
			v.x = clothParam._vs[ino.x + 0u]; v.y = clothParam._vs[ino.x + 1u]; v.z = clothParam._vs[ino.x + 2u];
		}
		else {
			mass = 0.0; ino.x *= 3u;
			p.x = obsParam._ns[ino.x + 0u]; p.y = obsParam._ns[ino.x + 1u]; p.z = obsParam._ns[ino.x + 2u];
			v.x = obsParam._vs[ino.x + 0u]; v.y = obsParam._vs[ino.x + 1u]; v.z = obsParam._vs[ino.x + 2u];
		}
		if (mass == 0.0) mass = 1.0e+10;
		//mass = 1.0;

		q = p - gc;
		L += mass * Cross(q, v - av);

		tmp = Dot(q, q);
		/*I[0] += tmp - q.x * q.x;	I[1] += -q.x * q.y;			I[2] += -q.x * q.z;
		I[3] += -q.y * q.x;			I[4] += tmp - q.y * q.y;	I[5] += -q.y * q.z;
		I[6] += -q.z * q.x;			I[7] += -q.z * q.y;			I[8] += tmp - q.z * q.z;*/
		I[0] += mass * (tmp - q.x * q.x);	I[1] += mass * (-q.x * q.y);		I[2] += mass * (-q.x * q.z);
		I[3] += mass * (-q.y * q.x);		I[4] += mass * (tmp - q.y * q.y);	I[5] += mass * (-q.y * q.z);
		I[6] += mass * (-q.z * q.x);		I[7] += mass * (-q.z * q.y);		I[8] += mass * (tmp - q.z * q.z);
	}
	REAL Iinv[9];
	CalcInvMat3(Iinv, I);
	REAL3 omg;
	omg.x = Iinv[0] * L.x + Iinv[1] * L.y + Iinv[2] * L.z;
	omg.y = Iinv[3] * L.x + Iinv[4] * L.y + Iinv[5] * L.z;
	omg.z = Iinv[6] * L.x + Iinv[7] * L.y + Iinv[8] * L.z;

	REAL lomg = Length(omg);
	if (lomg) {
		omg *= 1.0 / lomg;
		REAL3 xf, xr, px;
		REAL comg = cos(dt * lomg);
		REAL somg = sin(dt * lomg);
		for (i = istart; i < iend; i++) {
			ino = riz._ids[i];
			if (!ino.y) {
				invM = clothParam._invMs[ino.x];
				if (invM) {
					ino.x *= 3u;
					p.x = clothParam._ns[ino.x + 0u]; p.y = clothParam._ns[ino.x + 1u]; p.z = clothParam._ns[ino.x + 2u];

					q = p - gc;
					xf = Dot(q, omg) * omg;
					xr = q - xf;
					px = gc + dt * av + xf + comg * xr + Cross(somg * omg, xr);
					v = (px - p) / dt;

					/*q = Cross(p - gc, omg);
					v = av - q;*/

					clothParam._vs[ino.x + 0u] = v.x;
					clothParam._vs[ino.x + 1u] = v.y;
					clothParam._vs[ino.x + 2u] = v.z;
				}
			}
		}
	}
}

#endif