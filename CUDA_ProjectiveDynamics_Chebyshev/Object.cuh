#include "Object.h"
#include "../include/CUDA_Custom/DeviceManager.cuh"

__global__ void compGravityForce_kernel(REAL* forces, REAL* ms, REAL3 gravity, uint numNodes) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numNodes)
		return;

	REAL m = ms[id];
	id *= 3u;

	REAL3 force;
	force.x = forces[id + 0u];
	force.y = forces[id + 1u];
	force.z = forces[id + 2u];

	force += m * gravity;
	forces[id + 0u] = force.x;
	forces[id + 1u] = force.y;
	forces[id + 2u] = force.z;
}

__global__ void compRotationForce_kernel(
	REAL* ns, REAL* forces, REAL* ms, REAL3* pivots, REAL3* degrees, uint* nodePhases, REAL invdt2, uint numNodes) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numNodes)
		return;

	uint phase = nodePhases[id];
	REAL3 pivot = pivots[phase];
	REAL3 degree = degrees[phase];

	REAL m = ms[id];
	id *= 3u;

	REAL3 force;
	force.x = forces[id + 0u];
	force.y = forces[id + 1u];
	force.z = forces[id + 2u];

	degree.x *= M_PI * 0.00555555555555555555555555555556;
	degree.y *= M_PI * 0.00555555555555555555555555555556;
	degree.z *= M_PI * 0.00555555555555555555555555555556;

	REAL cx = cos(degree.x);
	REAL sx = sin(degree.x);
	REAL cy = cos(degree.y);
	REAL sy = -sin(degree.y);
	REAL cz = cos(degree.z);
	REAL sz = sin(degree.z);

	REAL3 n, pn;
	n.x = ns[id + 0u];
	n.y = ns[id + 1u];
	n.z = ns[id + 2u];
	n -= pivot;

	pn.x = n.x * cz * cy + n.y * (cz * sy * sx - sz * cx) + n.z * (cz * sy * cx + sz * sx);
	pn.y = n.x * sz * cy + n.y * (sz * sy * sx + cz * cx) + n.z * (sz * sy * cx - cz * sx);
	pn.z = n.x * -sy + n.y * cy * sx + n.z * cy * cx;

	force += m * invdt2 * (pn - n);
	forces[id + 0u] = force.x;
	forces[id + 1u] = force.y;
	forces[id + 2u] = force.z;
}

__global__ void compPredictPosition_kernel(REAL* ns, REAL* vs, REAL* forces, REAL* invMs, REAL dt, uint numNodes) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numNodes)
		return;

	REAL invM = invMs[id];
	id *= 3u;

	REAL3 n, v, force;
	n.x = ns[id + 0u];
	n.y = ns[id + 1u];
	n.z = ns[id + 2u];
	v.x = vs[id + 0u];
	v.y = vs[id + 1u];
	v.z = vs[id + 2u];
	force.x = forces[id + 0u];
	force.y = forces[id + 1u];
	force.z = forces[id + 2u];

	v += invM * dt * force; n += dt * v;
	ns[id + 0u] = n.x;
	ns[id + 1u] = n.y;
	ns[id + 2u] = n.z;
	vs[id + 0u] = v.x;
	vs[id + 1u] = v.y;
	vs[id + 2u] = v.z;
}
__global__ void updateVelocitiy_kernel(REAL* ns, REAL* n0s, REAL* vs, REAL invdt, uint numNodes) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numNodes)
		return;

	id *= 3u;
	REAL3 n, n0, v;
	n.x = ns[id + 0u];
	n.y = ns[id + 1u];
	n.z = ns[id + 2u];
	n0.x = n0s[id + 0u];
	n0.y = n0s[id + 1u];
	n0.z = n0s[id + 2u];

	v = (n - n0) * invdt;
	vs[id + 0u] = v.x;
	vs[id + 1u] = v.y;
	vs[id + 2u] = v.z;
}
__global__ void updatePosition_kernel(REAL* ns, REAL* vs, REAL dt, uint numNodes) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numNodes)
		return;

	id *= 3u;
	REAL3 n, v;
	n.x = ns[id + 0u];
	n.y = ns[id + 1u];
	n.z = ns[id + 2u];
	v.x = vs[id + 0u];
	v.y = vs[id + 1u];
	v.z = vs[id + 2u];
	n += v * dt;

	ns[id + 0u] = n.x;
	ns[id + 1u] = n.y;
	ns[id + 2u] = n.z;
}

__global__ void compNormals_kernel(uint* fs, REAL* ns, REAL* fNorms, REAL* nNorms, uint numFaces) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numFaces)
		return;

	id *= 3u;
	uint iv0 = fs[id + 0u]; iv0 *= 3u;
	uint iv1 = fs[id + 1u]; iv1 *= 3u;
	uint iv2 = fs[id + 2u]; iv2 *= 3u;

	REAL3 v0, v1, v2;
	v0.x = ns[iv0 + 0u]; v0.y = ns[iv0 + 1u]; v0.z = ns[iv0 + 2u];
	v1.x = ns[iv1 + 0u]; v1.y = ns[iv1 + 1u]; v1.z = ns[iv1 + 2u];
	v2.x = ns[iv2 + 0u]; v2.y = ns[iv2 + 1u]; v2.z = ns[iv2 + 2u];

	REAL3 norm = Cross(v1 - v0, v2 - v0);
	Normalize(norm);

	fNorms[id + 0u] = norm.x;
	fNorms[id + 1u] = norm.y;
	fNorms[id + 2u] = norm.z;

	REAL radian = AngleBetweenVectors(v1 - v0, v2 - v0);
	//radian = 1.0;
	atomicAdd_REAL(nNorms + iv0 + 0u, norm.x * radian);
	atomicAdd_REAL(nNorms + iv0 + 1u, norm.y * radian);
	atomicAdd_REAL(nNorms + iv0 + 2u, norm.z * radian);

	radian = AngleBetweenVectors(v2 - v1, v0 - v1);
	//radian = 1.0;
	atomicAdd_REAL(nNorms + iv1 + 0u, norm.x * radian);
	atomicAdd_REAL(nNorms + iv1 + 1u, norm.y * radian);
	atomicAdd_REAL(nNorms + iv1 + 2u, norm.z * radian);

	radian = AngleBetweenVectors(v0 - v2, v1 - v2);
	//radian = 1.0;
	atomicAdd_REAL(nNorms + iv2 + 0u, norm.x * radian);
	atomicAdd_REAL(nNorms + iv2 + 1u, norm.y * radian);
	atomicAdd_REAL(nNorms + iv2 + 2u, norm.z * radian);
}
__global__ void nodeNormNormalize_kernel(REAL* nNorms, uint numNodes) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numNodes)
		return;

	id *= 3u;
	REAL3 norm;
	norm.x = nNorms[id + 0u];
	norm.y = nNorms[id + 1u];
	norm.z = nNorms[id + 2u];
	
	Normalize(norm);

	nNorms[id + 0u] = norm.x;
	nNorms[id + 1u] = norm.y;
	nNorms[id + 2u] = norm.z;
}

__global__ void airDamping_kernel(REAL* vs, REAL airDamp, uint numNodes) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numNodes)
		return;

	id *= 3u;
	REAL3 v;
	v.x = vs[id + 0u];
	v.y = vs[id + 1u];
	v.z = vs[id + 2u];

	v *= airDamp;

	vs[id + 0u] = v.x;
	vs[id + 1u] = v.y;
	vs[id + 2u] = v.z;
}
__global__ void laplacianDamping_kernel(REAL* oldVs, REAL* newVs, REAL* ms, uint* nbNs, uint* inbNs, REAL lapDamp, uint numNodes) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numNodes)
		return;

	REAL m = ms[id];
	REAL3 nbVs, v;
	uint ino;
	uint i, istart, iend;

	nbVs.x = nbVs.y = nbVs.z = 0.0;
	if (m > 0.0) {
		istart = inbNs[id];
		iend = inbNs[id + 1u];
		if (iend - istart > 0u) {
			for (i = istart; i < iend; i++) {
				ino = nbNs[id]; ino *= 3u;

				v.x = oldVs[ino + 0u];
				v.y = oldVs[ino + 1u];
				v.z = oldVs[ino + 2u];
				nbVs += v;
			}

			ino = id * 3u;
			v.x = oldVs[ino + 0u];
			v.y = oldVs[ino + 1u];
			v.z = oldVs[ino + 2u];

			//nbVs *= 1.0 / (REAL)(iend - istart);
			//v += (nbVs - v) * lapDamp;
			v += (nbVs - (REAL)(iend - istart) * v) * lapDamp;

			newVs[ino + 0u] = v.x;
			newVs[ino + 1u] = v.y;
			newVs[ino + 2u] = v.z;
		}
	}
}

__global__ void maxVelocitiy_kernel(REAL* vs, uint numNodes, REAL* maxVel) {
	extern __shared__ REAL s_maxVels[];

	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= numNodes)
		s_maxVels[threadIdx.x] = 0.0;

	uint ino = id * 3u;
	REAL3 v;
	v.x = vs[ino + 0u];
	v.y = vs[ino + 1u];
	v.z = vs[ino + 2u];

	REAL l = LengthSquared(v);
	s_maxVels[threadIdx.x] = l;

	for (ino = blockDim.x >> 1u; ino > 32u; ino >>= 1u) {
		__syncthreads();
		if (threadIdx.x < ino)
			if (s_maxVels[threadIdx.x] < s_maxVels[threadIdx.x + ino])
				s_maxVels[threadIdx.x] = s_maxVels[threadIdx.x + ino];
	}
	__syncthreads();
	if (threadIdx.x < 32u) {
		warpMax(s_maxVels, threadIdx.x);
		if (threadIdx.x == 0u)
			atomicMax_REAL(maxVel, s_maxVels[0]);
	}
}