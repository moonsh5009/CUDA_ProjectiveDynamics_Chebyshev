#include "Constraint.h"
#include "../include/CUDA_Custom/DeviceManager.cuh"

__global__ void initCBSPDConstraints_kernel(REAL* ns, REAL w, uint istart, uint iend, CBSPDSpring springs) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	id += istart;
	if (id >= iend)
		return;

	uint ino = id << 1u;
	uint ino0 = springs._inos[ino + 0u];
	uint ino1 = springs._inos[ino + 1u];

	springs._ws[id] = w;
	atomicAdd_REAL(springs._Bs + ino0, w);
	atomicAdd_REAL(springs._Bs + ino1, w);

	ino0 *= 3u; ino1 *= 3u;
	REAL3 n0, n1;
	n0.x = ns[ino0 + 0u]; n0.y = ns[ino0 + 1u]; n0.z = ns[ino0 + 2u];
	n1.x = ns[ino1 + 0u]; n1.y = ns[ino1 + 1u]; n1.z = ns[ino1 + 2u];

	REAL length = Length(n1 - n0);
	springs._restLengths[id] = length;
	//printf("%f %f %f %f\n", length, w, springs._Bs[ino0 / 3u], springs._Bs[ino1 / 3u]);
}

__global__ void jacobiProject0_kernel(REAL* ns, REAL* n0s, REAL* ms, REAL invdt2, uint numNodes, CBSPDSpring springs)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numNodes)
		return;

	uint ino = id * 3u;
	REAL m = ms[id];
	REAL3 n0;
	n0.x = n0s[ino + 0u]; n0.y = n0s[ino + 1u]; n0.z = n0s[ino + 2u];
	n0 *= m * invdt2;

	springs._errors[ino + 0u] = n0.x;
	springs._errors[ino + 1u] = n0.y;
	springs._errors[ino + 2u] = n0.z;
}
__global__ void jacobiProject1_kernel(REAL* ns, CBSPDSpring springs) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= springs._numSprings)
		return;

	uint ino = id << 1u;
	uint ino0 = springs._inos[ino + 0u];
	uint ino1 = springs._inos[ino + 1u];

	REAL w = springs._ws[id];

	ino0 *= 3u; ino1 *= 3u;
	REAL3 n0, n1;
	n0.x = ns[ino0 + 0u]; n0.y = ns[ino0 + 1u]; n0.z = ns[ino0 + 2u];
	n1.x = ns[ino1 + 0u]; n1.y = ns[ino1 + 1u]; n1.z = ns[ino1 + 2u];

	REAL3 error0 = make_REAL3(0.0);
	REAL3 error1 = make_REAL3(0.0);

	error0 += w * n1;
	error1 += w * n0;

	REAL3 d = n0 - n1;
	REAL restLength = springs._restLengths[id];
	REAL length = Length(d);
	if (length) {
		REAL newL = w * restLength / length;
		error0 += newL * d;
		error1 -= newL * d;
	}

	atomicAdd_REAL(springs._errors + ino0 + 0u, error0.x);
	atomicAdd_REAL(springs._errors + ino0 + 1u, error0.y);
	atomicAdd_REAL(springs._errors + ino0 + 2u, error0.z);
	atomicAdd_REAL(springs._errors + ino1 + 0u, error1.x);
	atomicAdd_REAL(springs._errors + ino1 + 1u, error1.y);
	atomicAdd_REAL(springs._errors + ino1 + 2u, error1.z);
}
__global__ void jacobiProject2_kernel(
	REAL* ns, REAL* prevNs, REAL* ms, REAL invdt2, REAL underRelax, REAL omega, uint numNodes, CBSPDSpring springs)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numNodes)
		return;

	REAL m = ms[id];
	if (m > 0.0) {
		uint ino = id * 3u;
		REAL b = springs._Bs[id];
		b += m * invdt2;
		REAL3 error;
		error.x = springs._errors[ino + 0u]; error.y = springs._errors[ino + 1u]; error.z = springs._errors[ino + 2u];

		REAL3 n, prevN, newN;
		n.x = ns[ino + 0u]; n.y = ns[ino + 1u]; n.z = ns[ino + 2u];
		prevN.x = prevNs[ino + 0u]; prevN.y = prevNs[ino + 1u]; prevN.z = prevNs[ino + 2u];

		newN = 1.0 / b * error;
		newN = omega * (underRelax * (newN - n) + n - prevN) + prevN;

		prevNs[ino + 0u] = n.x;
		prevNs[ino + 1u] = n.y;
		prevNs[ino + 2u] = n.z;
		ns[ino + 0u] = newN.x;
		ns[ino + 1u] = newN.y;
		ns[ino + 2u] = newN.z;
	}
}

__global__ void jacobiProject0_kernel(REAL* n0s, REAL* ms, REAL* newNs, REAL invdt2, uint numNodes)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numNodes)
		return;

	uint ino = id * 3u;
	REAL m = ms[id];
	REAL3 n0;
	n0.x = n0s[ino + 0u]; n0.y = n0s[ino + 1u]; n0.z = n0s[ino + 2u];
	n0 *= m * invdt2;

	newNs[ino + 0u] = n0.x;
	newNs[ino + 1u] = n0.y;
	newNs[ino + 2u] = n0.z;
}
__global__ void jacobiProject1_kernel(REAL* ns, REAL* newNs, CBSPDSpring springs) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= springs._numSprings)
		return;

	uint ino = id << 1u;
	uint ino0 = springs._inos[ino + 0u];
	uint ino1 = springs._inos[ino + 1u];

	REAL w = springs._ws[id];

	ino0 *= 3u; ino1 *= 3u;
	REAL3 n0, n1;
	n0.x = ns[ino0 + 0u]; n0.y = ns[ino0 + 1u]; n0.z = ns[ino0 + 2u];
	n1.x = ns[ino1 + 0u]; n1.y = ns[ino1 + 1u]; n1.z = ns[ino1 + 2u];

	REAL3 error0 = w * n1;
	REAL3 error1 = w * n0;

	REAL3 d = n0 - n1;
	REAL length = Length(d);
	if (length) {
		REAL restLength = springs._restLengths[id];
		REAL newL = w * restLength / length;
		error0 += newL * d;
		error1 -= newL * d;
	}

	atomicAdd_REAL(newNs + ino0 + 0u, error0.x);
	atomicAdd_REAL(newNs + ino0 + 1u, error0.y);
	atomicAdd_REAL(newNs + ino0 + 2u, error0.z);
	atomicAdd_REAL(newNs + ino1 + 0u, error1.x);
	atomicAdd_REAL(newNs + ino1 + 1u, error1.y);
	atomicAdd_REAL(newNs + ino1 + 2u, error1.z);
}
__global__ void jacobiProject2_kernel(
	REAL* ns, REAL* prevNs, REAL* ms, REAL* Bs, REAL* newNs, REAL invdt2, REAL underRelax, REAL omega, uint numNodes)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numNodes)
		return;

	REAL m = ms[id];
	REAL b = Bs[id];
	if (m > 0.0) {
		uint ino = id * 3u;
		b += m * invdt2;
		REAL3 newN;
		newN.x = newNs[ino + 0u]; newN.y = newNs[ino + 1u]; newN.z = newNs[ino + 2u];

		REAL3 n, prevN;
		n.x = ns[ino + 0u]; n.y = ns[ino + 1u]; n.z = ns[ino + 2u];
		prevN.x = prevNs[ino + 0u]; prevN.y = prevNs[ino + 1u]; prevN.z = prevNs[ino + 2u];

		newN = 1.0 / b * newN;
		newN = omega * (underRelax * (newN - n) + n - prevN) + prevN;

		prevNs[ino + 0u] = n.x;
		prevNs[ino + 1u] = n.y;
		prevNs[ino + 2u] = n.z;
		ns[ino + 0u] = newN.x;
		ns[ino + 1u] = newN.y;
		ns[ino + 2u] = newN.z;
	}
}
__global__ void jacobiProject2_kernel(
	REAL* ns, REAL* prevNs, REAL* ms, REAL* Bs, REAL* newNs, REAL invdt2,
	REAL underRelax, REAL omega, uint numNodes, REAL* maxError)
{
	extern __shared__ REAL s_maxError[];
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	uint ino;

	s_maxError[threadIdx.x] = 0.0;
	if (id < numNodes) {
		REAL m = ms[id];
		REAL b = Bs[id];
		if (m > 0.0) {
			ino = id * 3u;

			REAL3 n, prevN, newN;
			n.x = ns[ino + 0u];
			n.y = ns[ino + 1u];
			n.z = ns[ino + 2u];

			if (b > 0.0) {
				prevN.x = prevNs[ino + 0u];
				prevN.y = prevNs[ino + 1u];
				prevN.z = prevNs[ino + 2u];
				newN.x = newNs[ino + 0u];
				newN.y = newNs[ino + 1u];
				newN.z = newNs[ino + 2u];

				newN *= 1.0 / (b + m * invdt2);
				newN = omega * (underRelax * (newN - n) + n - prevN) + prevN;

				ns[ino + 0u] = newN.x;
				ns[ino + 1u] = newN.y;
				ns[ino + 2u] = newN.z;

				s_maxError[threadIdx.x] = Length(newN - n);
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