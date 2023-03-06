#ifndef __DEVICE_MANAGER_CUH__
#define __DEVICE_MANAGER_CUH__

#pragma once
#include "DeviceManager.h"

//#define MIN(x, y)			x < y ? x : y
//#define MAX(x, y)			x > y ? x : y
#define MIN(x, y)			min(x, y)
#define MAX(x, y)			max(x, y)

static __inline__ __device__ __forceinline__ double atomicMax_double(double* address, double val)
{
	unsigned long long* address_as_ull = (unsigned long long*)address;
	unsigned long long old = *address_as_ull, assumed;
	while (val > __longlong_as_double(old))
	{
		assumed = old;
		if ((old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val))) == assumed)
			break;
	}
	return __longlong_as_double(old);
}
static __inline__ __device__ __forceinline__ double atomicMin_double(double* address, double val)
{
	unsigned long long* address_as_ull = (unsigned long long*)address;
	unsigned long long old = *address_as_ull, assumed;

	while (val < __longlong_as_double(old))
	{
		assumed = old;
		if ((old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val))) == assumed)
			break;
	}
	return __longlong_as_double(old);
}
static __inline__ __device__ __forceinline__ double atomicAdd_double(double* address, double val)
{
	unsigned long long* address_as_ull = (unsigned long long*)address;
	unsigned long long old = *address_as_ull, assumed;

	do {
		assumed = old;
	} while ((old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)))) != assumed);

	return __longlong_as_double(old);
}
static __inline__ __device__ __forceinline__ double atomicExch_double(double* address, double val)
{
	unsigned long long* address_as_ull = (unsigned long long*)address;
	unsigned long long old = __double_as_longlong(*address), assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
	} while (assumed != old);

	return __longlong_as_double(old);
}

static __inline__ __device__ __forceinline__ float atomicMax_float(float* address, float val)
{
	unsigned int* address_as_ull = (unsigned int*)address;
	unsigned int old = *address_as_ull, assumed;
	while (val > __int_as_float(old))
	{
		assumed = old;
		if ((old = atomicCAS(address_as_ull, assumed, __float_as_int(val))) == assumed)
			break;
	}
	return __int_as_float(old);
}
static __inline__ __device__ __forceinline__ float atomicMin_float(float* address, float val)
{
	unsigned int* address_as_ull = (unsigned int*)address;
	unsigned int old = *address_as_ull, assumed;

	while (val < __int_as_float(old))
	{
		assumed = old;
		if ((old = atomicCAS(address_as_ull, assumed, __float_as_int(val))) == assumed)
			break;
	}
	return __int_as_float(old);
}
static __inline__ __device__ __forceinline__ float atomicAdd_float(float* address, float val)
{
	unsigned int* address_as_ull = (unsigned int*)address;
	unsigned int old = *address_as_ull, assumed;

	do {
		assumed = old;
	} while ((old = atomicCAS(address_as_ull, assumed, __float_as_int(val + __int_as_float(assumed)))) != assumed);

	return __int_as_float(old);
}
static __inline__ __device__ __forceinline__ float atomicExch_float(float* address, float val)
{
	unsigned int* address_as_ull = (unsigned int*)address;
	unsigned int old = __float_as_int(*address), assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __float_as_int(val));
	} while (assumed != old);

	return __int_as_float(old);
}

static __inline__ __device__ __forceinline__ REAL MyAtomicOr(uint* address, uint val)
{
	uint ret = *address;
	while ((ret & val) != val)
	{
		uint old = ret;
		if ((ret = atomicCAS(address, old, old | val)) == old)
			break;
	}
	return ret;
}
static __inline__ __device__ __forceinline__ void cudaLock(uint* mutex, uint i) {
	while (atomicCAS(mutex + i, 0, 1) != 0);
}
static __inline__ __device__ __forceinline__ void cudaUnLock(uint* mutex, uint i) {
	atomicExch(mutex + i, 0);
}

static __inline__ __device__ __forceinline__ void warpAdd(volatile uint* s_data, uint tid) {
	s_data[tid] += s_data[tid + 32];
	s_data[tid] += s_data[tid + 16];
	s_data[tid] += s_data[tid + 8];
	s_data[tid] += s_data[tid + 4];
	s_data[tid] += s_data[tid + 2];
	s_data[tid] += s_data[tid + 1];
}
static __inline__ __device__ __forceinline__ void warpAdd(volatile double* s_data, uint tid) {
	s_data[tid] += s_data[tid + 32];
	s_data[tid] += s_data[tid + 16];
	s_data[tid] += s_data[tid + 8];
	s_data[tid] += s_data[tid + 4];
	s_data[tid] += s_data[tid + 2];
	s_data[tid] += s_data[tid + 1];
}
static __inline__ __device__ __forceinline__ void warpAdd(volatile float* s_data, uint tid) {
	s_data[tid] += s_data[tid + 32];
	s_data[tid] += s_data[tid + 16];
	s_data[tid] += s_data[tid + 8];
	s_data[tid] += s_data[tid + 4];
	s_data[tid] += s_data[tid + 2];
	s_data[tid] += s_data[tid + 1];
}
static __inline__ __device__ __forceinline__ void warpMin(volatile uint* s_data, uint tid) {
	if (s_data[tid] > s_data[tid + 32])
		s_data[tid] = s_data[tid + 32];
	if (s_data[tid] > s_data[tid + 16])
		s_data[tid] = s_data[tid + 16];
	if (s_data[tid] > s_data[tid + 8])
		s_data[tid] = s_data[tid + 8];
	if (s_data[tid] > s_data[tid + 4])
		s_data[tid] = s_data[tid + 4];
	if (s_data[tid] > s_data[tid + 2])
		s_data[tid] = s_data[tid + 2];
	if (s_data[tid] > s_data[tid + 1])
		s_data[tid] = s_data[tid + 1];
}
static __inline__ __device__ __forceinline__ void warpMin(volatile double* s_data, uint tid) {
	if (s_data[tid] > s_data[tid + 32])
		s_data[tid] = s_data[tid + 32];
	if (s_data[tid] > s_data[tid + 16])
		s_data[tid] = s_data[tid + 16];
	if (s_data[tid] > s_data[tid + 8])
		s_data[tid] = s_data[tid + 8];
	if (s_data[tid] > s_data[tid + 4])
		s_data[tid] = s_data[tid + 4];
	if (s_data[tid] > s_data[tid + 2])
		s_data[tid] = s_data[tid + 2];
	if (s_data[tid] > s_data[tid + 1])
		s_data[tid] = s_data[tid + 1];
}
static __inline__ __device__ __forceinline__ void warpMin(volatile float* s_data, uint tid) {
	if (s_data[tid] > s_data[tid + 32])
		s_data[tid] = s_data[tid + 32];
	if (s_data[tid] > s_data[tid + 16])
		s_data[tid] = s_data[tid + 16];
	if (s_data[tid] > s_data[tid + 8])
		s_data[tid] = s_data[tid + 8];
	if (s_data[tid] > s_data[tid + 4])
		s_data[tid] = s_data[tid + 4];
	if (s_data[tid] > s_data[tid + 2])
		s_data[tid] = s_data[tid + 2];
	if (s_data[tid] > s_data[tid + 1])
		s_data[tid] = s_data[tid + 1];
}
static __inline__ __device__ __forceinline__ void warpMax(volatile uint* s_data, uint tid) {
	if (s_data[tid] < s_data[tid + 32])
		s_data[tid] = s_data[tid + 32];
	if (s_data[tid] < s_data[tid + 16])
		s_data[tid] = s_data[tid + 16];
	if (s_data[tid] < s_data[tid + 8])
		s_data[tid] = s_data[tid + 8];
	if (s_data[tid] < s_data[tid + 4])
		s_data[tid] = s_data[tid + 4];
	if (s_data[tid] < s_data[tid + 2])
		s_data[tid] = s_data[tid + 2];
	if (s_data[tid] < s_data[tid + 1])
		s_data[tid] = s_data[tid + 1];
}
static __inline__ __device__ __forceinline__ void warpMax(volatile double* s_data, uint tid) {
	if (s_data[tid] < s_data[tid + 32])
		s_data[tid] = s_data[tid + 32];
	if (s_data[tid] < s_data[tid + 16])
		s_data[tid] = s_data[tid + 16];
	if (s_data[tid] < s_data[tid + 8])
		s_data[tid] = s_data[tid + 8];
	if (s_data[tid] < s_data[tid + 4])
		s_data[tid] = s_data[tid + 4];
	if (s_data[tid] < s_data[tid + 2])
		s_data[tid] = s_data[tid + 2];
	if (s_data[tid] < s_data[tid + 1])
		s_data[tid] = s_data[tid + 1];
}
static __inline__ __device__ __forceinline__ void warpMax(volatile float* s_data, uint tid) {
	if (s_data[tid] < s_data[tid + 32])
		s_data[tid] = s_data[tid + 32];
	if (s_data[tid] < s_data[tid + 16])
		s_data[tid] = s_data[tid + 16];
	if (s_data[tid] < s_data[tid + 8])
		s_data[tid] = s_data[tid + 8];
	if (s_data[tid] < s_data[tid + 4])
		s_data[tid] = s_data[tid + 4];
	if (s_data[tid] < s_data[tid + 2])
		s_data[tid] = s_data[tid + 2];
	if (s_data[tid] < s_data[tid + 1])
		s_data[tid] = s_data[tid + 1];
}

static __inline__ __forceinline__ __global__ void reorderIdsUint2_kernel(
	uint2* xs, uint* ixs, uint size, uint isize)
{
	extern __shared__ uint s_ids[];
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	uint curr;
	if (id < size) {
		uint2 tmp = xs[id];
		curr = tmp.x;
		s_ids[threadIdx.x + 1u] = curr;
		if (id > 0u && threadIdx.x == 0u) {
			tmp = xs[id - 1u];
			s_ids[0] = tmp.x;
		}
	}
	__syncthreads();

	if (id < size) {
		uint i;
		uint prev = s_ids[threadIdx.x];
		if (id == 0u || prev != curr) {
			if (id == 0u) {
				ixs[0] = 0u;
				prev = 0u;
			}
			for (i = prev + 1u; i <= curr; i++)
				ixs[i] = id;
		}
		if (id == size - 1u) {
			for (i = curr + 1u; i < isize; i++)
				ixs[i] = id + 1u;
		}
	}
}

inline __forceinline__ __device__ void setDouble3(double* ptr, uint ino, const double3& x) {
	ino *= 3u;
	ptr[ino + 0u] = x.x;
	ptr[ino + 1u] = x.y;
	ptr[ino + 2u] = x.z;
}
inline __forceinline__ __device__ void setDouble3(double* ptr, uint ino, const double* x) {
	ino *= 3u;
	ptr[ino + 0u] = x[0];
	ptr[ino + 1u] = x[1];
	ptr[ino + 2u] = x[2];
}
inline __forceinline__ __device__ void setFloat3(float* ptr, uint ino, const float3& x) {
	ino *= 3u;
	ptr[ino + 0u] = x.x;
	ptr[ino + 1u] = x.y;
	ptr[ino + 2u] = x.z;
}
inline __forceinline__ __device__ void setFloat3(float* ptr, uint ino, const float* x) {
	ino *= 3u;
	ptr[ino + 0u] = x[0];
	ptr[ino + 1u] = x[1];
	ptr[ino + 2u] = x[2];
}

inline __forceinline__ __device__ void getDouble3(const double* ptr, uint ino, double3& x) {
	ino *= 3u;
	x.x = ptr[ino + 0u];
	x.y = ptr[ino + 1u];
	x.z = ptr[ino + 2u];
}
inline __forceinline__ __device__ void getDouble3(const double* ptr, uint ino, double* x) {
	ino *= 3u;
	x[0] = ptr[ino + 0u];
	x[1] = ptr[ino + 1u];
	x[2] = ptr[ino + 2u];
}
inline __forceinline__ __device__ void getFloat3(const float* ptr, uint ino, float3& x) {
	ino *= 3u;
	x.x = ptr[ino + 0u];
	x.y = ptr[ino + 1u];
	x.z = ptr[ino + 2u];
}
inline __forceinline__ __device__ void getFloat3(const float* ptr, uint ino, float* x) {
	ino *= 3u;
	x[0] = ptr[ino + 0u];
	x[1] = ptr[ino + 1u];
	x[2] = ptr[ino + 2u];
}

inline __forceinline__ __device__ void sumDouble3(double* ptr, uint ino, const double3& x) {
	ino *= 3u;
	atomicAdd_double(ptr + ino + 0u, x.x);
	atomicAdd_double(ptr + ino + 1u, x.y);
	atomicAdd_double(ptr + ino + 2u, x.z);
}
inline __forceinline__ __device__ void sumDouble3(double* ptr, uint ino, const double* x) {
	ino *= 3u;
	atomicAdd_double(ptr + ino + 0u, x[0]);
	atomicAdd_double(ptr + ino + 1u, x[1]);
	atomicAdd_double(ptr + ino + 2u, x[2]);
}
inline __forceinline__ __device__ void sumFloat3(float* ptr, uint ino, const float3& x) {
	ino *= 3u;
	atomicAdd_float(ptr + ino + 0u, x.x);
	atomicAdd_float(ptr + ino + 1u, x.y);
	atomicAdd_float(ptr + ino + 2u, x.z);
}
inline __forceinline__ __device__ void sumFloat3(float* ptr, uint ino, const float* x) {
	ino *= 3u;
	atomicAdd_float(ptr + ino + 0u, x[0]);
	atomicAdd_float(ptr + ino + 1u, x[1]);
	atomicAdd_float(ptr + ino + 2u, x[2]);
}

#endif