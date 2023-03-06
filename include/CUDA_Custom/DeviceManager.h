#ifndef __DEVICE_MANAGER_H__
#define __DEVICE_MANAGER_H__

#pragma once
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "thrust/device_ptr.h"
#include "thrust/scan.h"
#include "thrust/sort.h"
#include "thrust/unique.h"
#include "thrust/extrema.h"
#include "thrust/execution_policy.h"
#include <chrono>
#include <vector>

using namespace std;
typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef std::chrono::system_clock::time_point ctimer;

#define M_PI				3.14159265359

#if 1
typedef double				REAL;
typedef double2				REAL2;
typedef double3				REAL3;
typedef double4				REAL4;
typedef unsigned long long	REAL_INT;

#define REAL_AS_INT			__double_as_longlong
#define INT_AS_REAL			__longlong_as_double
#define REAL_MAX			DBL_MAX
#define REAL_INT_MAX		ULLONG_MAX
#define REAL_EPSILON		DBL_EPSILON

#define make_REAL2			make_double2
#define make_REAL3			make_double3
#define make_REAL4			make_double4

#define atomicMax_REAL		atomicMax_double
#define atomicMin_REAL		atomicMin_double
#define atomicAdd_REAL		atomicAdd_double
#define atomicExch_REAL		atomicExch_double

#define setVector			setDouble3
#define getVector			getDouble3
#define sumVector			sumDouble3
#else
typedef float				REAL;
typedef float2				REAL2;
typedef float3				REAL3;
typedef float4				REAL4;
typedef unsigned int		REAL_INT;

#define REAL_AS_INT			__float_as_int
#define INT_AS_REAL			__int_as_float
#define REAL_MAX			FLT_MAX
#define REAL_INT_MAX		UINT_MAX
#define REAL_EPSILON		FLT_EPSILON

#define make_REAL2			make_float2
#define make_REAL3			make_float3
#define make_REAL4			make_float4

#define atomicMax_REAL		atomicMax_float
#define atomicMin_REAL		atomicMin_float
#define atomicAdd_REAL		atomicAdd_float
#define atomicExch_REAL		atomicExch_float

#define setVector			setfloat3
#define getVector			getfloat3
#define sumVector			sumfloat3
#endif

#define MAX_DBLOCKSIZE		2048
#define MAX_BLOCKSIZE		1024
#define DBLOCKSIZE			256
#define BLOCKSIZE			128
#define HBLOCKSIZE			64
#define WARPSIZE			32
#define EPS					1.0e-20

#define INV3				0.33333333333333333333333333333

//#define TESTTIMER
#define CUDA_DEBUG

#ifndef CUDA_DEBUG
#define CUDA_CHECK(x)	(x)
#else
#define CUDA_CHECK(x)	do {\
		(x); \
		cudaError_t e = cudaGetLastError(); \
		if (e != cudaSuccess) { \
			printf("cuda failure %s:%d: '%s'\n", \
				__FILE__, __LINE__, cudaGetErrorString(e)); \
			/*exit(1);*/ \
		}\
	} while(0)
#endif

#define CNOW		std::chrono::system_clock::now()

struct uint2_CMP
{
	__host__ __device__
		bool operator()(const uint2& a, const uint2& b) const {
		if (a.x != b.x)
			return a.x < b.x;
		return a.y < b.y;
	}
};

static __inline__ __host__ __device__ bool operator==(const uint2& a, const uint2& b)
{
	return a.x == b.x && a.y == b.y;
}
//-------------------------------------------------------------------------------------------------------------
class StreamParam {
public:
	vector<cudaStream_t>	_streams;
public:
	StreamParam() {
		initStream(0u);
	}
	StreamParam(uint num) {
		initStream(num);
	}
	virtual ~StreamParam() {
		freeStream();
	}
public:
	void initStream(uint num) {
		if (_streams.size() > 0)
			freeStream();
		_streams.resize(num);

		for (int i = 0; i < num; i++)
			CUDA_CHECK(cudaStreamCreate(&_streams[i]));
	}
	void freeStream(void) {
		for (int i = 0; i < _streams.size(); i++)
			CUDA_CHECK(cudaStreamDestroy(_streams[i]));
	}
public:
	inline cudaStream_t* begin(void) {
		return &_streams[0];
	}
	inline cudaStream_t* end(void) {
		return (&_streams[0]) + _streams.size();
	}
	inline cudaStream_t& operator[](size_t i) {
		if (i >= _streams.size()) {
			printf("Error : StreamParam_[] : index out\n");
			exit(1);
		}
		return _streams[i];
	}
};
//-------------------------------------------------------------------------------------------------------------
static __inline__ __host__ __device__ uint divup(uint x, uint y) {
	return max((x + y - 1) / y, 1u);
}
//-------------------------------------------------------------------------------------------------------------
static __inline__ __host__ __device__ void setV(double3& a, int i, double x)
{
	if (i == 0) a.x = x;
	else if (i == 1) a.y = x;
	else a.z = x;
}
static __inline__ __host__ __device__ double& getV(const double3& a, int i)
{
	/*if (i == 0) return a.x;
	else if (i == 1) return a.y;
	return a.z;*/
	return ((double*)&a)[i];
}
static __inline__ __host__ __device__ double3 operator+(const double3& a, const double3& b)
{
	return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
static __inline__ __host__ __device__ double3 operator-(const double3& a, const double3& b)
{
	return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
static __inline__ __host__ __device__ double operator*(const double3& a, const double3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
static __inline__ __host__ __device__ double3 operator+(const double3& a, const double b)
{
	return make_double3(a.x + b, a.y + b, a.z + b);
}
static __inline__ __host__ __device__ double3 operator-(const double3& a, const double b)
{
	return make_double3(a.x - b, a.y - b, a.z - b);
}
static __inline__ __host__ __device__ double3 operator*(const double3& a, const double b)
{
	return make_double3(a.x * b, a.y * b, a.z * b);
}
static __inline__ __host__ __device__ double3 operator/(const double3& a, const double b)
{
	return make_double3(a.x / b, a.y / b, a.z / b);
}
static __inline__ __host__ __device__ double3 operator*(const double b, const double3& a)
{
	return make_double3(a.x * b, a.y * b, a.z * b);
}
static __inline__ __host__ __device__ double3 operator/(const double b, const double3& a)
{
	return make_double3(b / a.x, b / a.y, b / a.z);
}
static __inline__ __host__ __device__ void operator+=(double3& a, const double3& b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}
static __inline__ __host__ __device__ void operator-=(double3& a, const double3& b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}
static __inline__ __host__ __device__ void operator*=(double3& a, const double3& b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}
static __inline__ __host__ __device__ void operator/=(double3& a, const double3& b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
}
static __inline__ __host__ __device__ void operator+=(double3& a, const double b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}
static __inline__ __host__ __device__ void operator-=(double3& a, const double b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}
static __inline__ __host__ __device__ void operator*=(double3& a, const double b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}
static __inline__ __host__ __device__ void operator/=(double3& a, const double b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
}
static __inline__ __host__ __device__ bool operator==(const double3& a, const double3& b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z;
}
static __inline__ __host__ __device__ bool operator!=(const double3& a, const double3& b)
{
	return a.x != b.x && a.y != b.y && a.z != b.z;
}
static __inline__ __host__ __device__ bool operator==(const double3& a, const double b)
{
	return a.x == b && a.y == b && a.z == b;
}
static __inline__ __host__ __device__ bool operator!=(const double3& a, const double b)
{
	return a.x != b && a.y != b && a.z != b;
}
static __inline__ __host__ __device__ double3 make_double3(double s)
{
	return make_double3(s, s, s);
}
static __inline__ __host__ __device__ double3 make_double3(const double * a)
{
	return make_double3(a[0], a[1], a[2]);
}
static __inline__ __host__ __device__ double3 make_double3(const double2 & a)
{
	return make_double3(a.x, a.y, 0.0f);
}
static __inline__ __host__ __device__ double3 make_double3(const double2 & a, double s)
{
	return make_double3(a.x, a.y, s);
}
static __inline__ __host__ __device__ double3 make_double3(const double3 a)
{
	return make_double3(a.x, a.y, a.z);
}
static __inline__ __host__ __device__ double3 make_double3(const double4 & a)
{
	return make_double3(a.x, a.y, a.z);
}

static __inline__ __host__ __device__ double3 minVec(const double3 a, const double3 b)
{
	double3 x;
	if (a.x <= b.x) x.x = a.x;
	else x.x = b.x;
	if (a.y <= b.y) x.y = a.y;
	else x.y = b.y;
	if (a.z <= b.z) x.z = a.z;
	else x.z = b.z;
	return x;
}
static __inline__ __host__ __device__ double3 maxVec(const double3 a, const double3 b)
{
	double3 x;
	if (a.x >= b.x) x.x = a.x;
	else x.x = b.x;
	if (a.y >= b.y) x.y = a.y;
	else x.y = b.y;
	if (a.z >= b.z) x.z = a.z;
	else x.z = b.z;
	return x;
}
static __inline__ __host__ __device__ void Print(const double3& a)
{
	printf("%f, %f, %f\n", a.x, a.y, a.z);
}
static __inline__ __host__ __device__ double LengthSquared(const double3& a)
{
	return a * a;
}
static __inline__ __host__ __device__ double Length(const double3& a)
{
	return sqrt(a * a);
}
static __inline__ __host__ __device__ bool Normalize(double3& a)
{
	double norm = Length(a);
	if (norm == 0) {
		//printf("Error Normalize Length 0\n"); 
		return false;
	}
	a *= 1.0 / norm;
	return true;
}
static __inline__ __host__ __device__ double Dot(const double3& a, const double3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
static __inline__ __host__ __device__ double Dot(const double* a, const double* b)
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
static __inline__ __host__ __device__ double3 Cross(const double3& a, const double3& b)
{
	double3 result = make_double3(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x);
	return result;
}
static __inline__ __host__ __device__ double3 Cross(const double* a, const double* b)
{
	double3 result = make_double3(
		a[1] * b[2] - a[2] * b[1],
		a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0]);
	return result;
}
static __inline__ __host__ __device__ double AngleBetweenVectors(const double3& arg1, const double3& arg2)
{
	double c = Dot(arg1, arg2);
	double s = Length(Cross(arg1, arg2));
	return atan2(s, c);
}
//-------------------------------------------------------------------------------------------------------------
static __inline__ __host__ __device__ void setV(float3& a, int i, float x)
{
	if (i == 0) a.x = x;
	else if (i == 1) a.y = x;
	else a.z = x;
}
static __inline__ __host__ __device__ float& getV(const float3& a, int i)
{
	/*if (i == 0) return a.x;
	else if (i == 1) return a.y;
	return a.z;*/
	return ((float*)&a)[i];
}
static __inline__ __host__ __device__ float3 operator+(const float3& a, const float3& b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
static __inline__ __host__ __device__ float3 operator-(const float3& a, const float3& b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
static __inline__ __host__ __device__ float operator*(const float3& a, const float3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
static __inline__ __host__ __device__ float3 operator+(const float3& a, const float b)
{
	return make_float3(a.x + b, a.y + b, a.z + b);
}
static __inline__ __host__ __device__ float3 operator-(const float3& a, const float b)
{
	return make_float3(a.x - b, a.y - b, a.z - b);
}
static __inline__ __host__ __device__ float3 operator*(const float3& a, const float b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}
static __inline__ __host__ __device__ float3 operator/(const float3& a, const float b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}
static __inline__ __host__ __device__ float3 operator*(const float b, const float3& a)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}
static __inline__ __host__ __device__ float3 operator/(const float b, const float3& a)
{
	return make_float3(b / a.x, b / a.y, b / a.z);
}
static __inline__ __host__ __device__ void operator+=(float3& a, const float3& b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}
static __inline__ __host__ __device__ void operator-=(float3& a, const float3& b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}
static __inline__ __host__ __device__ void operator*=(float3& a, const float3& b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}
static __inline__ __host__ __device__ void operator/=(float3& a, const float3& b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
}
static __inline__ __host__ __device__ void operator+=(float3& a, const float b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}
static __inline__ __host__ __device__ void operator-=(float3& a, const float b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}
static __inline__ __host__ __device__ void operator*=(float3& a, const float b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}
static __inline__ __host__ __device__ void operator/=(float3& a, const float b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
}
static __inline__ __host__ __device__ bool operator==(const float3& a, const float3& b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z;
}
static __inline__ __host__ __device__ bool operator!=(const float3& a, const float3& b)
{
	return a.x != b.x && a.y != b.y && a.z != b.z;
}
static __inline__ __host__ __device__ bool operator==(const float3& a, const float b)
{
	return a.x == b && a.y == b && a.z == b;
}
static __inline__ __host__ __device__ bool operator!=(const float3& a, const float b)
{
	return a.x != b && a.y != b && a.z != b;
}
static __inline__ __host__ __device__ float3 make_float3(float s)
{
	return make_float3(s, s, s);
}
static __inline__ __host__ __device__ float3 make_float3(const float* a)
{
	return make_float3(a[0], a[1], a[2]);
}
static __inline__ __host__ __device__ float3 make_float3(const float2& a)
{
	return make_float3(a.x, a.y, 0.0f);
}
static __inline__ __host__ __device__ float3 make_float3(const float2& a, float s)
{
	return make_float3(a.x, a.y, s);
}
static __inline__ __host__ __device__ float3 make_float3(const float3 a)
{
	return make_float3(a.x, a.y, a.z);
}
static __inline__ __host__ __device__ float3 make_float3(const float4& a)
{
	return make_float3(a.x, a.y, a.z);
}

static __inline__ __host__ __device__ float3 minVec(const float3 a, const float3 b)
{
	float3 x;
	if (a.x <= b.x) x.x = a.x;
	else x.x = b.x;
	if (a.y <= b.y) x.y = a.y;
	else x.y = b.y;
	if (a.z <= b.z) x.z = a.z;
	else x.z = b.z;
	return x;
}
static __inline__ __host__ __device__ float3 maxVec(const float3 a, const float3 b)
{
	float3 x;
	if (a.x >= b.x) x.x = a.x;
	else x.x = b.x;
	if (a.y >= b.y) x.y = a.y;
	else x.y = b.y;
	if (a.z >= b.z) x.z = a.z;
	else x.z = b.z;
	return x;
}
static __inline__ __host__ __device__ void Print(const float3& a)
{
	printf("%f, %f, %f\n", a.x, a.y, a.z);
}
static __inline__ __host__ __device__ float LengthSquared(const float3& a)
{
	return a * a;
}
static __inline__ __host__ __device__ float Length(const float3& a)
{
	return sqrt(a * a);
}
static __inline__ __host__ __device__ bool Normalize(float3& a)
{
	float norm = Length(a);
	if (norm == 0) {
		//printf("Error Normalize Length 0\n"); 
		return false;
	}
	a *= 1.0 / norm;
	return true;
}
static __inline__ __host__ __device__ float Dot(const float3& a, const float3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
static __inline__ __host__ __device__ float Dot(const float* a, const float* b)
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
static __inline__ __host__ __device__ float3 Cross(const float3& a, const float3& b)
{
	float3 result = make_float3(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x);
	return result;
}
static __inline__ __host__ __device__ float3 Cross(const float* a, const float* b)
{
	float3 result = make_float3(
		a[1] * b[2] - a[2] * b[1],
		a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0]);
	return result;
}
static __inline__ __host__ __device__ float AngleBetweenVectors(const float3& arg1, const float3& arg2)
{
	float c = Dot(arg1, arg2);
	float s = Length(Cross(arg1, arg2));
	return atan2(s, c);
}
//-------------------------------------------------------------------------------------------------------------
static __inline__ __host__ __device__ uint Log2(uint num)
{
	uint k = 2, n = 0;
	while (k << n <= num) n++;
	return n;
}
static __inline__ __host__ __device__ uint MaxBinary(uint num)
{
	uint n = 1;
	while (n < num)
		n = n << 1;
	return n;
}
//-------------------------------------------------------------------------------------------------------------
struct AABB {
	REAL3 _min;
	REAL3 _max;
};
static __inline__ __host__ __device__ void printAABB(const AABB & a) {
	printf("Min: (%f, %f, %f)\nMax: (%f, %f, %f)\n", a._min.x, a._min.y, a._min.z, a._max.x, a._max.y, a._max.z);
}
inline __host__ __device__ __forceinline__ void resetAABB(AABB& aabb) {
	aabb._min.x = aabb._min.y = aabb._min.z = DBL_MAX;
	aabb._max.x = aabb._max.y = aabb._max.z = -DBL_MAX;
}
static __inline__ __host__ __device__ void setAABB(AABB& a, const REAL3& min, const REAL3& max) {
	a._min = min;
	a._max = max;
}
static __inline__ __host__ __device__ void setAABB(AABB& a, const REAL3& cen, const REAL delta) {
	a._min = cen - delta;
	a._max = cen + delta;
}
static __inline__ __host__ __device__ void setAABB(AABB& a, const AABB& b) {
	a._min = b._min;
	a._max = b._max;
}
static __inline__ __host__ __device__ void addAABB(AABB& a, const REAL3& x) {
	if (a._min.x > x.x)
		a._min.x = x.x;
	if (a._max.x < x.x)
		a._max.x = x.x;
	if (a._min.y > x.y)
		a._min.y = x.y;
	if (a._max.y < x.y)
		a._max.y = x.y;
	if (a._min.z > x.z)
		a._min.z = x.z;
	if (a._max.z < x.z)
		a._max.z = x.z;
}
static __inline__ __host__ __device__ void addAABB(AABB& a, const REAL3& x, REAL delta) {
	if (a._min.x > x.x - delta)
		a._min.x = x.x - delta;
	if (a._max.x < x.x + delta)
		a._max.x = x.x + delta;
	if (a._min.y > x.y - delta)
		a._min.y = x.y - delta;
	if (a._max.y < x.y + delta)
		a._max.y = x.y + delta;
	if (a._min.z > x.z - delta)
		a._min.z = x.z - delta;
	if (a._max.z < x.z + delta)
		a._max.z = x.z + delta;
}
static __inline__ __host__ __device__ void addAABB(AABB& a, const AABB& x) {
	if (a._min.x > x._min.x)
		a._min.x = x._min.x;
	if (a._max.x < x._max.x)
		a._max.x = x._max.x;
	if (a._min.y > x._min.y)
		a._min.y = x._min.y;
	if (a._max.y < x._max.y)
		a._max.y = x._max.y;
	if (a._min.z > x._min.z)
		a._min.z = x._min.z;
	if (a._max.z < x._max.z)
		a._max.z = x._max.z;
}
inline  __host__ __device__ __forceinline__ bool intersect(const AABB& a, const AABB& b) {
	return a._min.x <= b._max.x
		&& a._min.y <= b._max.y
		&& a._min.z <= b._max.z
		&& a._max.x >= b._min.x
		&& a._max.y >= b._min.y
		&& a._max.z >= b._min.z;
}
inline  __host__ __device__ __forceinline__ bool intersect(const AABB& a, const REAL3& b, const REAL delta) {
	return a._min.x <= b.x + delta
		&& a._min.y <= b.y + delta
		&& a._min.z <= b.z + delta
		&& a._max.x >= b.x - delta
		&& a._max.y >= b.y - delta
		&& a._max.z >= b.z - delta;
}

#endif