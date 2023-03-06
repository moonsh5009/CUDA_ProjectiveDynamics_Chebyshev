#ifndef __CUDA_DEVICE_VECTOR_H__
#define __CUDA_DEVICE_VECTOR_H__

#pragma once
#include "DeviceManager.h"

template <typename T>
class Dvector {
public:
	T		*_list;
	size_t	_size;
public:
	Dvector() {
		_list = nullptr;
		_size = 0u; 
	}
	Dvector(size_t size) {
		_list = nullptr;
		if (size > 0u)
			alloc(size);
		else
			_size = 0u;
	}
	Dvector(const Dvector<T>& v) {
		alloc(v.size());
		copyFromDevice(v);
	}
	Dvector(const vector<T>& v) {
		alloc(v.size());
		copyFromHost(v);
	}
	virtual ~Dvector() {
		clear();
	}
public:
	inline void alloc(size_t size) {
		CUDA_CHECK(cudaMalloc((void**)&_list, size * sizeof(T)));
		_size = size;
	}
	inline void resize(size_t size) {
		if (_size == size) return;
		clear();
		if (size > 0u)
			alloc(size);
	}
	inline void free(void) {
		CUDA_CHECK(cudaFree(_list));
		_size = 0u;
	}
	inline void clear(void) {
		if (_size > 0u) {
			CUDA_CHECK(cudaFree(_list));
			_size = 0u;
			_list = nullptr;
		}
	}
	inline T* begin(void) const {
		return _list;
	}
	inline T* end(void) const {
		return (_list + _size);
	}
	inline size_t size(void) const {
		return _size;
	}
	inline void memset(int value) {
		if (_size > 0u)
			CUDA_CHECK(cudaMemset(_list, value, _size * sizeof(T)));
	}
	inline void memset(int value, cudaStream_t& s) {
		if (_size > 0u)
			CUDA_CHECK(cudaMemsetAsync(_list, value, _size * sizeof(T), s));
	}
	inline void swap(Dvector<T>& x) {
		T* tmp = _list;
		_list = x._list;
		x._list = tmp;

		uint stmp = _size;
		_size = x._size;
		x._size = stmp;
	}
	inline void overCopy(Dvector<T>& x) {
		clear();
		_list = x._list;
		_size = x._size;
		x._list = nullptr;
		x._size = 0u;
	}
	inline void extend(uint size) {
		if (size <= _size)
			return;
		Dvector<T> newVec(size);
		CUDA_CHECK(cudaMemcpy(newVec._list, _list, _size * sizeof(T), cudaMemcpyDeviceToDevice));
		overCopy(newVec);
	}
	inline void insert(Dvector<T>& v) {
		Dvector<T> newVec(_size + v._size);
		CUDA_CHECK(cudaMemcpy(newVec._list, _list, _size * sizeof(T), cudaMemcpyDeviceToDevice));
		CUDA_CHECK(cudaMemcpy(newVec._list + _size, v._list, v._size * sizeof(T), cudaMemcpyDeviceToDevice));
		overCopy(newVec);
	}
	inline void insert(vector<T>& v) {
		Dvector<T> newVec(_size + v.size());
		CUDA_CHECK(cudaMemcpy(newVec._list, _list, _size * sizeof(T), cudaMemcpyDeviceToDevice));
		CUDA_CHECK(cudaMemcpy(newVec._list + _size, &v[0], v.size() * sizeof(T), cudaMemcpyHostToDevice));
		overCopy(newVec);
	}
public:
	inline void copyToHost(vector<T>& host) const {
		host.resize(_size);
		if (_size > 0u)
			CUDA_CHECK(cudaMemcpy(&host[0], _list, _size * sizeof(T), cudaMemcpyDeviceToHost));
	}
	inline void copyToDevice(Dvector<T>& device) const {
		device.resize(_size);
		if (_size > 0u)
			CUDA_CHECK(cudaMemcpy(device._list, _list, _size * sizeof(T), cudaMemcpyDeviceToDevice));
	}
	inline void copyFromHost(const vector<T>& host) {
		resize(host.size());
		if (_size > 0u)
			CUDA_CHECK(cudaMemcpy(_list, &host[0], _size * sizeof(T), cudaMemcpyHostToDevice));
	}
	inline void copyFromDevice(const Dvector<T>& device) {
		resize(device._size);
		if (_size > 0u)
			CUDA_CHECK(cudaMemcpy(_list, device._list, _size * sizeof(T), cudaMemcpyDeviceToDevice));
	}
	inline void copyToHost(vector<T>& host, cudaStream_t& s) const {
		host.resize(_size);
		if (_size > 0u)
			CUDA_CHECK(cudaMemcpyAsync(&host[0], _list, _size * sizeof(T), cudaMemcpyDeviceToHost, s));
	}
	inline void copyToDevice(Dvector<T>& device, cudaStream_t& s) const {
		device.resize(_size);
		if (_size > 0u)
			CUDA_CHECK(cudaMemcpyAsync(device._list, _list, _size * sizeof(T), cudaMemcpyDeviceToDevice, s));
	}
	inline void copyFromHost(const vector<T>& host, cudaStream_t& s) {
		resize(host.size());
		if (_size > 0u)
			CUDA_CHECK(cudaMemcpyAsync(_list, &host[0], _size * sizeof(T), cudaMemcpyHostToDevice, s));
	}
	inline void copyFromDevice(const Dvector<T>& device, cudaStream_t& s) {
		resize(device._size);
		if (_size > 0u)
			CUDA_CHECK(cudaMemcpyAsync(_list, device._list, _size * sizeof(T), cudaMemcpyDeviceToDevice, s));
	}
public:
	inline Dvector<T>& operator=(const vector<T>& v) {
		copyFromHost(v);
		return *this;
	}
	inline Dvector<T>& operator=(const Dvector<T>& v) {
		copyFromDevice(v);
		return *this;
	}
	inline T* operator()(void) const {
		return _list;
	}
	/*__inline__ __host__ __device__ */
	/*inline T& operator[](size_t i) {
		if (i >= _size) {
			printf("Error : Dvector_[] : index out\n");
			exit(1);
		}
		return _list[i];
	}*/
};

Dvector<REAL> operator+(const Dvector<REAL>& a, const REAL b);
Dvector<REAL> operator*(const Dvector<REAL>& a, const REAL b);
Dvector<REAL> operator+(const Dvector<REAL>& a, const REAL3 b);
Dvector<REAL> operator*(const Dvector<REAL>& a, const REAL3 b);
Dvector<REAL> operator+(const Dvector<REAL>& a, const Dvector<REAL>& b);
Dvector<REAL> operator-(const Dvector<REAL>& a, const Dvector<REAL>& b);
Dvector<REAL> operator*(const Dvector<REAL>& a, const Dvector<REAL>& b);
Dvector<REAL> operator/(const Dvector<REAL>& a, const Dvector<REAL>& b);
void operator+=(Dvector<REAL>& v, const REAL a);
void operator*=(Dvector<REAL>& v, const REAL a);
void operator+=(Dvector<REAL>& v, const REAL3 a);
void operator*=(Dvector<REAL>& v, const REAL3 a);
void operator+=(Dvector<REAL>& v, const Dvector<REAL>& a);
void operator-=(Dvector<REAL>& v, const Dvector<REAL>& a);
void operator*=(Dvector<REAL>& v, const Dvector<REAL>& a);
void operator/=(Dvector<REAL>& v, const Dvector<REAL>& a);

#endif