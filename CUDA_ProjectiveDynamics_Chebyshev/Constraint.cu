#include "Constraint.cuh"

void CBSPDConstraint::addConstratins(const Dvector<uint>& es, const Dvector<REAL>& ns, REAL w) {
	cudaDeviceSynchronize();
	ctimer timer = CNOW;

	uint numNodes = ns.size() / 3u;
	uint numSprings = es.size() >> 1u;
	uint numNodes0 = _numNodes;
	uint numSprings0 = _numSprings;
	_numNodes = numNodes;
	_numSprings += numSprings;

	extendSprings();
	CUDA_CHECK(cudaMemcpy(_inos() + (numSprings0 << 1u), es(), es.size() * sizeof(uint), cudaMemcpyDeviceToDevice));
	if (_numNodes - numNodes0)
		CUDA_CHECK(cudaMemset(_Bs() + numNodes0, 0, (_numNodes - numNodes0) * sizeof(REAL)));
	initCBSPDConstraints_kernel << <divup(_numSprings - numSprings0, CONSTRAINT_BLOCKSIZE), CONSTRAINT_BLOCKSIZE >> > (
		ns(), w, numSprings0, _numSprings, springs());
	CUDA_CHECK(cudaPeekAtLastError());

	cudaDeviceSynchronize();
	printf("Add CBSPD Constraints: %f msec, numSpring: %d\n", (CNOW - timer) / 10000.0, _numSprings);
}

void CBSPDConstraint::getOmega(uint itr, REAL& omg) {
	if (itr <= 10u)			omg = 1.0;
	else if (itr == 11u)	omg = 2.0 / (2.0 - _rho * _rho);
	else					omg = 4.0 / (4.0 - _rho * _rho * omg);
}
void CBSPDConstraint::jacobiProject0(Dvector<REAL>& n0s, Dvector<REAL>& ms, Dvector<REAL>& newNs, REAL invdt2) {
	jacobiProject0_kernel << <divup(_numNodes, CONSTRAINT_BLOCKSIZE), CONSTRAINT_BLOCKSIZE >> > (
		n0s(), ms(), newNs(), invdt2, _numNodes);
	CUDA_CHECK(cudaPeekAtLastError());
}
void CBSPDConstraint::jacobiProject1(Dvector<REAL>& ns, Dvector<REAL>& newNs) {
	jacobiProject1_kernel << <divup(_numSprings, CONSTRAINT_BLOCKSIZE), CONSTRAINT_BLOCKSIZE >> > (
		ns(), newNs(), springs());
	CUDA_CHECK(cudaPeekAtLastError());
}
void CBSPDConstraint::jacobiProject2(
	Dvector<REAL>& ns, Dvector<REAL>& prevNs,
	Dvector<REAL>& ms, Dvector<REAL>& Bs,
	Dvector<REAL>& newNs, REAL invdt2, REAL omg)
{
	jacobiProject2_kernel << <divup(_numNodes, CONSTRAINT_BLOCKSIZE), CONSTRAINT_BLOCKSIZE >> > (
		ns(), prevNs(), ms(), Bs(), newNs(), invdt2, _underRelax, omg, _numNodes);
	CUDA_CHECK(cudaPeekAtLastError());
}
void CBSPDConstraint::jacobiProject2(
	Dvector<REAL>& ns, Dvector<REAL>& prevNs,
	Dvector<REAL>& ms, Dvector<REAL>& Bs,
	Dvector<REAL>& newNs, REAL invdt2,
	REAL underRelax, REAL omg, REAL& maxError)
{
	REAL* d_maxError;
	CUDA_CHECK(cudaMalloc((void**)&d_maxError, sizeof(REAL)));
	CUDA_CHECK(cudaMemset(d_maxError, 0, sizeof(REAL)));

	jacobiProject2_kernel << <divup(_numNodes, CONSTRAINT_BLOCKSIZE), CONSTRAINT_BLOCKSIZE, CONSTRAINT_BLOCKSIZE * sizeof(REAL) >> > (
		ns(), prevNs(), ms(), Bs(), newNs(), invdt2, underRelax, omg, _numNodes, d_maxError);
	CUDA_CHECK(cudaPeekAtLastError());

	CUDA_CHECK(cudaMemcpy(&maxError, d_maxError, sizeof(REAL), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(d_maxError));
}
void CBSPDConstraint::project(Dvector<REAL>& ns, Dvector<REAL>& ms, REAL invdt2, uint iteration) {
	Dvector<REAL> n0s;
	Dvector<REAL> prevNs;
	REAL omg;

	uint nodeBlockSize = divup(_numNodes, CONSTRAINT_BLOCKSIZE);
	uint springBlockSize = divup(_numSprings, CONSTRAINT_BLOCKSIZE);
	n0s = ns;
	prevNs = ns;

	for (uint itr = 0u; itr < iteration; itr++) {
		if (itr <= 10u)			omg = 1.0;
		else if (itr == 11u)	omg = 2.0 / (2.0 - _rho * _rho);
		else					omg = 4.0 / (4.0 - _rho * _rho * omg);

		jacobiProject0_kernel << <nodeBlockSize, CONSTRAINT_BLOCKSIZE >> > (
			ns(), n0s(), ms(), invdt2, _numNodes, springs());
		CUDA_CHECK(cudaPeekAtLastError());
		jacobiProject1_kernel << <springBlockSize, CONSTRAINT_BLOCKSIZE >> > (
			ns(), springs());
		CUDA_CHECK(cudaPeekAtLastError());
		jacobiProject2_kernel << <nodeBlockSize, CONSTRAINT_BLOCKSIZE >> > (
			ns(), prevNs(), ms(), invdt2, _underRelax, omg, _numNodes, springs());
		CUDA_CHECK(cudaPeekAtLastError());
	}
}