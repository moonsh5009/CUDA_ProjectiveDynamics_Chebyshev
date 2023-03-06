#include "Cloth.h"

void Cloth::init(uint maxIter) {
	Object::init();
	_maxIter = maxIter;
	_constraints = new CBSPDConstraint();
	//_constraints->init(0.9992, 0.9);
	_constraints->init(0.992, 0.9);
	//_constraints->init(0.1, 0.9);
	//_constraints->init(0.0, 1.0);
}
void Cloth::addCloth(Mesh* mesh, REAL mass, bool isSaved) {
	uint numSes0 = d_ses.arraySize() >> 1u;
	uint numBes0 = d_bes.arraySize() >> 1u;

	addElements(mesh, mass);

	uint numSes = d_ses.arraySize() >> 1u;
	uint numBes = d_bes.arraySize() >> 1u;
	Dvector<uint> es((numSes - numSes0) << 1u);
	CUDA_CHECK(cudaMemcpy(es(), d_ses._array() + (numSes0 << 1u), es.size() * sizeof(uint), cudaMemcpyDeviceToDevice));
	_constraints->addConstratins(es, d_ns, 100000.0);

	es.resize((numBes - numBes0) << 1u);
	CUDA_CHECK(cudaMemcpy(es(), d_bes._array() + (numBes0 << 1u), es.size() * sizeof(uint), cudaMemcpyDeviceToDevice));
	_constraints->addConstratins(es, d_ns, 10000.0);

	es.clear();

	//fix();

	if (isSaved) {
		h_fs0 = h_fs;
		h_ns0 = h_ns;
		h_ses0 = h_ses;
		h_bes0 = h_bes;
		h_nbFs0 = h_nbFs;
		h_nbNs0 = h_nbNs;
		h_ms0 = h_ms;
		h_invMs0 = h_invMs;
		h_nodePhases0 = h_nodePhases;
	}
	initBVH();
}
void Cloth::fix(void) {
	/*h_ms[0] = 0.0;
	h_invMs[0] = 0.0;*/

	/*uint mxmy = 0u;
	uint Mxmy = 0u;
	uint mxMy = 0u;
	uint MxMy = 0u;
	for (uint i = 1u; i < _numNodes; i++) {
		REAL3 n = make_REAL3(h_ns[i * 3u + 0u], h_ns[i * 3u + 1u], h_ns[i * 3u + 2u]);

		if (n.x - h_ns[mxmy * 3u + 0u] < 1.0e-5 && n.y - h_ns[mxmy * 3u + 1u] < 1.0e-5)
			mxmy = i;
		if (n.x - h_ns[Mxmy * 3u + 0u] > -1.0e-5 && n.y - h_ns[Mxmy * 3u + 1u] < 1.0e-5)
			Mxmy = i;
		if (n.x - h_ns[mxMy * 3u + 0u] < 1.0e-5 && n.y - h_ns[mxMy * 3u + 1u] > -1.0e-5)
			mxMy = i;
		if (n.x - h_ns[MxMy * 3u + 0u] > -1.0e-5 && n.y - h_ns[MxMy * 3u + 1u] > -1.0e-5)
			MxMy = i;
	}

	_fixed.push_back(Mxmy);
	_fixed.push_back(MxMy);
	_fixed.push_back(mxmy);
	_fixed.push_back(mxMy);
	h_ms[mxmy] = h_ms[Mxmy] = h_ms[mxMy] = h_ms[MxMy] = 0.0;
	h_invMs[mxmy] = h_invMs[Mxmy] = h_invMs[mxMy] = h_invMs[MxMy] = 0.0;*/


	_fixed.resize(2, 0u);
	for (uint i = 0u; i < _numNodes; i++) {
		REAL3 n = make_REAL3(h_ns[i * 3u + 0u], h_ns[i * 3u + 1u], h_ns[i * 3u + 2u]);

		if (n.x > 1.0 - 0.05) {
			_fixed.push_back(i);
			h_ms[i] = h_invMs[i] = 0.0;
		}
	}
	_fixed[0] = _fixed.size();

	for (uint i = 0u; i < _numNodes; i++) {
		REAL3 n = make_REAL3(h_ns[i * 3u + 0u], h_ns[i * 3u + 1u], h_ns[i * 3u + 2u]);

		if (n.x < -1.0 + 0.05) {
			_fixed.push_back(i);
			h_ms[i] = h_invMs[i] = 0.0;
		}
	}
	_fixed[1] = _fixed.size();

	d_ms = h_ms;
	d_invMs = h_invMs;
}
void Cloth::rotateFixed(REAL invdt) {
	if (!_fixed.size())
		return;;
	//vector<REAL> h_vs;
	//d_vs.copyToHost(h_vs);

	//REAL3 degree = _degree;
	//REAL3 a, b, pa, pb;
	//REAL3 pivot, va, vb;
	//REAL cx, sx, cy, sy, cz, sz;

	//degree *= M_PI * 0.00555555555555555555555555555556;
	//for (uint i = 0u; i < 3u; i+=2) {
	//	if (i > 0u) break;
	//	degree *= -1.0;
	//	cx = cos(degree.x);
	//	sx = sin(degree.x);
	//	cy = cos(degree.y);
	//	sy = -sin(degree.y);
	//	cz = cos(degree.z);
	//	sz = sin(degree.z);

	//	a.x = h_ns[_fixed[i + 0u] * 3u + 0u];
	//	a.y = h_ns[_fixed[i + 0u] * 3u + 1u];
	//	a.z = h_ns[_fixed[i + 0u] * 3u + 2u];
	//	b.x = h_ns[_fixed[i + 1u] * 3u + 0u];
	//	b.y = h_ns[_fixed[i + 1u] * 3u + 1u];
	//	b.z = h_ns[_fixed[i + 1u] * 3u + 2u];
	//	pivot = (a + b) * 0.5;
	//	
	//	a -= pivot;
	//	b -= pivot;

	//	pa.x = a.x * cz * cy + a.y * (cz * sy * sx - sz * cx) + a.z * (cz * sy * cx + sz * sx);
	//	pa.y = a.x * sz * cy + a.y * (sz * sy * sx + cz * cx) + a.z * (sz * sy * cx - cz * sx);
	//	pa.z = a.x * -sy + a.y * cy * sx + a.z * cy * cx;
	//	pb.x = b.x * cz * cy + b.y * (cz * sy * sx - sz * cx) + b.z * (cz * sy * cx + sz * sx);
	//	pb.y = b.x * sz * cy + b.y * (sz * sy * sx + cz * cx) + b.z * (sz * sy * cx - cz * sx);
	//	pb.z = b.x * -sy + b.y * cy * sx + b.z * cy * cx;

	//	/*pa += 0.002 * (0.4 - Length(pa)) / Length(pa) * (pa);
	//	pb += 0.002 * (0.4 - Length(pb)) / Length(pb) * (pb);
	//	
	//	if (i == 0u) {
	//		pa.x += (-0.5 - pivot.x) * 0.002;
	//		pb.x += (-0.5 - pivot.x) * 0.002;
	//	}
	//	else {
	//		pa.x += (0.5 - pivot.x) * 0.002;
	//		pb.x += (0.5 - pivot.x) * 0.002;
	//	}*/

	//	va = invdt * (pa - a);
	//	vb = invdt * (pb - b);
	//	h_vs[_fixed[i + 0u] * 3u + 0u] = va.x;
	//	h_vs[_fixed[i + 0u] * 3u + 1u] = va.y;
	//	h_vs[_fixed[i + 0u] * 3u + 2u] = va.z;
	//	h_vs[_fixed[i + 1u] * 3u + 0u] = vb.x;
	//	h_vs[_fixed[i + 1u] * 3u + 1u] = vb.y;
	//	h_vs[_fixed[i + 1u] * 3u + 2u] = vb.z;
	//}
	//d_vs = h_vs;

	static REAL asdfasdf = 1.0;
	asdfasdf -= 0.0005;

	vector<REAL> h_vs;
	d_vs.copyToHost(h_vs);

	REAL3 degree = _degree;
	REAL3 a, b, pa, pb, center;
	REAL3 pivot, va, vb;
	REAL mov, cx, sx, cy, sy, cz, sz;

	degree *= M_PI * 0.00555555555555555555555555555556;
	cx = cos(degree.x);
	sx = sin(degree.x);
	cy = cos(degree.y);
	sy = -sin(degree.y);
	cz = cos(degree.z);
	sz = sin(degree.z);

	/*center = make_REAL3(0.0);
	for (uint i = 2u; i < _fixed[0]; i++) {
		a.x = h_ns[_fixed[i] * 3u + 0u];
		a.y = h_ns[_fixed[i] * 3u + 1u];
		a.z = h_ns[_fixed[i] * 3u + 2u];
		center += a;
	}
	center *= 1.0 / (REAL)(_fixed[0] - 2u);
	mov = 0.002 * (0.5 - center.x);*/
	if (asdfasdf <= 0.0)
		mov = 0.0;
	else if (asdfasdf > 0.5)
		mov = -0.0008;
	else
		mov = 0.0008;

	for (uint i = 2u; i < _fixed[0]; i++) {
		a.x = h_ns[_fixed[i] * 3u + 0u];
		a.y = h_ns[_fixed[i] * 3u + 1u];
		a.z = h_ns[_fixed[i] * 3u + 2u];

		pa.x = a.x * cz * cy + a.y * (cz * sy * sx - sz * cx) + a.z * (cz * sy * cx + sz * sx);
		pa.y = a.x * sz * cy + a.y * (sz * sy * sx + cz * cx) + a.z * (sz * sy * cx - cz * sx);
		pa.z = a.x * -sy + a.y * cy * sx + a.z * cy * cx;

		pa.x += mov;

		va = invdt * (pa - a);
		h_vs[_fixed[i] * 3u + 0u] = va.x;
		h_vs[_fixed[i] * 3u + 1u] = va.y;
		h_vs[_fixed[i] * 3u + 2u] = va.z;
	}

	degree = degree * -1.0;
	cx = cos(degree.x);
	sx = sin(degree.x);
	cy = cos(degree.y);
	sy = -sin(degree.y);
	cz = cos(degree.z);
	sz = sin(degree.z);

	/*center = make_REAL3(0.0);
	for (uint i = _fixed[0]; i < _fixed[1]; i++) {
		a.x = h_ns[_fixed[i] * 3u + 0u];
		a.y = h_ns[_fixed[i] * 3u + 1u];
		a.z = h_ns[_fixed[i] * 3u + 2u];
		center += a;
	}
	center *= 1.0 / (REAL)(_fixed[1] - _fixed[0]);
	mov = 0.001 * (-0.5 - center.x);*/
	mov = -mov;

	for (uint i = _fixed[0]; i < _fixed[1]; i++) {
		a.x = h_ns[_fixed[i] * 3u + 0u];
		a.y = h_ns[_fixed[i] * 3u + 1u];
		a.z = h_ns[_fixed[i] * 3u + 2u];

		pa.x = a.x * cz * cy + a.y * (cz * sy * sx - sz * cx) + a.z * (cz * sy * cx + sz * sx);
		pa.y = a.x * sz * cy + a.y * (sz * sy * sx + cz * cx) + a.z * (sz * sy * cx - cz * sx);
		pa.z = a.x * -sy + a.y * cy * sx + a.z * cy * cx;

		pa.x += mov;

		va = invdt * (pa - a);
		h_vs[_fixed[i] * 3u + 0u] = va.x;
		h_vs[_fixed[i] * 3u + 1u] = va.y;
		h_vs[_fixed[i] * 3u + 2u] = va.z;
	}
	d_vs = h_vs;
}
void Cloth::computeExternalForce(REAL3& gravity) {
	d_n0s = d_ns;

	d_forces.memset(0);
	compGravityForce(gravity);
}
void Cloth::update(REAL3& gravity, REAL dt, REAL invdt) {
	//Damping(0.999, 0.1, 4u);
	computeExternalForce(gravity);
	compPredictPosition(dt);

	_constraints->project(d_ns, d_ms, invdt * invdt, _maxIter);

	updateVelocity(invdt);
}

void Cloth::reset(void) {
	h_fs = h_fs0;
	h_ns = h_ns0;
	h_ses = h_ses0;
	h_bes = h_bes0;
	h_nbFs = h_nbFs0;
	h_nbNs = h_nbNs0;
	h_ms = h_ms0;
	h_invMs = h_invMs0;
	h_nodePhases = h_nodePhases0;

	_numFaces = h_fs.size() / 3u;
	_numNodes = h_ns.size() / 3u;

	initVelocities();
	initNoramls();

	copyToDevice();
	copyNbToDevice();
	copyMassToDevice();

	computeNormal();

	initBVH();
}