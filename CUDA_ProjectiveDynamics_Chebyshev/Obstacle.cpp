#include "Obstacle.h"

void Obstacle::init(void) {
	Object::init();
	_priTree = new PrimalTree();
}
void Obstacle::addObject(Mesh* mesh, REAL mass, REAL3& pivot, REAL3& rotation, bool isSaved) {
	addElements(mesh, mass);
	h_pivots.push_back(pivot);
	h_rotations.push_back(rotation);
	d_pivots = h_pivots;
	d_rotations = h_rotations;
	if (isSaved) {
		h_fs0 = h_fs;
		h_ns0 = h_ns;
		h_ses0 = h_ses;
		h_bes0 = h_bes;
		h_nbFs0 = h_nbFs;
		h_nbNs0 = h_nbFs;
		h_ms0 = h_ms;
		h_invMs0 = h_invMs;
		h_nodePhases0 = h_nodePhases;
	}
	initBVH();

	ObjParam p;
	p._fs = &h_fs[0];
	p._ns = &h_ns[0];
	p._ms = &h_ms[0];
	p._invMs = &h_invMs[0];
	p._numFaces = _numFaces;
	p._numNodes = _numNodes;
	_priTree->buildTree(p, param(), h_nbFs, h_fNorms, h_nNorms, 0.1, 9u);
}
void Obstacle::computeExternalForce(REAL invdt)
{
	d_n0s = d_ns;

	d_forces.memset(0);
	compRotationForce(d_pivots, d_rotations, invdt * invdt);
}
void Obstacle::update(REAL dt, REAL invdt) {
	computeExternalForce(invdt);

	/*vector<REAL> f(_numNodes * 3u);
	for (int i = 0; i < _numNodes; i++) {
		f[i * 3 + 0] = 8800 * dt;
		f[i * 3 + 1] = 1000 * dt;
	}
	d_forces = f;*/
	
	d_vs.memset(0);
	compPredictPosition(dt);

	updateVelocity(invdt);
}

void Obstacle::reset(void) {
	h_fs = h_fs0;
	h_ns = h_ns0;
	h_ses = h_ses0;
	h_bes = h_bes0;
	h_nbFs = h_nbFs0;
	h_nbNs = h_nbFs0;
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