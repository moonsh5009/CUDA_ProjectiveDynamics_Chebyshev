#include "Object.cuh"

void Object::init(void) {
	_streams = new StreamParam();
	_streams->initStream(11u);

	_bvh = new BVH();
	_RTri = new RTriangle();

	_numFaces = _numNodes = 0u;

	h_ses._index.resize(1, 0);
	h_bes._index.resize(1, 0);
	h_nbFs._index.resize(1, 0);
	h_nbNs._index.resize(1, 0);

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
void Object::addElements(Mesh* mesh, REAL mass) {
	uint newPhase = _masses.size();
	_masses.push_back(mass);

	h_fs.insert(h_fs.end(), mesh->_fs.begin(), mesh->_fs.end());
	h_ns.insert(h_ns.end(), mesh->_ns.begin(), mesh->_ns.end());

	for (uint i = _numFaces * 3u; i < h_fs.size(); i++)
		h_fs[i] += _numNodes;

	PrefixArray<uint> buffer = mesh->_ses;
	for (uint i = 0u; i < buffer._array.size(); i++)
		buffer._array[i] += _numNodes;
	for (uint i = 0u; i < buffer._index.size(); i++)
		buffer._index[i] += h_ses.arraySize() >> 1u;
	h_ses._array.insert(h_ses._array.end(), buffer._array.begin(), buffer._array.end());
	h_ses._index.insert(h_ses._index.end(), buffer._index.begin() + 1, buffer._index.end());

	buffer = mesh->_bes;
	for (uint i = 0u; i < buffer._array.size(); i++)
		buffer._array[i] += _numNodes;
	for (uint i = 0u; i < buffer._index.size(); i++)
		buffer._index[i] += h_bes.arraySize() >> 1u;
	h_bes._array.insert(h_bes._array.end(), buffer._array.begin(), buffer._array.end());
	h_bes._index.insert(h_bes._index.end(), buffer._index.begin() + 1, buffer._index.end());

	buffer = mesh->_nbFs;
	for (uint i = 0u; i < buffer._array.size(); i++)
		buffer._array[i] += _numFaces;
	for (uint i = 0u; i < buffer._index.size(); i++)
		buffer._index[i] += h_nbFs.arraySize();
	h_nbFs._array.insert(h_nbFs._array.end(), buffer._array.begin(), buffer._array.end());
	h_nbFs._index.insert(h_nbFs._index.end(), buffer._index.begin() + 1, buffer._index.end());

	buffer = mesh->_nbNs;
	for (uint i = 0u; i < buffer._array.size(); i++)
		buffer._array[i] += _numNodes;
	for (uint i = 0u; i < buffer._index.size(); i++)
		buffer._index[i] += h_nbNs.arraySize();
	h_nbNs._array.insert(h_nbNs._array.end(), buffer._array.begin(), buffer._array.end());
	h_nbNs._index.insert(h_nbNs._index.end(), buffer._index.begin() + 1, buffer._index.end());

	_numFaces += mesh->_numFaces;
	_numNodes += mesh->_numVertices;
	h_nodePhases.resize(_numNodes, newPhase);

	REAL invM = 0.0;
	if (_masses[newPhase]) invM = 1.0 / _masses[newPhase];
	h_ms.resize(_numNodes, _masses[newPhase]);
	h_invMs.resize(_numNodes, invM);

	initVelocities();
	initNoramls();

	d_nodePhases = h_nodePhases;
	copyToDevice();
	copyNbToDevice();
	copyMassToDevice();

	computeNormal();
}
void Object::initVelocities(void) {
	d_vs.resize(_numNodes * 3u);
	d_vs.memset(0, (*_streams)[2]);
	d_forces.resize(_numNodes * 3u);
}
void Object::initNoramls(void) {
	h_fNorms.resize(_numFaces * 3u);
	h_nNorms.resize(_numNodes * 3u);
	d_fNorms.resize(_numFaces * 3u);
	d_nNorms.resize(_numNodes * 3u);
}
void Object::initBVH(void) {
	_bvh->build(d_fs, d_ns);
	_RTri->init(d_fs, d_nbFs);
}

void Object::compGravityForce(REAL3& gravity) {
	if (!_numFaces)
		return;
	compGravityForce_kernel << <divup(_numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		d_forces(), d_ms(), gravity, _numNodes);
	CUDA_CHECK(cudaPeekAtLastError());
}
void Object::compRotationForce(Dvector<REAL3>& pivots, Dvector<REAL3>& rotations, REAL invdt2) {
	if (!_numFaces)
		return;
	compRotationForce_kernel << <divup(_numNodes, BLOCKSIZE), BLOCKSIZE >> > (
		d_ns(), d_forces(), d_ms(), pivots(), rotations(), d_nodePhases(), invdt2, _numNodes);
	CUDA_CHECK(cudaPeekAtLastError());
}
void Object::compPredictPosition(REAL dt) {
	if (!_numFaces)
		return;
	compPredictPosition_kernel << <divup(_numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		d_ns(), d_vs(), d_forces(), d_invMs(), dt, _numNodes);
	CUDA_CHECK(cudaPeekAtLastError());
}
void Object::updateVelocity(REAL invdt) {
	if (!_numFaces)
		return;
	updateVelocitiy_kernel << <divup(_numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		d_ns(), d_n0s(), d_vs(), invdt, _numNodes);
	CUDA_CHECK(cudaPeekAtLastError());
}
void Object::updateVelocity(Dvector<REAL>& ns, Dvector<REAL>& n0s, Dvector<REAL>& vs, REAL invdt, uint numNodes) {
	if (!_numFaces)
		return;
	updateVelocitiy_kernel << <divup(numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		ns(), n0s(), vs(), invdt, numNodes);
	CUDA_CHECK(cudaPeekAtLastError());
}
void Object::updatePosition(REAL dt) {
	if (!_numFaces)
		return;
	updatePosition_kernel << <divup(_numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		d_ns(), d_vs(), dt, _numNodes);
	CUDA_CHECK(cudaPeekAtLastError());
}
void Object::updatePosition(Dvector<REAL>& ns, Dvector<REAL>& vs, REAL dt, uint numNodes) {
	if (!numNodes)
		return;
	updatePosition_kernel << <divup(numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		ns(), vs(), dt, numNodes);
	CUDA_CHECK(cudaPeekAtLastError());
}
void Object::maxVelocitiy(REAL& maxVel) {
	if (!_numFaces)
		return;

	REAL* d_manVel;
	CUDA_CHECK(cudaMalloc((void**)&d_manVel, sizeof(REAL)));
	CUDA_CHECK(cudaMemset(d_manVel, 0, sizeof(REAL)));

	maxVelocitiy_kernel << <divup(_numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE, MAX_BLOCKSIZE * sizeof(REAL) >> > (
		d_vs(), _numNodes, d_manVel);
	CUDA_CHECK(cudaPeekAtLastError());

	CUDA_CHECK(cudaMemcpy(&maxVel, d_manVel, sizeof(REAL), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(d_manVel));
	maxVel = sqrt(maxVel);
}
void Object::computeNormal(void) {
	if (!_numFaces)
		return;
	d_nNorms.memset(0);
	compNormals_kernel << <divup(_numFaces, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		d_fs(), d_ns(), d_fNorms(), d_nNorms(), _numFaces);
	CUDA_CHECK(cudaPeekAtLastError());
	nodeNormNormalize_kernel << <divup(_numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		d_nNorms(), _numNodes);
	CUDA_CHECK(cudaPeekAtLastError());
	copyNormToHost();
}
void Object::Damping(REAL airDamp, REAL lapDamp, uint lapIter) {
	if (!_numNodes)
		return;

	airDamping_kernel << <divup(_numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		d_vs(), airDamp, _numNodes);
	CUDA_CHECK(cudaPeekAtLastError());

	/*if (lapIter) {
		Dvector<REAL> vBuffer;
		vBuffer = d_vs;
		REAL* tmp;
		for (uint i = 0u; i < lapIter; i++) {
			laplacianDamping_kernel << <divup(_numNodes, BLOCKSIZE), BLOCKSIZE >> > (
				d_vs(), vBuffer(), d_ms(), d_nbNs._array(), d_nbNs._index(), lapDamp, _numNodes);
			CUDA_CHECK(cudaPeekAtLastError());
			tmp = d_vs._list;
			d_vs._list = vBuffer._list;
			vBuffer._list = tmp;
		}
	}*/
}

void Object::draw(float* frontColor, float* backColor, bool smooth, bool phaseColor) {
	drawSurface(frontColor, backColor, smooth, phaseColor);
	if (!smooth) 
		drawWire();
}
void Object::drawWire(void)
{
	glPushMatrix();
	glDisable(GL_LIGHTING);
	glColor3d(0, 0, 0);

	for (int i = 0; i < _numFaces; i++) {
		glBegin(GL_LINE_LOOP);
		for (int j = 0; j < 3; j++) {
			auto x = h_ns[h_fs[i * 3 + j] * 3 + 0];
			auto y = h_ns[h_fs[i * 3 + j] * 3 + 1];
			auto z = h_ns[h_fs[i * 3 + j] * 3 + 2];
			glVertex3f(x, y, z);
		}
		glEnd();
	}

	glEnable(GL_LIGHTING);
	glPopMatrix();
}
void Object::drawSurface(float* frontColor, float* backColor, bool smooth, bool phaseColor)
{
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1); // turn on two-sided lighting.
	float black[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	float white[] = { 0.4f, 0.4f, 0.4f, 1.0f };
	glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, frontColor);
	glMaterialfv(GL_FRONT, GL_SPECULAR, white);
	glMaterialf(GL_FRONT, GL_SHININESS, 64);
	glMaterialfv(GL_BACK, GL_AMBIENT_AND_DIFFUSE, backColor); // back material
	glMaterialfv(GL_BACK, GL_SPECULAR, black); // no specular highlights

	if (smooth) {
		float blue[4] = { 0.0f, 0.44705882352941176470588235294118f, 0.66666666666666666666666666666667f, 1.0f };
		float yellow[4] = { 0.6f, 0.6f, 0.0f, 1.0f };
		float purple[4] = { 0.8f, 0.4f, 0.9f, 1.0f };
		float green[4] = { 0.4f, 0.9f, 0.4f, 1.0f };
		for (uint i = 0u; i < _numFaces; i++) {
			uint ino0 = h_fs[i * 3u + 0u];
			uint ino1 = h_fs[i * 3u + 1u];
			uint ino2 = h_fs[i * 3u + 2u];
			REAL3 a = make_REAL3(h_ns[ino0 * 3u + 0u], h_ns[ino0 * 3u + 1u], h_ns[ino0 * 3u + 2u]);
			REAL3 b = make_REAL3(h_ns[ino1 * 3u + 0u], h_ns[ino1 * 3u + 1u], h_ns[ino1 * 3u + 2u]);
			REAL3 c = make_REAL3(h_ns[ino2 * 3u + 0u], h_ns[ino2 * 3u + 1u], h_ns[ino2 * 3u + 2u]);

			if (phaseColor) {
				switch (h_nodePhases[ino0]) {
				case 0u:
					glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, blue);
					break;
				case 1u:
					glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, yellow);
					break;
				case 2u:
					glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, purple);
					break;
				default:
					glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, green);
					break;
				}
			}
			glBegin(GL_TRIANGLES);

			glNormal3f(h_nNorms[ino0 * 3u + 0u], h_nNorms[ino0 * 3u + 1u], h_nNorms[ino0 * 3u + 2u]);
			glVertex3f(a.x, a.y, a.z);
			glNormal3f(h_nNorms[ino1 * 3u + 0u], h_nNorms[ino1 * 3u + 1u], h_nNorms[ino1 * 3u + 2u]);
			glVertex3f(b.x, b.y, b.z);
			glNormal3f(h_nNorms[ino2 * 3u + 0u], h_nNorms[ino2 * 3u + 1u], h_nNorms[ino2 * 3u + 2u]);
			glVertex3f(c.x, c.y, c.z);

			glEnd();
		}
		/*for (uint i = 0u; i < _numFaces; i++) {
			uint ino0 = h_fs[i * 3u + 0u];
			uint ino1 = h_fs[i * 3u + 1u];
			uint ino2 = h_fs[i * 3u + 2u];
			REAL3 a = make_REAL3(h_ns[ino0 * 3u + 0u], h_ns[ino0 * 3u + 1u], h_ns[ino0 * 3u + 2u]);
			REAL3 b = make_REAL3(h_ns[ino1 * 3u + 0u], h_ns[ino1 * 3u + 1u], h_ns[ino1 * 3u + 2u]);
			REAL3 c = make_REAL3(h_ns[ino2 * 3u + 0u], h_ns[ino2 * 3u + 1u], h_ns[ino2 * 3u + 2u]);

			glBegin(GL_TRIANGLES);

			glNormal3f(h_nNorms[ino0 * 3u + 0u], h_nNorms[ino0 * 3u + 1u], h_nNorms[ino0 * 3u + 2u]);
			glVertex3f(a.x, a.y, a.z);
			glNormal3f(h_nNorms[ino1 * 3u + 0u], h_nNorms[ino1 * 3u + 1u], h_nNorms[ino1 * 3u + 2u]);
			glVertex3f(b.x, b.y, b.z);
			glNormal3f(h_nNorms[ino2 * 3u + 0u], h_nNorms[ino2 * 3u + 1u], h_nNorms[ino2 * 3u + 2u]);
			glVertex3f(c.x, c.y, c.z);

			glEnd();
		}*/
	}
	else {
		for (uint i = 0u; i < _numFaces; i++) {
			uint ino0 = h_fs[i * 3u + 0u];
			uint ino1 = h_fs[i * 3u + 1u];
			uint ino2 = h_fs[i * 3u + 2u];
			REAL3 a = make_REAL3(h_ns[ino0 * 3u + 0u], h_ns[ino0 * 3u + 1u], h_ns[ino0 * 3u + 2u]);
			REAL3 b = make_REAL3(h_ns[ino1 * 3u + 0u], h_ns[ino1 * 3u + 1u], h_ns[ino1 * 3u + 2u]);
			REAL3 c = make_REAL3(h_ns[ino2 * 3u + 0u], h_ns[ino2 * 3u + 1u], h_ns[ino2 * 3u + 2u]);

			glBegin(GL_TRIANGLES);

			glNormal3f(h_fNorms[i * 3u + 0u], h_fNorms[i * 3u + 1u], h_fNorms[i * 3u + 2u]);
			glVertex3f(a.x, a.y, a.z);
			glVertex3f(b.x, b.y, b.z);
			glVertex3f(c.x, c.y, c.z);

			glEnd();
		}
	}
}

void Object::copyToDevice(void) {
	d_fs.copyFromHost(h_fs, (*_streams)[0]);
	d_ns.copyFromHost(h_ns, (*_streams)[1]);
}
void Object::copyToHost(void) {
	d_fs.copyToHost(h_fs, (*_streams)[0]);
	d_ns.copyToHost(h_ns, (*_streams)[1]);
}
void Object::copyNbToDevice(void) {
	d_ses.copyFromHost(h_ses, &(*_streams)[2]);
	d_bes.copyFromHost(h_bes, &(*_streams)[4]);
	d_nbFs.copyFromHost(h_nbFs, &(*_streams)[6]);
	d_nbNs.copyFromHost(h_nbNs, &(*_streams)[8]);
}
void Object::copyNbToHost(void) {
	d_ses.copyToHost(h_ses, &(*_streams)[2]);
	d_bes.copyToHost(h_bes, &(*_streams)[4]);
	d_nbFs.copyToHost(h_nbFs, &(*_streams)[6]);
	d_nbNs.copyToHost(h_nbNs, &(*_streams)[8]);
}
void Object::copyMassToDevice(void) {
	d_ms.copyFromHost(h_ms, (*_streams)[9]);
	d_invMs.copyFromHost(h_invMs, (*_streams)[10]);
}
void Object::copyMassToHost(void) {
	d_ms.copyToHost(h_ms, (*_streams)[9]);
	d_invMs.copyToHost(h_invMs, (*_streams)[10]);
}
void Object::copyNormToDevice(void) {
	d_fNorms.copyFromHost(h_fNorms, (*_streams)[0]);
	d_nNorms.copyFromHost(h_nNorms, (*_streams)[1]);
}
void Object::copyNormToHost(void) {
	d_fNorms.copyToHost(h_fNorms, (*_streams)[0]);
	d_nNorms.copyToHost(h_nNorms, (*_streams)[1]);
}