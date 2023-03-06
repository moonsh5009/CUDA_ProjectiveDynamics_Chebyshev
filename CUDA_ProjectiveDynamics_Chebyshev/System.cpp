#include "System.h"

void System::init(REAL3& gravity, REAL dt) {
	_gravity = gravity;
	_dt = dt;
	_invdt = 1.0 / dt;

	_boundary._min = make_REAL3(-1.5);
	_boundary._max = make_REAL3(1.5);

	_cloths = new Cloth(30u);
	_obstacles = new Obstacle();
	_frame = 0u;
}

void System::addCloth(Mesh* mesh, REAL mass, bool isSaved) {
	_cloths->addCloth(mesh, mass, isSaved);
}
void System::addObstacle(Mesh* mesh, REAL mass, REAL3& pivot, REAL3& rotate, bool isSaved) {
	_obstacles->addObject(mesh, mass, pivot, rotate, isSaved);
}
void System::update(void) {
#if 0
	_obstacles->update(_dt, _invdt);
	_cloths->update(_gravity, _dt, _invdt);
#else
	_obstacles->update(_dt, _invdt);
	_obstacles->d_ns = _obstacles->d_n0s;
	
	//_cloths->_bvh->_isDetecteds.memset(0);

	_cloths->Damping(0.999, 0.0, 0u);
	_cloths->computeExternalForce(_gravity);
	_cloths->compPredictPosition(_dt);
	_cloths->d_ns = _cloths->d_n0s;
	//_cloths->rotateFixed(_invdt);
	{
		cudaDeviceSynchronize();
		ctimer timer = CNOW;

		REAL minTime = 1.0;
		REAL clothFriction = 0.0;
		REAL boundaryFriction = 0.0;
		REAL thickness = 0.014;

		REAL omg = 1.0, colOmg = 1.0;
		REAL underRelax = 0.9;
		REAL maxError;

		uint lastBvhSize = 1u << _cloths->_bvh->_maxLevel - 1u;
		Dvector<uint> lastBvttIds(lastBvhSize + 1u);

		ContactElems ceParam;
		ContactElemsSDF ceSDFParam;
		ceSDFParam.resize(_cloths->_numFaces, _cloths->_numNodes);

		Dvector<REAL>& Bs = _cloths->_constraints->_Bs;
		Dvector<REAL> Xs(_cloths->_numNodes * 3u);
		Dvector<REAL> Zs(_cloths->_numNodes * 3u);
		Dvector<REAL> prevNs(_cloths->_numNodes * 3u);
		Dvector<REAL> newNs(_cloths->_numNodes * 3u);

		Dvector<REAL> impulses(_cloths->_numNodes * 3u);
		Dvector<REAL> infos(_cloths->_numNodes);

		ObjParam clothParam = _cloths->param();
		ObjParam obsParam = _obstacles->param();

		Xs = _cloths->d_ns;
		_cloths->updatePosition(Xs, _cloths->d_vs, _dt, _cloths->_numNodes);
		Zs = Xs;

		ClothCollisionSolver::getClothContactElements(ceParam, clothParam, _cloths->_bvh, _cloths->_RTri,
			obsParam, _obstacles->_bvh, _obstacles->_RTri, lastBvttIds, lastBvhSize, thickness);
		ClothCollisionSolver::getClothContactElementsSDF(ceSDFParam, clothParam, _obstacles->_priTree->d_tree);

		uint itr = 0u;
		bool isDetected = false;
		while (1) {
			if (itr < 11u)			omg = 1.0;
			else if (itr == 11u)	omg = 2.0 / (2.0 - 0.9992 * 0.9992);
			else					omg = 4.0 / (4.0 - 0.9992 * 0.9992 * omg);
			itr++;

			_cloths->_constraints->jacobiProject0(Zs, _cloths->d_ms, newNs, _invdt * _invdt);
			_cloths->_constraints->jacobiProject1(Xs, newNs);
			_cloths->_constraints->jacobiProject2(
				Xs, prevNs, _cloths->d_ms, Bs, newNs, _invdt * _invdt, underRelax, omg, maxError);
			_cloths->updateVelocity(Xs, _cloths->d_n0s, _cloths->d_vs, _invdt, _cloths->_numNodes);

			if (itr < 11u)			colOmg = 1.0;
			else if (itr == 11u)	colOmg = 2.0 / (2.0 - 0.9992 * 0.9992);
			else					colOmg = 4.0 / (4.0 - 0.9992 * 0.9992 * colOmg);

			impulses.memset(0);
			infos.memset(0);
			ClothCollisionSolver::compClothBoundaryCollisionImpulse(
				clothParam, _boundary, impulses, infos, boundaryFriction, thickness, _dt);
			ClothCollisionSolver::compClothCollisionImpulse(ceParam, clothParam, obsParam, impulses, infos, clothFriction, thickness, _dt);
			//ClothCollisionSolver::compClothCollisionSDFImpulse(ceSDFParam, clothParam, impulses, infos, clothFriction, thickness, _dt);
			isDetected = ClothCollisionSolver::applyImpulse(clothParam, impulses, infos, thickness, _dt, colOmg);
			Xs = _cloths->d_ns;
			_cloths->updatePosition(Xs, _cloths->d_vs, _dt, _cloths->_numNodes);
			if (itr >= 100u)
				break;
		}
		//for (itr = 0u; itr < 100u; itr++) {
		//	impulses.memset(0);
		//	infos.memset(0);
		//	ClothCollisionSolver::compClothBoundaryCollisionImpulse(
		//		clothParam, _boundary, impulses, infos, boundaryFriction, thickness, _dt);
		//	ClothCollisionSolver::compClothCollisionImpulse(ceParam, clothParam, obsParam, impulses, infos, clothFriction, thickness, _dt);
		//	//ClothCollisionSolver::compClothCollisionSDFImpulse(ceSDFParam, clothParam, impulses, infos, clothFriction, thickness, _dt);
		//	isDetected = ClothCollisionSolver::applyImpulse(clothParam, impulses, infos, thickness, _dt, colOmg);
		//	if (!isDetected)
		//		break;
		//}

		ClothCollisionSolver::compCollisionIteration(
			ceParam, lastBvttIds, lastBvhSize, ceSDFParam, _boundary, clothParam, obsParam,
			_cloths->_bvh, _obstacles->_bvh, _cloths->_RTri, _obstacles->_RTri, _obstacles->_priTree->d_tree,
			_cloths->h_nbNs, _obstacles->h_nbNs,
			boundaryFriction, clothFriction, thickness, _dt);

		//if (isDetected) {
		//	for (itr = 0u; itr < 100u; itr++) {
		//		impulses.memset(0);
		//		infos.memset(0);
		//		ClothCollisionSolver::compClothBoundaryCollisionImpulse(
		//			clothParam, _boundary, impulses, infos, boundaryFriction, thickness, _dt);
		//		ClothCollisionSolver::compClothCollisionCCDImpulse(
		//			clothParam, _cloths->_bvh, _cloths->_RTri, obsParam, _obstacles->_bvh, _obstacles->_RTri,
		//			lastBvttIds, lastBvhSize, impulses, infos, thickness, _dt);
		//		//ClothCollisionSolver::compClothCollisionSDFImpulse(ceSDFParam, clothParam, impulses, infos, clothFriction, thickness, _dt);
		//		isDetected = ClothCollisionSolver::applyImpulse(clothParam, impulses, infos, thickness, _dt, colOmg);
		//		if (!isDetected)
		//			break;
		//	}
		//	printf("%d\n", itr);
		//}

		printf("Frame: %d\n", _frame);
		_obstacles->updatePosition(_dt);
		_cloths->updatePosition(_dt);
	}
#endif
}
void System::simulation(void) {
	ctimer timer;
	if (_frame == 0u) {
		_cloths->_degree = make_REAL3(3.0, 0.0, 0.0);
	}
	if (_frame == 1000u) {
		_cloths->_degree = make_REAL3(-3.0, 0.0, 0.0);
	}
	if (_frame == 2000u) {
		_cloths->_degree = make_REAL3(0.0, 0.0, 0.0);
	}
	/*if ((_frame % 80) == 0u) {
		for (int i = 0; i < _obstacles->h_rotations.size(); i++) {
			_obstacles->h_rotations[i] *= -1.0;
		}
		_obstacles->d_rotations = _obstacles->h_rotations;
	}*/
	/*if (_frame > 400) {
		for (int i = 0; i < _obstacles->h_rotations.size(); i++) {
			_obstacles->h_rotations[i] = make_REAL3(0.0, 1.0, 0.0);
		}
		_obstacles->d_rotations = _obstacles->h_rotations;
	}*/
	_frame++;

	printf("\n==========================\n");

	CUDA_CHECK(cudaDeviceSynchronize());
	timer = CNOW;

	update();

	CUDA_CHECK(cudaDeviceSynchronize());
	printf("Update: %f\n", (CNOW - timer) / 10000.0);
	timer = CNOW;

	_cloths->computeNormal();
	_obstacles->computeNormal();

	CUDA_CHECK(cudaDeviceSynchronize());
	printf("Compute Normals: %f\n", (CNOW - timer) / 10000.0);
	timer = CNOW;

	_cloths->copyToHost();
	_obstacles->copyToHost();

	CUDA_CHECK(cudaDeviceSynchronize());
	printf("Copy to Host: %f\n", (CNOW - timer) / 10000.0);
}
void System::reset(void) {
	_frame = 0u;
	_cloths->reset();
	_obstacles->reset();
}
void System::draw(void) {
	float blue[4] = { 0.0f, 0.44705882352941176470588235294118f, 0.66666666666666666666666666666667f, 1.0f };
	float yellow[4] = { 0.6f, 0.6f, 0.0f, 1.0f };
	float gray[4] = { 0.2f, 0.2f, 0.2f, 1.0f };

	//_cloths->draw(blue, yellow, false);
	//_obstacles->draw(gray, gray, false);
	_cloths->draw(blue, yellow, true);
	_obstacles->draw(gray, gray, true);
	//_cloths->draw(blue, yellow, true, true);
	//_obstacles->draw(gray, gray, true);

	//_cloths->_bvh->draw();
	//_obstacles->_bvh->draw();
	//_obstacles->_priTree->draw();

	//drawBoundary();
}
void System::drawBoundary(void) {
	glPushMatrix();
	glDisable(GL_LIGHTING);
	glColor3d(1, 1, 1);
	glLineWidth(3.0f);

	glBegin(GL_LINES);
	glVertex3f(_boundary._min.x, _boundary._min.y, _boundary._min.z);
	glVertex3f(_boundary._max.x, _boundary._min.y, _boundary._min.z);
	glVertex3f(_boundary._min.x, _boundary._min.y, _boundary._min.z);
	glVertex3f(_boundary._min.x, _boundary._max.y, _boundary._min.z);
	glVertex3f(_boundary._min.x, _boundary._min.y, _boundary._min.z);
	glVertex3f(_boundary._min.x, _boundary._min.y, _boundary._max.z);
	glVertex3f(_boundary._max.x, _boundary._max.y, _boundary._max.z);
	glVertex3f(_boundary._min.x, _boundary._max.y, _boundary._max.z);
	glVertex3f(_boundary._max.x, _boundary._max.y, _boundary._max.z);
	glVertex3f(_boundary._max.x, _boundary._min.y, _boundary._max.z);
	glVertex3f(_boundary._max.x, _boundary._max.y, _boundary._max.z);
	glVertex3f(_boundary._max.x, _boundary._max.y, _boundary._min.z);
	glVertex3f(_boundary._max.x, _boundary._min.y, _boundary._min.z);
	glVertex3f(_boundary._max.x, _boundary._min.y, _boundary._max.z);
	glVertex3f(_boundary._max.x, _boundary._min.y, _boundary._min.z);
	glVertex3f(_boundary._max.x, _boundary._max.y, _boundary._min.z);
	glVertex3f(_boundary._min.x, _boundary._max.y, _boundary._min.z);
	glVertex3f(_boundary._min.x, _boundary._max.y, _boundary._max.z);
	glVertex3f(_boundary._min.x, _boundary._max.y, _boundary._min.z);
	glVertex3f(_boundary._max.x, _boundary._max.y, _boundary._min.z);
	glVertex3f(_boundary._min.x, _boundary._min.y, _boundary._max.z);
	glVertex3f(_boundary._min.x, _boundary._max.y, _boundary._max.z);
	glVertex3f(_boundary._min.x, _boundary._min.y, _boundary._max.z);
	glVertex3f(_boundary._max.x, _boundary._min.y, _boundary._max.z);
	glEnd();

	glLineWidth(1.0f);
	glEnable(GL_LIGHTING);
	glPopMatrix();
}