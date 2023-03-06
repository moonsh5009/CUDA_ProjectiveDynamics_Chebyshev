#include "CollisionDetection.cuh"
#include "CollisionResponse.cuh"
#include "ClothCollisionSolver.h"

//-------------------------------------------------------------------------
void ClothCollisionSolver::getSelfLastBvtts(
	BVHParam& clothBvh,
	Dvector<uint2>& lastBvtts, Dvector<uint>& LastBvttIds,
	uint lastBvhSize, uint& lastBvttSize)
{
#ifdef COLLISION_TESTTIMER
	CUDA_CHECK(cudaDeviceSynchronize());
	ctimer timer = CNOW;
#endif

	getNumLastBvtts_kernel << <divup(lastBvhSize, BLOCKSIZE), BLOCKSIZE >> > (
		clothBvh, LastBvttIds(), lastBvhSize);
	CUDA_CHECK(cudaPeekAtLastError());

	thrust::inclusive_scan(thrust::device_ptr<uint>(LastBvttIds.begin()), 
		thrust::device_ptr<uint>(LastBvttIds.begin() + lastBvhSize + 1u),
		thrust::device_ptr<uint>(LastBvttIds.begin()));
	CUDA_CHECK(cudaMemcpy(&lastBvttSize, LastBvttIds() + lastBvhSize, sizeof(uint), cudaMemcpyDeviceToHost));

	if (lastBvtts.size() < lastBvttSize)
		lastBvtts.resize(lastBvttSize);
	getLastBvtts_kernel << <divup(lastBvhSize, BLOCKSIZE), BLOCKSIZE >> > (
		clothBvh, lastBvtts(), LastBvttIds(), lastBvhSize);
	CUDA_CHECK(cudaPeekAtLastError());

#ifdef COLLISION_TESTTIMER
	CUDA_CHECK(cudaDeviceSynchronize());
	printf("getSelfLastBvtts: %lf msec\n", (CNOW - timer) / 10000.0);
#endif
}
void ClothCollisionSolver::getObstacleLastBvtts(
	BVHParam& clothBvh, BVHParam& obsBvh,
	Dvector<uint2>& lastBvtts, Dvector<uint>& LastBvttIds,
	uint lastBvhSize, uint& lastBvttSize)
{
#ifdef COLLISION_TESTTIMER
	CUDA_CHECK(cudaDeviceSynchronize());
	ctimer timer = CNOW;
#endif

	getNumLastBvtts_kernel << <divup(lastBvhSize, BLOCKSIZE), BLOCKSIZE >> > (
		clothBvh, obsBvh, LastBvttIds(), lastBvhSize);
	CUDA_CHECK(cudaPeekAtLastError());

	thrust::inclusive_scan(thrust::device_ptr<uint>(LastBvttIds.begin()),
		thrust::device_ptr<uint>(LastBvttIds.begin() + lastBvhSize + 1u),
		thrust::device_ptr<uint>(LastBvttIds.begin()));
	CUDA_CHECK(cudaMemcpy(&lastBvttSize, LastBvttIds() + lastBvhSize, sizeof(uint), cudaMemcpyDeviceToHost));

	if (lastBvtts.size() < lastBvttSize)
		lastBvtts.resize(lastBvttSize);
	getLastBvtts_kernel << <divup(lastBvhSize, BLOCKSIZE), BLOCKSIZE >> > (
		clothBvh, obsBvh, lastBvtts(), LastBvttIds(), lastBvhSize);
	CUDA_CHECK(cudaPeekAtLastError());

#ifdef COLLISION_TESTTIMER
	CUDA_CHECK(cudaDeviceSynchronize());
	printf("getObstacleLastBvtts: %lf msec\n", (CNOW - timer) / 10000.0);
#endif
}
//-------------------------------------------------------------------------
void ClothCollisionSolver::getClothContactElements(
	ContactElems& ceParam,
	const ObjParam& clothParam, BVHParam& clothBvh, RTriParam& clothRTri,
	const ObjParam& obsParam, BVHParam& obsBvh, RTriParam& obsRTri,
	Dvector<uint>& lastBvttIds, uint lastBvhSize,
	const REAL thickness)
{
#ifdef COLLISION_TESTTIMER
	CUDA_CHECK(cudaDeviceSynchronize());
	ctimer timer = CNOW;
#endif

#ifndef USED_LASTBVTT
	ceParam._size = 0u;
	ceParam.d_tmp.memset(0);

	if (clothParam._numFaces) {
		getNumSelfContactElements_kernel << <divup(clothParam._numFaces, BLOCKSIZE), BLOCKSIZE >> > (
			ceParam.param(), clothParam, clothBvh, clothRTri, thickness);
		CUDA_CHECK(cudaPeekAtLastError());
	}
	if (obsParam._numFaces) {
		getNumObstacleContactElements_kernel << <divup(clothParam._numFaces, BLOCKSIZE), BLOCKSIZE >> > (
			ceParam.param(), clothParam, clothBvh, clothRTri, obsParam, obsBvh, obsRTri, thickness);
		CUDA_CHECK(cudaPeekAtLastError());
	}

	CUDA_CHECK(cudaMemcpy(&ceParam._size, ceParam.d_tmp(), sizeof(uint), cudaMemcpyDeviceToHost));
	if (ceParam._size) {
		if (ceParam._elems.size() < ceParam._size)
			ceParam.resize();

		ceParam.d_tmp.memset(0);

		if (clothParam._numFaces) {
			getSelfContactElements_kernel << <divup(clothParam._numFaces, BLOCKSIZE), BLOCKSIZE >> > (
				ceParam.param(), clothParam, clothBvh, clothRTri, thickness);
			CUDA_CHECK(cudaPeekAtLastError());
		}
		if (obsParam._numFaces) {
			getObstacleContactElements_kernel << <divup(clothParam._numFaces, BLOCKSIZE), BLOCKSIZE >> > (
				ceParam.param(), clothParam, clothBvh, clothRTri, obsParam, obsBvh, obsRTri, thickness);
			CUDA_CHECK(cudaPeekAtLastError());
		}
		CUDA_CHECK(cudaMemcpy(&ceParam._size, ceParam.d_tmp(), sizeof(uint), cudaMemcpyDeviceToHost));
	}
#else
	ceParam._size = 0u;
	ceParam.d_tmp.memset(0);

	Dvector<uint2> selfLastBvtts;
	Dvector<uint2> obsLastBvtts;
	uint selfLastBvttSize, obsLastBvttSize;
	getSelfLastBvtts(
		clothBvh, selfLastBvtts, lastBvttIds, lastBvhSize, selfLastBvttSize);
#ifndef USED_SDF
	getObstacleLastBvtts(
		clothBvh, obsBvh, obsLastBvtts, lastBvttIds, lastBvhSize, obsLastBvttSize);
#endif

	if (selfLastBvttSize) {
		getNumSelfContactElements_LastBvtt_kernel << <divup(selfLastBvttSize, BLOCKSIZE), BLOCKSIZE, BLOCKSIZE * sizeof(uint) >> > (
			clothParam, clothBvh, clothRTri,
			selfLastBvtts(), selfLastBvttSize, ceParam.d_tmp(),
			thickness);
		CUDA_CHECK(cudaPeekAtLastError());
	}
#ifndef USED_SDF
	if (obsLastBvttSize) {
		getNumObstacleContactElements_LastBvtt_kernel << <divup(obsLastBvttSize, BLOCKSIZE), BLOCKSIZE, BLOCKSIZE * sizeof(uint) >> > (
			clothParam, clothBvh, clothRTri,
			obsParam, obsBvh, obsRTri,
			obsLastBvtts(), obsLastBvttSize, ceParam.d_tmp(),
			thickness);
		CUDA_CHECK(cudaPeekAtLastError());
	}
#endif
	CUDA_CHECK(cudaMemcpy(&ceParam._size, ceParam.d_tmp(), sizeof(uint), cudaMemcpyDeviceToHost));
	if (ceParam._size > 0u) {
		if (ceParam._elems.size() < ceParam._size)
			ceParam.resize();
		ceParam.d_tmp.memset(0);
		if (selfLastBvttSize) {
			getSelfContactElements_LastBvtt_kernel << <divup(selfLastBvttSize, BLOCKSIZE), BLOCKSIZE >> > (
				clothParam, clothBvh, clothRTri,
				selfLastBvtts(), selfLastBvttSize, ceParam.param(),
				thickness);
			CUDA_CHECK(cudaPeekAtLastError());
		}
#ifndef USED_SDF
		if (obsLastBvttSize) {
			getObstacleContactElements_LastBvtt_kernel << <divup(obsLastBvttSize, BLOCKSIZE), BLOCKSIZE >> > (
				clothParam, clothBvh, clothRTri,
				obsParam, obsBvh, obsRTri,
				obsLastBvtts(), obsLastBvttSize, ceParam.param(),
				thickness);
			CUDA_CHECK(cudaPeekAtLastError());
		}
#endif
		CUDA_CHECK(cudaMemcpy(&ceParam._size, ceParam.d_tmp(), sizeof(uint), cudaMemcpyDeviceToHost));
	}
#endif

#ifdef COLLISION_TESTTIMER
	CUDA_CHECK(cudaDeviceSynchronize());
	printf("get Cloth Collision Proximity Elements: %lf msec\n", (CNOW - timer) / 10000.0);
#endif
}
//-------------------------------------------------------------------------
void ClothCollisionSolver::getClothContactElements(
	ContactElems& ceParam,
	const ObjParam& clothParam, BVH* clothBvh, RTriangle* clothRTri,
	const ObjParam& obsParam, BVH* obsBvh, RTriangle* obsRTri,
	Dvector<uint>& lastBvttIds, uint lastBvhSize,
	const REAL thickness)
{
	clothBvh->refit(clothParam._fs, clothParam._ns, thickness);
	obsBvh->refit(obsParam._fs, obsParam._ns, thickness);
	getClothContactElements(
		ceParam,
		clothParam, clothBvh->param(), clothRTri->param(),
		obsParam, obsBvh->param(), obsRTri->param(),
		lastBvttIds, lastBvhSize,
		thickness);
}
void ClothCollisionSolver::getClothContactElementsSDF(
	ContactElemsSDF& ceParam,
	const ObjParam& clothParam, const PRITree& priTree)
{
#ifdef USED_SDF
	getClothCollisionElementsSDF_T_kernel << <divup(clothParam._numFaces, BLOCKSIZE), BLOCKSIZE >> > (
		ceParam.param(), clothParam, priTree);
	CUDA_CHECK(cudaPeekAtLastError());
	getClothCollisionElementsSDF_V_kernel << <divup(clothParam._numNodes, BLOCKSIZE), BLOCKSIZE >> > (
		ceParam.param(), clothParam, priTree);
	CUDA_CHECK(cudaPeekAtLastError());
#endif
}
void ClothCollisionSolver::getClothCCDtime(
	const ObjParam& clothParam, BVH* clothBvh, RTriangle* clothRTri,
	const ObjParam& obsParam, BVH* obsBvh, RTriangle* obsRTri,
	PRITree& priTree,
	const REAL thickness, const REAL dt, REAL* minTime)
{
	REAL CCD_thickness = thickness * COL_CCD_THICKNESS_DETECT;
	//REAL CCD_thickness = thickness * COL_THICKNESS_RATIO;
	//REAL CCD_thickness = thickness;
	//*minTime = REAL_MAX;
	*minTime = 10.0;

	REAL* d_minTime;
	CUDA_CHECK(cudaMalloc((void**)&d_minTime, sizeof(REAL)));
	CUDA_CHECK(cudaMemcpy(d_minTime, minTime, sizeof(REAL), cudaMemcpyHostToDevice));

	clothBvh->refit(clothParam._fs, clothParam._ns, clothParam._vs, CCD_thickness, dt);
	obsBvh->refit(obsParam._fs, obsParam._ns, obsParam._vs, CCD_thickness, dt);

	if (clothParam._numFaces) {
		getSelfCCDtime_kernel << <divup(clothParam._numFaces, BLOCKSIZE), BLOCKSIZE >> > (
			clothParam, clothBvh->param(), clothRTri->param(),
			thickness, dt, d_minTime);
		CUDA_CHECK(cudaPeekAtLastError());
#ifndef USED_SDF
		if (obsParam._numFaces) {
			getObstacleCCDtime_kernel << <divup(clothParam._numFaces, BLOCKSIZE), BLOCKSIZE >> > (
				clothParam, clothBvh->param(), clothRTri->param(),
				obsParam, obsBvh->param(), obsRTri->param(),
				thickness, dt, d_minTime);
			CUDA_CHECK(cudaPeekAtLastError());
		}
#else
		/*getSDFCCDtime_T_kernel << <divup(clothParam._numFaces, BLOCKSIZE), BLOCKSIZE >> > (
			clothParam, priTree, thickness, dt, d_minTime);
		CUDA_CHECK(cudaPeekAtLastError());
		getSDFCCDtime_V_kernel << <divup(clothParam._numNodes, BLOCKSIZE), BLOCKSIZE >> > (
			clothParam, priTree, thickness, dt, d_minTime);
		CUDA_CHECK(cudaPeekAtLastError());*/
#endif
	}

	CUDA_CHECK(cudaMemcpy(minTime, d_minTime, sizeof(REAL), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(d_minTime));
}
//-------------------------------------------------------------------------
void ClothCollisionSolver::MakeClothRigidImpactZone(
	const ContactElems& d_ceParam,
	RIZone& h_riz, DRIZone& d_riz,
	const PrefixArray<uint>& clothNbNs, const PrefixArray<uint>& obsNbNs)
{
	vector<ContactElem> h_ceParam(d_ceParam._size);
	CUDA_CHECK(cudaMemcpy(&h_ceParam[0], d_ceParam._elems(), d_ceParam._size * sizeof(ContactElem), cudaMemcpyDeviceToHost));

	uint2 nodes[4];
	REAL w[4];
	for (uint ice = 0; ice < h_ceParam.size(); ice++) {
		if (h_ceParam[ice]._info == 0.0)
			continue;

		nodes[0].x = h_ceParam[ice]._i[0];
		nodes[1].x = h_ceParam[ice]._i[1];
		nodes[2].x = h_ceParam[ice]._i[2];
		nodes[3].x = h_ceParam[ice]._i[3];
		if (!h_ceParam[ice]._isObs)
			nodes[0].y = nodes[1].y = nodes[2].y = nodes[3].y = 0u;
		else if (h_ceParam[ice]._isObs == 1u) {
			if (h_ceParam[ice]._isFV) {
				nodes[0].y = nodes[1].y = nodes[2].y = 0u;
				nodes[3].y = 1u;
			}
			else {
				nodes[0].y = nodes[1].y = 0u;
				nodes[2].y = nodes[3].y = 1u;
			}
		}
		else {
			if (h_ceParam[ice]._isFV) {
				nodes[0].y = nodes[1].y = nodes[2].y = 1u;
				nodes[3].y = 0u;
			}
			else {
				nodes[0].y = nodes[1].y = 1u;
				nodes[2].y = nodes[3].y = 0u;
			}
		}
		w[0] = h_ceParam[ice]._w[0];
		w[1] = h_ceParam[ice]._w[1];
		if (h_ceParam[ice]._isFV) {
			w[2] = 1.0 - w[0] - w[1];
			w[3] = 1.0;
		}
		else {
			w[3] = w[1]; w[1] = w[0];
			w[0] = 1.0 - w[1];
			w[2] = 1.0 - w[3];
		}

		set<uint> ind_inc;
		for (uint i = 0; i < 4; i++) {
			uint2 ino = nodes[i];
			if (w[i] <= 0.0)
				continue;
			for (uint iriz = 0; iriz < h_riz.size(); iriz++) {
				if (h_riz[iriz].find(ino) != h_riz[iriz].end())
					ind_inc.insert(iriz);
				else {
					if (ino.y == 0u) {
						for (uint j = clothNbNs._index[ino.x]; j < clothNbNs._index[ino.x + 1u]; j++) {
							uint2 jno = make_uint2(clothNbNs._array[j], 0u);
							if (h_riz[iriz].find(jno) != h_riz[iriz].end()) {
								ind_inc.insert(iriz);
								break;
							}
						}
					}
					else {
						for (uint j = obsNbNs._index[ino.x]; j < obsNbNs._index[ino.x + 1u]; j++) {
							uint2 jno = make_uint2(obsNbNs._array[j], 1u);
							if (h_riz[iriz].find(jno) != h_riz[iriz].end()) {
								ind_inc.insert(iriz);
								break;
							}
						}
					}
				}
			}
		}
		uint ind0;
		if (ind_inc.size() == 0) {
			ind0 = (uint)h_riz.size();
			h_riz.resize(ind0 + 1u);
		}
		else if (ind_inc.size() == 1u)
			ind0 = *(ind_inc.begin());
		else {
			RIZone h_riz1;
			for (uint iriz = 0; iriz < h_riz.size(); iriz++) {
				if (ind_inc.find(iriz) != ind_inc.end()) continue;
				h_riz1.push_back(h_riz[iriz]);
			}
			ind0 = (uint)h_riz1.size();
			h_riz1.resize(ind0 + 1);
			for (auto itr = ind_inc.begin(); itr != ind_inc.end(); itr++) {
				uint ind1 = *itr;
				for (auto jtr = h_riz[ind1].begin(); jtr != h_riz[ind1].end(); jtr++)
					h_riz1[ind0].insert(*jtr);
			}
			h_riz = h_riz1;
		}

		for (uint i = 0; i < 4; i++)
			h_riz[ind0].insert(nodes[i]);
	}

	vector<uint2> h_ids;
	vector<uint> h_zones;
	h_zones.resize(h_riz.size() + 1u);
	h_zones[0] = 0;
	for (uint i = 0; i < h_riz.size(); i++)
		h_zones[i + 1] = h_zones[i] + h_riz[i].size();

	h_ids.resize(h_zones.back());
	uint n = 0;
	for (uint i = 0; i < h_riz.size(); i++) {
		for (auto jtr = h_riz[i].begin(); jtr != h_riz[i].end(); jtr++)
			h_ids[n++] = *jtr;
	}

	d_riz._ids = h_ids;
	d_riz._zones = h_zones;
}
bool ClothCollisionSolver::ResolveClothRigidImpactZone(
	ContactElems& ceParam,
	RIZone& h_riz, DRIZone& d_riz,
	const ObjParam& clothParam, const ObjParam& obsParam,
	const PrefixArray<uint>& clothNbNs, const PrefixArray<uint>& obsNbNs,
	const REAL thickness, const REAL dt)
{
#ifdef COLLISION_TESTTIMER
	CUDA_CHECK(cudaDeviceSynchronize());
	ctimer timer = CNOW;
#endif
	bool result = false;
	if (ceParam._size) {
		h_riz.clear();
		d_riz.clear();

		bool* d_applied;

		CUDA_CHECK(cudaDeviceSynchronize());
		ctimer timer = CNOW;

		CUDA_CHECK(cudaMalloc((void**)&d_applied, sizeof(bool)));
		CUDA_CHECK(cudaMemset(d_applied, 0, sizeof(bool)));

		resetCE_kernel << <divup(ceParam._size, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
			ceParam.param());
		CUDA_CHECK(cudaPeekAtLastError());
		compClothDetected_CE_kernel << <divup(ceParam._size, BLOCKSIZE), BLOCKSIZE >> > (
			ceParam.param(), clothParam, obsParam, thickness, dt, d_applied);
		CUDA_CHECK(cudaPeekAtLastError());

		CUDA_CHECK(cudaMemcpy(&result, d_applied, sizeof(bool), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaFree(d_applied));

		if (result) {
			MakeClothRigidImpactZone(ceParam, h_riz, d_riz, clothNbNs, obsNbNs);
			ApplyClothRigidImpactZone_kernel << <divup(d_riz._zones.size() - 1u, BLOCKSIZE), BLOCKSIZE >> > (
				clothParam, obsParam, d_riz.param(), dt);
			CUDA_CHECK(cudaPeekAtLastError());
		}
	}

	return result;

#ifdef COLLISION_TESTTIMER
	CUDA_CHECK(cudaDeviceSynchronize());
	printf("Resolve Cloth Collision RIZ: %lf msec\n", (CNOW - timer) / 10000.0);
#endif
}
void ClothCollisionSolver::compClothRigidImpactZone(
	ContactElems& ceParam,
	const ObjParam& clothParam, const ObjParam& obsParam,
	const PrefixArray<uint>& clothNbNs, const PrefixArray<uint>& obsNbNs,
	const REAL thickness, const REAL dt) 
{
	RIZone h_riz;
	DRIZone d_riz;
	uint itr = 0u;
	//Dvector<uint> clothZones(clothParam._numNodes);
	//Dvector<uint> obsZones(obsParam._numNodes);
	while (ResolveClothRigidImpactZone(ceParam, h_riz, d_riz, clothParam, obsParam, clothNbNs, obsNbNs, thickness, dt))
		itr++;

	//-------------------------------------------------------------------------------
	//CUDA_CHECK(cudaDeviceSynchronize());
	//printf("step 3: %lf msec\n", (CNOW - timer) / 10000.0);
	//timer = CNOW;
	//-------------------------------------------------------------------------------
	if (itr > 0u) printf("Rigid Impact Zone %d\n", itr);
}
//-------------------------------------------------------------------------
void ClothCollisionSolver::compClothBoundaryCollisionImpulse(
	const ObjParam& clothParam, const AABB& boundary,
	Dvector<REAL>& impulses, Dvector<REAL>& infos,
	const REAL friction, const REAL thickness, const REAL dt)
{
	compClothBoundaryCollisionImpulse_kernel << <divup(clothParam._numNodes, BLOCKSIZE), BLOCKSIZE >> > (
		clothParam, boundary, impulses(), infos(), friction, thickness * COL_THICKNESS_RATIO, dt);
	CUDA_CHECK(cudaPeekAtLastError());
}
void ClothCollisionSolver::compClothCollisionSDFImpulse(
	ContactElemsSDF& ceParam, const ObjParam& clothParam, 
	Dvector<REAL>& impulses, Dvector<REAL>& infos,
	const REAL friction, const REAL thickness, const REAL dt)
{
#ifdef USED_SDF
	compClothCollisionSDFImpulse_T_kernel << <divup(clothParam._numFaces, BLOCKSIZE), BLOCKSIZE >> > (
		ceParam.param(), clothParam, impulses(), infos(), friction, thickness, dt);
	CUDA_CHECK(cudaPeekAtLastError());
	compClothCollisionSDFImpulse_V_kernel << <divup(clothParam._numNodes, BLOCKSIZE), BLOCKSIZE >> > (
		ceParam.param(), clothParam, impulses(), infos(), friction, thickness, dt);
	CUDA_CHECK(cudaPeekAtLastError());
#endif
}
void ClothCollisionSolver::compClothCollisionImpulse(
	ContactElems& ceParam,
	const ObjParam& clothParam, const ObjParam& obsParam,
	Dvector<REAL>& impulses, Dvector<REAL>& infos,
	const REAL friction, const REAL thickness, const REAL dt)
{
	if (ceParam._size > 0u) {
		compClothCollisionImpulse_CE_kernel << <divup(ceParam._size, BLOCKSIZE), BLOCKSIZE >> > (
			ceParam.param(), clothParam, obsParam, impulses(), infos(), friction, thickness, dt);
		CUDA_CHECK(cudaPeekAtLastError());
	}
}
void ClothCollisionSolver::compClothCollisionCCDImpulse(
	const ObjParam& clothParam, BVH* clothBvh, RTriangle* clothRTri,
	const ObjParam& obsParam, BVH* obsBvh, RTriangle* obsRTri,
	Dvector<uint>& lastBvttIds, uint lastBvhSize,
	Dvector<REAL>& impulses, Dvector<REAL>& infos,
	const REAL thickness, const REAL dt)
{
#ifdef COLLISION_TESTTIMER
		CUDA_CHECK(cudaDeviceSynchronize());
		ctimer timer = CNOW;
#endif
		clothBvh->refit(clothParam._fs, clothParam._ns, clothParam._vs, thickness * COL_CCD_THICKNESS_DETECT, dt);
		obsBvh->refit(obsParam._fs, obsParam._ns, obsParam._vs, thickness * COL_CCD_THICKNESS_DETECT, dt);

		BVHParam clothBVHParam = clothBvh->param();
		BVHParam obsBVHParam = obsBvh->param();
		RTriParam clothRTriParam = clothRTri->param();
		RTriParam obsRTriParam = obsRTri->param();

		Dvector<uint2> selfLastBvtts;
		Dvector<uint2> obsLastBvtts;
		uint selfLastBvttSize, obsLastBvttSize;
		getSelfLastBvtts(
			clothBVHParam, selfLastBvtts, lastBvttIds, lastBvhSize, selfLastBvttSize);
#ifndef USED_SDF
		getObstacleLastBvtts(
			clothBVHParam, obsBVHParam, obsLastBvtts, lastBvttIds, lastBvhSize, obsLastBvttSize);
#endif

		if (selfLastBvttSize) {
			compSelfCollisionCCDImpulse_LastBvtt_kernel << <divup(selfLastBvttSize, BLOCKSIZE), BLOCKSIZE >> > (
				clothParam, clothBVHParam, clothRTriParam,
				selfLastBvtts(), selfLastBvttSize, 
				impulses(), infos(), thickness, dt);
			CUDA_CHECK(cudaPeekAtLastError());
		}
#ifndef USED_SDF
		if (obsLastBvttSize) {
			compObstacleCollisionCCDImpulse_LastBvtt_kernel << <divup(obsLastBvttSize, BLOCKSIZE), BLOCKSIZE >> > (
				clothParam, clothBVHParam, clothRTriParam,
				obsParam, obsBVHParam, obsRTriParam,
				obsLastBvtts(), obsLastBvttSize,
				impulses(), infos(), thickness, dt);
			CUDA_CHECK(cudaPeekAtLastError());
		}
#endif
#ifdef COLLISION_TESTTIMER
		CUDA_CHECK(cudaDeviceSynchronize());
		printf("ClothCollisionSolver::compClothCollisionSDFImpulse: %lf msec\n", (CNOW - timer) / 10000.0);
#endif
}
//-------------------------------------------------------------------------
bool ClothCollisionSolver::applyImpulse(
	const ObjParam& clothParam,
	Dvector<REAL>& impulses, Dvector<REAL>& infos,
	REAL thickness, REAL dt, REAL omg)
{
	bool result = false;
	bool* d_applied;

	CUDA_CHECK(cudaMalloc((void**)&d_applied, sizeof(bool)));
	CUDA_CHECK(cudaMemset(d_applied, 0, sizeof(bool)));

	applyClothCollision_kernel << <divup(clothParam._numNodes, BLOCKSIZE), BLOCKSIZE >> > (
		clothParam, impulses(), infos(), thickness, dt, omg, d_applied);
	CUDA_CHECK(cudaPeekAtLastError());

	CUDA_CHECK(cudaMemcpy(&result, d_applied, sizeof(bool), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(d_applied));

	return result;
}
void ClothCollisionSolver::updatePosition(
	const ObjParam& clothParam, const ObjParam& obsParam, REAL dt)
{
	updatePosition_kernel << <divup(clothParam._numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		clothParam, dt);
	CUDA_CHECK(cudaPeekAtLastError());
	updatePosition_kernel << <divup(obsParam._numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		obsParam, dt);
	CUDA_CHECK(cudaPeekAtLastError());
}
void ClothCollisionSolver::updateVelocity(
	const ObjParam& clothParam, const ObjParam& obsParam, REAL dt)
{
	updateVelocity_kernel << <divup(clothParam._numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		clothParam, 1.0 / dt);
	CUDA_CHECK(cudaPeekAtLastError());
	updateVelocity_kernel << <divup(obsParam._numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		obsParam, 1.0 / dt);
	CUDA_CHECK(cudaPeekAtLastError());
}
void ClothCollisionSolver::compCollisionIteration(
	ContactElems& ceParam, Dvector<uint>& lastBvttIds, uint lastBvhSize, 
	ContactElemsSDF& ceSDFParam, const AABB& boundary,
	const ObjParam& clothParam, const ObjParam& obsParam,
	BVH* clothBvh, BVH* obsBvh, RTriangle* clothRTri, RTriangle* obsRTri, PRITree& priTree,
	const PrefixArray<uint>& clothNbNs, const PrefixArray<uint>& obsNbNs,
	const REAL boundaryFriction, const REAL clothFriction, const REAL thickness, const REAL dt)
{
	bool isDetected;
	Dvector<REAL> impulses(clothParam._numNodes * 3u);
	Dvector<REAL> ns(clothParam._numNodes * 3u);
	Dvector<REAL> infos(clothParam._numNodes);
	REAL subDt = dt;
	uint itr;
	REAL minTime = 0.0;

	ClothCollisionSolver::getClothCCDtime(clothParam, clothBvh, clothRTri,
		obsParam, obsBvh, obsRTri, priTree,
		thickness, subDt, &minTime);

	while (minTime <= 1.0) {
		printf("%.20f\n", minTime);
		minTime *= 0.9;
		updatePosition(clothParam, obsParam, subDt * minTime);
		subDt -= subDt * minTime;

		ClothCollisionSolver::getClothContactElements(ceParam, clothParam, clothBvh, clothRTri,
			obsParam, obsBvh, obsRTri, lastBvttIds, lastBvhSize, thickness);
		for (itr = 0u; itr < 150u; itr++) {
			impulses.memset(0);
			infos.memset(0);
			ClothCollisionSolver::compClothBoundaryCollisionImpulse(
				clothParam, boundary, impulses, infos, boundaryFriction, thickness, subDt);
			ClothCollisionSolver::compClothCollisionImpulse(ceParam, clothParam, obsParam, impulses, infos, clothFriction, thickness, subDt);
			isDetected = ClothCollisionSolver::applyImpulse(clothParam, impulses, infos, thickness, subDt, 0.0);
			if (!isDetected)
				break;
		}

		if ((minTime < 0.1 || subDt > 0.8 * dt) && isDetected)
			compClothRigidImpactZone(ceParam, clothParam, obsParam, clothNbNs, obsNbNs, thickness, subDt);

		ClothCollisionSolver::getClothCCDtime(clothParam, clothBvh, clothRTri,
			obsParam, obsBvh, obsRTri, priTree,
			thickness, subDt, &minTime);
	}
	updatePosition(clothParam, obsParam, subDt);
	updateVelocity(clothParam, obsParam, dt);
}