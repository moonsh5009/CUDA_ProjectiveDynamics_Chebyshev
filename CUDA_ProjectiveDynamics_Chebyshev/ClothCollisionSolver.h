#ifndef __SELF_COLLISION_SOLVER_H__
#define __SELF_COLLISION_SOLVER_H__

#pragma once
#include "CollisionSolver.h"
#include "PrimalTree.h"

namespace ClothCollisionSolver {
	//--------------------------------------------------------------------------------------
	void getSelfLastBvtts(
		BVHParam& clothBvh,
		Dvector<uint2>& lastBvtts, Dvector<uint>& LastBvttIds,
		uint lastBvhSize, uint& lastBvttSize);
	void getObstacleLastBvtts(
		BVHParam& clothBvh, BVHParam& obsBvh,
		Dvector<uint2>& lastBvtts, Dvector<uint>& LastBvttIds,
		uint lastBvhSize, uint& lastBvttSize);
	//--------------------------------------------------------------------------------------
	void getClothContactElements(
		ContactElems& ceParam,
		const ObjParam& clothParam, BVHParam& clothBvh, RTriParam& clothRTri,
		const ObjParam& obsParam, BVHParam& obsBvh, RTriParam& obsRTri,
		Dvector<uint>& lastBvttIds, uint lastBvhSize,
		const REAL thickness);
	//--------------------------------------------------------------------------------------
	void getClothContactElements(
		ContactElems& ceParam,
		const ObjParam& clothParam, BVH* clothBvh, RTriangle* clothRTri,
		const ObjParam& obsParam, BVH* obsBvh, RTriangle* obsRTri,
		Dvector<uint>& lastBvttIds, uint lastBvhSize,
		const REAL thickness);
	void getClothContactElementsSDF(
		ContactElemsSDF& ceParam,
		const ObjParam& clothParam, const PRITree& priTree);
	void getClothCCDtime(
		const ObjParam& clothParam, BVH* clothBvh, RTriangle* clothRTri,
		const ObjParam& obsParam, BVH* obsBvh, RTriangle* obsRTri,
		PRITree& priTree,
		const REAL thickness, const REAL dt, REAL* minTime);
	//--------------------------------------------------------------------------------------
	void MakeClothRigidImpactZone(
		const ContactElems& d_ceParam,
		RIZone& h_riz, DRIZone& d_riz,
		const PrefixArray<uint>& clothNbNs, const PrefixArray<uint>& obsNbNs);
	bool ResolveClothRigidImpactZone(
		ContactElems& ceParam,
		RIZone& h_riz, DRIZone& d_riz,
		const ObjParam& clothParam, const ObjParam& obsParam,
		const PrefixArray<uint>& clothNbNs, const PrefixArray<uint>& obsNbNs,
		const REAL thickness, const REAL dt);
	void compClothRigidImpactZone(
		ContactElems& ceParam,
		const ObjParam& clothParam, const ObjParam& obsParam,
		const PrefixArray<uint>& clothNbNs, const PrefixArray<uint>& obsNbNs,
		const REAL thickness, const REAL dt);
	//--------------------------------------------------------------------------------------
	void compClothBoundaryCollisionImpulse(
		const ObjParam& clothParam, const AABB& boundary,
		Dvector<REAL>& impulses, Dvector<REAL>& infos, 
		const REAL friction, const REAL thickness, const REAL dt);
	void compClothCollisionSDFImpulse(
		ContactElemsSDF& ceParam, const ObjParam& clothParam,
		Dvector<REAL>& impulses, Dvector<REAL>& infos,
		const REAL friction, const REAL thickness, const REAL dt);
	void compClothCollisionImpulse(
		ContactElems& ceParam,
		const ObjParam& clothParam, const ObjParam& obsParam,
		Dvector<REAL>& impulses, Dvector<REAL>& infos,
		const REAL friction, const REAL thickness, const REAL dt);
	void compClothCollisionCCDImpulse(
		const ObjParam& clothParam, BVH* clothBvh, RTriangle* clothRTri,
		const ObjParam& obsParam, BVH* obsBvh, RTriangle* obsRTri,
		Dvector<uint>& lastBvttIds, uint lastBvhSize,
		Dvector<REAL>& impulses, Dvector<REAL>& infos,
		const REAL thickness, const REAL dt);
	//--------------------------------------------------------------------------------------
	bool applyImpulse(
		const ObjParam& clothParam, 
		Dvector<REAL>& impulses, Dvector<REAL>& infos,
		REAL thickness, REAL dt, REAL omg);
	void updatePosition(
		const ObjParam& clothParam, const ObjParam& obsParam, REAL dt);
	void updateVelocity(
		const ObjParam& clothParam, const ObjParam& obsParam, REAL dt);
	void compCollisionIteration(
		ContactElems& ceParam, Dvector<uint>& lastBvttIds, uint lastBvhSize, 
		ContactElemsSDF& ceSDFParam, const AABB& boundary,
		const ObjParam& clothParam, const ObjParam& obsParam, 
		BVH* clothBvh, BVH* obsBvh, RTriangle* clothRTri, RTriangle* obsRTri, PRITree& priTree,
		const PrefixArray<uint>& clothNbNs, const PrefixArray<uint>& obsNbNs,
		const REAL boundaryFriction, const REAL clothFriction, const REAL thickness, const REAL dt);
	//--------------------------------------------------------------------------------------
};

#endif