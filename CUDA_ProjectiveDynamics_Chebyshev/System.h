#ifndef __SYSTEM_H__
#define __SYSTEM_H__

#pragma once
#include "ClothCollisionSolver.h"
#include "Cloth.h"
#include "Obstacle.h"

#define OBJ_CLOTH			1
#define OBJ_OBSTACLE		2

class System {
public:
	Cloth				*_cloths;
	Obstacle			*_obstacles;
public:
	AABB			_boundary;
public:
	REAL3			_gravity;
	REAL			_dt;
	REAL			_invdt;
	uint			_frame;
public:
	System() {}
	System(REAL3& gravity, REAL dt) {
		init(gravity, dt);
	}
	~System() {}
public:
	void	init(REAL3& gravity, REAL dt);
public:
	void	addCloth(Mesh* mesh, REAL mass, bool isSaved = true);
	void	addObstacle(Mesh* mesh, REAL mass, REAL3& pivot, REAL3& rotate, bool isSaved = true);
public:
	void	update(void);
	void	simulation(void);
	void	reset(void);
public:
	void	draw(void);
	void	drawBoundary(void);
};

#endif