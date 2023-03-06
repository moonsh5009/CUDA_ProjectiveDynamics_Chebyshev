#include "PrimalTree.cuh"

inline __global__ void getDistSqrKDPointToTri(
	KDTree kdTree,
	ObjParam objParams,
	const REAL3* points,
	const uint pointNum,
	REAL* output)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= pointNum)
		return;
	REAL3 p = points[id];
	output[id] = getDistSqrKD(p, kdTree, objParams);;
}
inline __global__ void subdivPRINodeInit(
	KDTree kdTree,
	ObjParam objParams,
	PRINode* nodes,
	REAL3 center,
	REAL nodeSide,
	uint* ichild)
{
	uint id = threadIdx.x;
	PRINode node;
	if (id == 0) {
		node.level = 0;
		node.pos.x = node.pos.y = node.pos.z = nodeSide * -0.5;
		node.child = 1;
		*ichild = 9;
	}
	else {
		REAL nx = (REAL)(--id & 1) - 0.5;
		REAL ny = (REAL)((id >> 1) & 1) - 0.5;
		REAL nz = (REAL)((id++ >> 2) & 1) - 0.5;
		node.level = -1;
		node.pos = make_REAL3(nx * nodeSide, ny * nodeSide, nz * nodeSide);
	}

	REAL3 point = node.pos + center;
	node.dist = ssqrt(getDistSqrKD(point, kdTree, objParams));
	nodes[id] = node;
}
inline __global__ void subdivPRINode(
	KDTree kdTree,
	ObjParam objParams,
	PRINode* nodes,
	uint nodeStart,
	uint nodeNum,
	REAL3 center,
	REAL nodeSide,
	REAL nodeCross,
	REAL error,
	uint* ichild)
{
	/*__shared__ bool s_isSubdivs[16];
	__shared__ uint s_ichilds[16];
	__shared__ REAL s_dists[BLOCKSIZE];
	__shared__ REAL s_dists2[BLOCKSIZE];
	uint id = (blockDim.x * blockIdx.x + threadIdx.x) >> 3;
	if (id >= nodeNum)
		return;
	uint inode = threadIdx.x >> 3;
	uint icd = threadIdx.x & 7;
	if (!icd) s_isSubdivs[inode] = false;
	__syncthreads();

	PRINode node = nodes[nodeStart + id];
	REAL3 point = node.pos + center;
	point.x += (threadIdx.x & 1) * nodeSide;
	point.y += ((threadIdx.x >> 1) & 1) * nodeSide;
	point.z += ((threadIdx.x >> 2) & 1) * nodeSide;
	if (icd)	s_dists[threadIdx.x] = ssqrt(getDistSqrKD(point, kdTree, objParams));
	else		s_dists[threadIdx.x] = node.dist;
	if (fabs(s_dists[threadIdx.x]) <= nodeCross)
		s_isSubdivs[inode] = true;
	__syncthreads();

	if (!s_isSubdivs[inode])
		return;

	REAL nodeSide2 = nodeSide + nodeSide;
	REAL3 point2 = node.pos + center;
	point2.x += (threadIdx.x & 1) * nodeSide2;
	point2.y += ((threadIdx.x >> 1) & 1) * nodeSide2;
	point2.z += ((threadIdx.x >> 2) & 1) * nodeSide2;
	if (icd)	s_dists2[threadIdx.x] = ssqrt(getDistSqrKD(point2, kdTree, objParams));
	__syncthreads();

	if (!icd) {
		REAL3 corners[7];
		REAL dists[15];
		for (uint i = 1; i < 8; i++) {
			dists[i + 6] = s_dists[threadIdx.x + i];
			dists[i - 1] = s_dists2[threadIdx.x + i];
			corners[i - 1] = node.pos + center;
			corners[i - 1].x += (i & 1) * nodeSide2;
			corners[i - 1].y += ((i >> 1) & 1) * nodeSide2;
			corners[i - 1].z += ((i >> 2) & 1) * nodeSide2;
		}
		dists[14] = node.dist;
		if (ComparisonPRI(kdTree, objParams, dists, corners, error))
			s_isSubdivs[inode] = false;
		else {
			node.child = atomicAdd(ichild, 8);
			node.level = -node.level;
			nodes[nodeStart + id] = node;
			s_ichilds[inode] = node.child;
			node.level = -node.level;
		}
	}
	__syncthreads();
	if (!s_isSubdivs[inode])
		return;

	PRINode child;
	child.level = node.level - 1;
	child.pos = point - center;
	child.dist = s_dists[threadIdx.x];
	nodes[s_ichilds[inode] + icd] = child;*/
	__shared__ bool s_isSubdivs[16];
	__shared__ uint s_isInside[16];
	__shared__ uint s_ichilds[16];
	__shared__ REAL s_dists[128];
	__shared__ REAL s_dists2[128];
	uint id = (blockDim.x * blockIdx.x + threadIdx.x) >> 3;

	PRINode node;
	REAL3 point;
	REAL dist, nodeSide2;
	uint inode, icd;
	bool flag = id < nodeNum;

	if (flag) {
		inode = threadIdx.x >> 3;
		icd = threadIdx.x & 7;
		if (!icd) {
			s_isSubdivs[inode] = false;
			s_isInside[inode] = 0;
		}
	}
	__syncthreads();
	if (flag) {
		node = nodes[nodeStart + id];
		point = node.pos + center;
		point.x += (threadIdx.x & 1) * nodeSide;
		point.y += ((threadIdx.x >> 1) & 1) * nodeSide;
		point.z += ((threadIdx.x >> 2) & 1) * nodeSide;

		if (icd)	dist = ssqrt(getDistSqrKD(point, kdTree, objParams));
		else		dist = node.dist;

		if (fabs(dist) <= max(nodeCross, 0.05))
			s_isSubdivs[inode] = true;
		if (dist <= 0.0)
			atomicAdd(s_isInside + inode, 1);
		s_dists[threadIdx.x] = dist;
	}
	__syncthreads();

	if (flag)
		if (!s_isSubdivs[inode] && (s_isInside[inode] == 0 || s_isInside[inode] == 8))
			flag = false;

	if (flag) {
		nodeSide2 = nodeSide + nodeSide;
		REAL3 point2 = node.pos + center;
		point2.x += (threadIdx.x & 1) * nodeSide2;
		point2.y += ((threadIdx.x >> 1) & 1) * nodeSide2;
		point2.z += ((threadIdx.x >> 2) & 1) * nodeSide2;
		if (icd)
			s_dists2[threadIdx.x] = ssqrt(getDistSqrKD(point2, kdTree, objParams));
	}
	__syncthreads();

	if (flag) {
		if (!icd) {
			REAL3 corners[7];
			REAL dists[15];
			for (uint i = 1; i < 8; i++) {
				dists[i + 6] = s_dists[threadIdx.x + i];
				dists[i - 1] = s_dists2[threadIdx.x + i];
				corners[i - 1] = node.pos + center;
				corners[i - 1].x += (i & 1) * nodeSide2;
				corners[i - 1].y += ((i >> 1) & 1) * nodeSide2;
				corners[i - 1].z += ((i >> 2) & 1) * nodeSide2;
			}
			dists[14] = node.dist;
			if (ComparisonPRI(kdTree, objParams, dists, corners, error))
				s_isSubdivs[inode] = false;
			else {
				node.child = atomicAdd(ichild, 8);
				node.level = -node.level;
				nodes[nodeStart + id] = node;
				s_ichilds[inode] = node.child;
				node.level = -node.level;
				s_isSubdivs[inode] = true;
			}
		}
	}
	__syncthreads();
	if (flag) {
		if (s_isSubdivs[inode]) {
			PRINode child;
			child.level = node.level - 1;
			child.pos = point - center;
			child.dist = s_dists[threadIdx.x];
			nodes[s_ichilds[inode] + icd] = child;
		}
	}
}

void PrimalTree::initConstant(void) {
	unsigned char NUM_P[8][8] = {
		{ 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 1, 0, 1, 0, 1, 0, 1 },
		{ 0, 0, 2, 2, 0, 0, 2, 2 },
		{ 0, 1, 2, 3, 0, 1, 2, 3 },
		{ 0, 0, 0, 0, 4, 4, 4, 4 },
		{ 0, 1, 0, 1, 4, 5, 4, 5 },
		{ 0, 0, 2, 2, 4, 4, 6, 6 },
		{ 0, 1, 2, 3, 4, 5, 6, 7 } };
	unsigned char NUM_C[8][8] = {
		{ 0, 1, 2, 3, 4, 5, 6, 7 },
		{ 1, 0, 3, 2, 5, 4, 7, 6 },
		{ 2, 3, 0, 1, 6, 7, 4, 5 },
		{ 3, 2, 1, 0, 7, 6, 5, 4 },
		{ 4, 5, 6, 7, 0, 1, 2, 3 },
		{ 5, 4, 7, 6, 1, 0, 3, 2 },
		{ 6, 7, 4, 5, 2, 3, 0, 1 },
		{ 7, 6, 5, 4, 3, 2, 1, 0 } };
	CUDA_CHECK(cudaMemcpyToSymbol(D_NUM_P, NUM_P, 64 * sizeof(uchar)));
	CUDA_CHECK(cudaMemcpyToSymbol(D_NUM_C, NUM_C, 64 * sizeof(uchar)));

	/*uchar* ptr;
	CUDA_CHECK(cudaGetSymbolAddress((void**)&ptr, D_NUM_P));
	CUDA_CHECK(cudaMemcpy(ptr, NUM_P, 64 * sizeof(uchar), cudaMemcpyHostToDevice));
	printf("%d %d\n", ptr, D_NUM_P);*/
	/*for (int i = 0; i < 8; i++) {
		CUDA_CHECK(cudaMemcpyToSymbol(D_NUM_P[i], NUM_P[i], 8 * sizeof(uchar)));
		CUDA_CHECK(cudaMemcpyToSymbol(D_NUM_C[i], NUM_C[i], 8 * sizeof(uchar)));
	}*/
}
void PrimalTree::buildKDTree(
	KDTree& tree, const ObjParam& h_objParams, const PrefixArray<uint>& nbFs, 
	const vector<REAL>& fNorms, const vector<REAL>& nNorms)
{
	uint maxLevel = min(Log2(h_objParams._numFaces - 1u) << 1u, 9u);
	printf("%d\n", maxLevel);
	uint nodeNum = (1u << maxLevel + 1u) - 1u;
	vector<KDNode> nodes(nodeNum);
	vector<AABB> aabbs(nodeNum);
	vector<vector<uint>> faces(nodeNum);
	vector<uint> h_faces;
	vector<uint> h_inds(1, 0);
	nodes[0].level = 0;
	resetAABB(aabbs[0]);
	for (uint i = 0; i < h_objParams._numFaces; i++) {
		uint ino0 = h_objParams._fs[i * 3 + 0];
		uint ino1 = h_objParams._fs[i * 3 + 1];
		uint ino2 = h_objParams._fs[i * 3 + 2];
		REAL3 v0 = make_REAL3(h_objParams._ns[ino0 * 3 + 0], h_objParams._ns[ino0 * 3 + 1], h_objParams._ns[ino0 * 3 + 2]);
		REAL3 v1 = make_REAL3(h_objParams._ns[ino1 * 3 + 0], h_objParams._ns[ino1 * 3 + 1], h_objParams._ns[ino1 * 3 + 2]);
		REAL3 v2 = make_REAL3(h_objParams._ns[ino2 * 3 + 0], h_objParams._ns[ino2 * 3 + 1], h_objParams._ns[ino2 * 3 + 2]);
		addAABB(aabbs[0], v0, 0.0);
		addAABB(aabbs[0], v1, 0.0);
		addAABB(aabbs[0], v2, 0.0);
		faces[0].push_back(i);
	}
	tree.aabb = aabbs[0];

	vector<uint> queue(1, 0u);
	while (queue.size()) {
		/*uint curr = queue.back();
		queue.pop_back();*/
		uint curr = queue[0];
		queue.erase(queue.begin());
		//if (nodes[curr].level >= maxLevel || faces[curr].size() * 1000 <= h_objParams._numFaces)
		if (nodes[curr].level >= maxLevel)
		{
			nodes[curr].level = -nodes[curr].level;
			nodes[curr].index = h_inds.size() - 1u;
			h_faces.insert(h_faces.end(), faces[curr].begin(), faces[curr].end());
			h_inds.push_back(h_faces.size());
			continue;
		}

		REAL maxDist = 0.0;
		for (int i = 0; i < 3; i++) {
			REAL dist = *((REAL*)&aabbs[curr]._max + i) - *((REAL*)&aabbs[curr]._min + i);
			if (maxDist < dist) {
				maxDist = dist;
				nodes[curr].divAxis = i;
			}
		}
		vector<REAL> minVertices;
		vector<REAL> maxVertices;
		vector<REAL> elems;
		uint dAxis = nodes[curr].divAxis;
		for (auto i : faces[curr]) {
			REAL minV, maxV;
			minV = maxV = h_objParams._ns[h_objParams._fs[i * 3 + 0] * 3 + dAxis];
			for (uint j = 1u; j < 3u; j++) {
				REAL v0 = h_objParams._ns[h_objParams._fs[i * 3 + j] * 3 + dAxis];
				if (minV > v0)		minV = v0;
				else if (maxV < v0)	maxV = v0;
			}
			minVertices.push_back(minV);
			maxVertices.push_back(maxV);
			elems.push_back(minV);
			elems.push_back(maxV);
		}
		sort(elems.begin(), elems.end());
		nodes[curr].divPos = (elems[(elems.size() - 1u) >> 1] + elems[((elems.size() - 1u) >> 1) + 1u]) * 0.5;

		uint ichild = (curr << 1) + 1;
		KDNode& lnode = nodes[ichild];
		KDNode& rnode = nodes[ichild + 1];
		lnode.level = rnode.level = nodes[curr].level + 1;

		AABB& laabb = aabbs[ichild];
		AABB& raabb = aabbs[ichild + 1];
		laabb = raabb = aabbs[curr];
		*((REAL*)&laabb._max + dAxis) = *((REAL*)&raabb._min + dAxis) = nodes[curr].divPos;

		vector<uint>& lfaces = faces[ichild];
		vector<uint>& rfaces = faces[ichild + 1];

		for (uint i = 0; i < faces[curr].size(); i++) {
			if (maxVertices[i] < nodes[curr].divPos)
				lfaces.push_back(faces[curr][i]);
			else if (minVertices[i] > nodes[curr].divPos)
				rfaces.push_back(faces[curr][i]);
			else {
				lfaces.push_back(faces[curr][i]);
				rfaces.push_back(faces[curr][i]);
			}
		}
		faces[curr].clear();
		queue.push_back(ichild);
		queue.push_back(ichild + 1);
	}
	tree.maxLevel = maxLevel;
	tree.fnum = h_faces.size();
	tree.leafNum = h_inds.size() - 1u;
	initKDTreeDevice(tree);
	CUDA_CHECK(cudaMemcpy(tree.nodes, &nodes[0], nodeNum * sizeof(KDNode), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(tree.aabbs, &aabbs[0], nodeNum * sizeof(AABB), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(tree.inds, &h_inds[0], h_inds.size() * sizeof(uint), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(tree.faces, &h_faces[0], tree.fnum * sizeof(uint), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMalloc((void**)&tree.mesh.vNorms, h_objParams._numNodes * sizeof(REAL3)));
	CUDA_CHECK(cudaMalloc((void**)&tree.mesh.fNorms, h_objParams._numFaces * sizeof(FaceNormal)));

	vector<FaceNormal> h_fNorms;
	FaceNormal fnorm;
	REAL3 tmp;
	for (uint i = 0u; i < h_objParams._numFaces; i++) {
		fnorm.fNorm = make_REAL3(fNorms[i * 3u + 0u], fNorms[i * 3u + 1u], fNorms[i * 3u + 2u]);
		for (uint j = 0; j < 3; j++) {
			uint ino0 = h_objParams._fs[i * 3u + j];
			uint ino1 = h_objParams._fs[i * 3u + ((j + 1u) % 3u)];
			for (uint k = nbFs._index[ino0]; k < nbFs._index[ino0 + 1u]; k++) {
				uint iface = nbFs._array[k];
				if (iface != i) {
					uint l;
					for (l = 0u; l < 3u; l++)
						if (ino1 == h_objParams._fs[iface * 3u + l])
							break;
					if (l < 3u) {
						fnorm.eNorm[j] = fnorm.fNorm +
							make_REAL3(fNorms[iface * 3u + 0u], fNorms[iface * 3u + 1u], fNorms[iface * 3u + 2u]);
						Normalize(fnorm.eNorm[j]);
						break;
					}
				}
			}
		}
		h_fNorms.push_back(fnorm);
	}

	CUDA_CHECK(cudaMemcpy(tree.mesh.vNorms, &nNorms[0], h_objParams._numNodes * sizeof(REAL3), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(tree.mesh.fNorms, &h_fNorms[0], h_objParams._numFaces * sizeof(FaceNormal), cudaMemcpyHostToDevice));
}
void PrimalTree::buildTree(const ObjParam& h_objParams, const ObjParam& d_objParams,
	const PrefixArray<uint>& nbFs, const vector<REAL>& fNorms, const vector<REAL>& nNorms,
	REAL delta, uint maxLevel)
{
	CUDA_CHECK(cudaDeviceSynchronize());
	ctimer timer = CNOW;
	printf("KDTree build: ");

	KDTree kdTree;
	buildKDTree(kdTree, h_objParams, nbFs, fNorms, nNorms);

	CUDA_CHECK(cudaDeviceSynchronize());
	printf("%lf msec\n", (CNOW - timer) / 10000.0);
	timer = CNOW;
	printf("PrimalTree build: ");

	d_tree._half =
		max(max(kdTree.aabb._max.x - kdTree.aabb._min.x, kdTree.aabb._max.y - kdTree.aabb._min.y),
		kdTree.aabb._max.z - kdTree.aabb._min.z) * 0.5 + delta;

	d_tree._center = (kdTree.aabb._min + kdTree.aabb._max) * 0.5;

	PRINode* d_nodesTmp;
	uint* d_ichild;
	uint h_ichild = 9;
	CUDA_CHECK(cudaMalloc((void**)&d_nodesTmp, 1e7 * sizeof(PRINode)));
	CUDA_CHECK(cudaMalloc((void**)&d_ichild, sizeof(uint)));

	REAL nodeSide = d_tree._half + d_tree._half;
	REAL nodeCross = sqrt(3.0) * nodeSide;
	//REAL nodeCross = sqrt(3.0) * nodeSide * 4.0;
	uint istart = 1;

	subdivPRINodeInit << <1, 9 >> >
		(kdTree, d_objParams, d_nodesTmp, d_tree._center, nodeSide, d_ichild);
	for (uint i = 1; i < maxLevel; i++) {
		nodeSide *= 0.5; nodeCross *= 0.5;
		if (h_ichild == istart)
			break;
		uint nodeNum = h_ichild - istart;
		subdivPRINode << <divup(nodeNum * 8, 128), 128 >> >
			(kdTree, d_objParams, d_nodesTmp, istart, nodeNum, d_tree._center, nodeSide, nodeCross, _error, d_ichild);
		istart = h_ichild;
		printf("%d, %d\n", istart, h_ichild);
		CUDA_CHECK(cudaMemcpy(&h_ichild, d_ichild, sizeof(uint), cudaMemcpyDeviceToHost));
	}
	CUDA_CHECK(cudaFree(d_ichild));

	CUDA_CHECK(cudaMalloc((void**)&d_tree, h_ichild * sizeof(PRINode)));
	CUDA_CHECK(cudaMemcpy(d_tree._nodes, d_nodesTmp, h_ichild * sizeof(PRINode), cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaFree(d_nodesTmp));

	destroyKDTreeDevice(kdTree);
	CUDA_CHECK(cudaDeviceSynchronize());

	_nodeNum = h_ichild;
	_maxLevel = maxLevel;
	printf("%lf msec, node: %d\n", (CNOW - timer) / 10000.0, _nodeNum);
}

void PrimalTree::draw(const AABB& aabb) {
	glDisable(GL_LIGHTING);
	glPushMatrix();
	glLineWidth(1.0f);
	glColor3d(0.6, 0.6, 0.6);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glBegin(GL_LINES);
	glVertex3d(aabb._min.x, aabb._min.y, aabb._min.z);
	glVertex3d(aabb._min.x, aabb._min.y, aabb._max.z);
	glVertex3d(aabb._min.x, aabb._max.y, aabb._min.z);
	glVertex3d(aabb._min.x, aabb._max.y, aabb._max.z);
	glVertex3d(aabb._max.x, aabb._min.y, aabb._min.z);
	glVertex3d(aabb._max.x, aabb._min.y, aabb._max.z);
	glVertex3d(aabb._max.x, aabb._max.y, aabb._min.z);
	glVertex3d(aabb._max.x, aabb._max.y, aabb._max.z);
	glEnd();
	glTranslated(0, 0, aabb._min.z);
	glRectd(aabb._min.x, aabb._min.y, aabb._max.x, aabb._max.y);
	glTranslated(0, 0, aabb._max.z - aabb._min.z);
	glRectd(aabb._min.x, aabb._min.y, aabb._max.x, aabb._max.y);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glPopMatrix();
	glEnable(GL_LIGHTING);

}
void PrimalTree::draw(void) {
	PRINode* h_tree = (PRINode*)malloc(_nodeNum * sizeof(PRINode));
	CUDA_CHECK(cudaMemcpy(h_tree, d_tree._nodes, _nodeNum * sizeof(PRINode), cudaMemcpyDeviceToHost));
	AABB aabb;
	aabb._min = d_tree._center - d_tree._half;
	aabb._max = d_tree._center + d_tree._half;
	draw(aabb);
	REAL half2 = d_tree._half + d_tree._half;
	for (uint i = 0; i < _nodeNum; i++) {
		if (fabs(h_tree[i].level) < _maxLevel - 3) continue;
		REAL half = half2 * (1.0 / (REAL)(1 << abs(h_tree[i].level)));
		aabb._min = (h_tree[i].pos - half) * d_tree._scale + d_tree._center;
		aabb._max = (h_tree[i].pos + half) * d_tree._scale + d_tree._center;
		draw(aabb);
	}
	free(h_tree);
}