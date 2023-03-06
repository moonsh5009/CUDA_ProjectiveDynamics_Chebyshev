#include "RTriangle.h"
#include "../include/CUDA_Custom/DeviceManager.cuh"

__global__ void RTriBuild_Kernel(
	uint* fs, uint* nbFs, uint* inbFs, RTriParam rtri)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= rtri._size)
		return;

	uint inos[3];
	uint ino = id * 3u, jno;
	inos[0] = fs[ino + 0u];
	inos[1] = fs[ino + 1u];
	inos[2] = fs[ino + 2u];

	uint iend, jstart, jend;
	uint i, j, itri, jtri;
	uint info = 0u;
	bool flag;

	jstart = inbFs[inos[0]];
	jend = inbFs[inos[0] + 1u];
	for (i = 0u; i < 3u; i++) {
		ino = jstart;
		iend = jend;
		itri = nbFs[ino++];
		if (itri == id)
			setRTriVertex(info, i);

		j = (i + 1u) % 3u;
		jstart = inbFs[inos[j]];
		jend = inbFs[inos[j] + 1u];

		/*flag = false;
		for (jno = jstart; jno < jend; jno++) {
			jtri = nbFs[jno];
			if (jtri >= itri) {
				flag = itri == jtri;
				break;
			}
		}
		for (; ino < iend && !flag; ino++) {
			itri = nbFs[ino];
			for (; jno < jend; jno++) {
				jtri = nbFs[jno];
				if (jtri >= itri) {
					flag = itri == jtri;
					break;
				}
			}
		}
		if (flag) setRTriEdge(info, i);*/
		{
			uint of = 0xffffffff;
			for (jno = jstart; jno < jend; jno++) {
				jtri = nbFs[jno];
				if (jtri == itri && id != itri) {
					of = itri;
					break;
				}
			}
			for (; ino < iend && of == 0xffffffff; ino++) {
				itri = nbFs[ino];
				for (jno = jstart; jno < jend; jno++) {
					jtri = nbFs[jno];
					if (jtri == itri && id != itri) {
						of = itri;
						break;
					}
				}
			}
			if (id < of)
				setRTriEdge(info, i);
		}
	}
	//printf("%d, %d, %d, %d, %d, %d, %d\n", id, (info >> 5u) & 1u, (info >> 4u) & 1u, (info >> 3u) & 1u, (info >> 2u) & 1u, (info >> 1u) & 1u, (info >> 0u) & 1u);
	rtri._info[id] = info;
	//rtri._info[id] = 0xffffffff;
}