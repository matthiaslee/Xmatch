#include "CudaManager.h"

namespace xmatch
{
	cudaError_t CudaManager::SetDevice(int id)
	{
		boost::mutex::scoped_lock lock(mtx);
		cudaError_t err = cudaSetDevice(id);
		return err;
	}

	cudaError_t CudaManager::Reset()
	{
		boost::mutex::scoped_lock lock(mtx);
		cudaError_t err = cudaDeviceReset();
		return err;
	}

	int CudaManager::GetDeviceCount()
	{
		int n;
		cudaError_t err = cudaGetDeviceCount(&n);
		if (err != cudaSuccess) n = -1;
		return n;
	}

	bool MeetsReq (const cudaDeviceProp& dev, const cudaDeviceProp& req)
	{
		if (dev.major < req.major) return false;
		if (dev.minor < req.minor) return false;
		if (dev.totalGlobalMem < req.totalGlobalMem) return false;
		return true;
	}

	std::vector<int> CudaManager::Query(const cudaDeviceProp& req)
	{
		std::vector<int> devId;
		int n;
		cudaError_t err = cudaGetDeviceCount(&n);
		if (err == cudaSuccess)
		{
			cudaDeviceProp prop;
			for (int i=0; i<n; i++)
			{
				cudaGetDeviceProperties(&prop,i);
				if (MeetsReq(prop, req))
				{
					devId.push_back(i);
				}
			}
		}
		return devId;
	}

	/*
	void CudaManager::GetDeviceProperties(int id, cudaDeviceProp *prop)
	{
		cudaDe
	}
	*/

	// Print device properties
	void Print(cudaDeviceProp devProp)
	{
		printf("Major revision number:         %d\n",  devProp.major);
		printf("Minor revision number:         %d\n",  devProp.minor);
		printf("Name:                          %s\n",  devProp.name);
		printf("Total global memory:           %u\n",  devProp.totalGlobalMem);
		printf("Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
		printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
		printf("Warp size:                     %d\n",  devProp.warpSize);
		printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
		printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
		for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
		for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
		printf("Clock rate:                    %d\n",  devProp.clockRate);
		printf("Total constant memory:         %u\n",  devProp.totalConstMem);
		printf("Texture alignment:             %u\n",  devProp.textureAlignment);
		printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
		printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
		printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
		return;
	}

}