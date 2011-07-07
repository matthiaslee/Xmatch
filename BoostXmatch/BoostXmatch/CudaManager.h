/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#pragma once
#ifndef CUDAMANAGER_H
#define CUDAMANAGER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <vector>

#pragma warning(push)
#pragma warning(disable: 4005)      // BOOST_COMPILER macro redefinition
#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>
#pragma warning(pop)


namespace xmatch
{
	class CudaManager
	{
		boost::mutex mtx;

	public:
		CudaManager() { };

		int GetDeviceCount();
		cudaError_t SetDevice(int id);
		cudaError_t Reset();

		std::vector<int> CudaManager::Query(const cudaDeviceProp& req);

		static void Print(cudaDeviceProp devProp);
	};

	typedef boost::shared_ptr<CudaManager> CudaManagerPtr;
}

#endif /* CUDAMANAGER_H */