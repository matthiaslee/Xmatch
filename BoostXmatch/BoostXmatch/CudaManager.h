/*
 *   ID:          $Id: SegmentManager.h 7013 2011-07-05 19:08:44Z budavari $
 *   Revision:    $Rev: 7013 $
 */
#pragma once
#ifndef CUDAMANAGER_H
#define CUDAMANAGER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

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

		static void Print(cudaDeviceProp devProp);
	};

	typedef boost::shared_ptr<CudaManager> CudaManagerPtr;
}

#endif /* CUDAMANAGER_H */