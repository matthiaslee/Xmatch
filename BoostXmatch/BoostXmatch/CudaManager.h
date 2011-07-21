/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#pragma once
#ifndef CUDAMANAGER_H
#define CUDAMANAGER_H
#include <vector>

#pragma warning(push)
#pragma warning(disable: 4005)      // BOOST_COMPILER macro redefinition
#include <boost/shared_array.hpp>
#include <boost/thread.hpp>
#pragma warning(pop)


namespace xmatch
{
	class CudaManager
	{
		boost::mutex mtx;
		boost::shared_array<bool> available;
		int nDevices;

	public:
		CudaManager();
		int NextDevice();
		void Release(int id);

		//std::vector<int> CudaManager::Query(const cudaDeviceProp& req);
		//static void Print(cudaDeviceProp devProp);

		inline int GetDeviceCount() { return nDevices; }
	};

	typedef boost::shared_ptr<CudaManager> CudaManagerPtr;
}

#endif /* CUDAMANAGER_H */
