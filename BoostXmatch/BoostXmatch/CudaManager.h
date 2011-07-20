/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#pragma once
#ifndef CUDAMANAGER_H
#define CUDAMANAGER_H
#include "CudaContext.h"

#include <vector>

#pragma warning(push)
#pragma warning(disable: 4005)      // BOOST_COMPILER macro redefinition
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#pragma warning(pop)


namespace xmatch
{
	class CudaManager
	{
		boost::mutex mtx;
		std::vector<int> devId;

	public:
		CudaManager();
		CudaContextPtr GetContext(void);
		CudaContextPtr GetContext(int id);

		//std::vector<int> CudaManager::Query(const cudaDeviceProp& req);
		//static void Print(cudaDeviceProp devProp);

		inline int GetDeviceCount() { return devId.size(); }
	};

	typedef boost::shared_ptr<CudaManager> CudaManagerPtr;
}

#endif /* CUDAMANAGER_H */
