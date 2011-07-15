/*
 *   ID:          $Id: $
 *   Revision:    $Rev: $
 */
#pragma once
#ifndef CUDACONTEXT_H
#define CUDACONTEXT_H

#include <cuda.h>

#pragma warning(push)
#pragma warning(disable: 4005)      // BOOST_COMPILER macro redefinition
#include <boost/shared_ptr.hpp>
#pragma warning(pop)


namespace xmatch
{	
	class CudaContext
	{
		CUcontext ctx;
		int id;

	public:
		CudaContext(int id);
		~CudaContext(void);

		int GetDeviceID() { return id; }
	};

	typedef boost::shared_ptr<CudaContext> CudaContextPtr;
}
#endif /* CUDACONTEXT_H */
