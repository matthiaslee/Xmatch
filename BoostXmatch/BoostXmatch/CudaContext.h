/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#pragma once
#ifndef CUDACONTEXT_H
#define CUDACONTEXT_H
#include "CudaManager.h"

#include <cuda.h>

#pragma warning(push)
#pragma warning(disable: 4005)      // BOOST_COMPILER macro redefinition
#include <boost/shared_ptr.hpp>
#pragma warning(pop)


namespace xmatch
{	
	class CudaContext
	{
		CudaManagerPtr cuman;
		CUcontext ctx;
		int id;

	public:
		CudaContext(CudaManagerPtr cuman);
		~CudaContext();

		int GetDeviceID() { return id; }
	};

	typedef boost::shared_ptr<CudaContext> CudaContextPtr;
}
#endif /* CUDACONTEXT_H */

