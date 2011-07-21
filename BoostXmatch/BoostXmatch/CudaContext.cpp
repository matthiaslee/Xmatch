#include "CudaContext.h"

namespace xmatch
{
	CudaContext::CudaContext(CudaManagerPtr cuman) : cuman(cuman)
	{	
		id = cuman->NextDevice();
		if (id < 0) return;
		CUdevice cuDevice;
		CUresult cuResult;
		cuResult = cuInit(0);
		cuResult = cuDeviceGet(&cuDevice, id);
		cuResult = cuCtxCreate(&ctx, 0, cuDevice);
		// cuResult = cuCtxCreate(&ctx,CU_CTX_SCHED_YIELD,id);
		if( cuResult != CUDA_SUCCESS )
		{
			id = -2;
		}
	}

	CudaContext::~CudaContext()
	{
		cuman->Release(id);
		CUresult cuResult = cuCtxDestroy(ctx);
	}
}

