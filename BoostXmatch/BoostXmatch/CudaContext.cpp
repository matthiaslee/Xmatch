#include "CudaContext.h"

namespace xmatch
{
	CudaContext::CudaContext(int id) : id(id)
	{	
		CUresult cuResult = cuCtxCreate(&ctx,CU_CTX_SCHED_YIELD,id);
		if( cuResult != CUDA_SUCCESS )
		{
			id = -1;
		}
	}

	CudaContext::~CudaContext(void)
	{
		CUresult cuResult = cuCtxDestroy(ctx);
	}
}

