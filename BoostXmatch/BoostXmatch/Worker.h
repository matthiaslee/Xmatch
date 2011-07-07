/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#pragma once
#ifndef WORKER_H
#define WORKER_H
#include "JobManager.h"
#include "CudaManager.h"

namespace xmatch
{	
	class Worker
	{    		
		uint32_t id;
		JobPtr oldjob;
		JobManagerPtr jobman;
		CudaManagerPtr cuman;
		std::string outpath;
		int verbosity;

		void Match(JobPtr job, std::ofstream& ofs);

	public:
		Worker(CudaManagerPtr cuman, uint32_t id, JobManagerPtr jobman, std::string outpath, int verbosity) 
			: cuman(cuman), id(id), jobman(jobman), outpath(outpath), verbosity(verbosity), oldjob((Job*)NULL) 
		{ }

		void operator()();		
	};
}
#endif /* WORKER_H */