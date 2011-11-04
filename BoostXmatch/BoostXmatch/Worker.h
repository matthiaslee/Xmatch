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
		CudaManagerPtr cuman;
		JobManagerPtr jobman;
		JobPtr oldjob;
		uint32_t maxout;
		std::string outpath;
		int verbosity;

		void Match(JobPtr job, std::ofstream& ofs);

	public:
		Worker(CudaManagerPtr cuman, uint32_t id, JobManagerPtr jobman, std::string outpath, uint32_t maxout, int verbosity) 
			: cuman(cuman), id(id), jobman(jobman), outpath(outpath), maxout(maxout), verbosity(verbosity), oldjob((Job*)NULL) 
		{ }

		void operator()();		
	};
}
#endif /* WORKER_H */

