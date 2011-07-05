/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#pragma once
#ifndef WORKER_H
#define WORKER_H
#include "JobManager.h"

namespace xmatch
{	
	class Worker
	{    		
		uint32_t id;
		JobPtr oldjob;
		JobManagerPtr jobman;
		std::string outpath;

	public:
		Worker(uint32_t id, JobManagerPtr jobman, std::string outpath) : id(id), jobman(jobman), outpath(outpath), oldjob((Job*)NULL) {	}
		void operator()();		
	};
}
#endif /* WORKER_H */