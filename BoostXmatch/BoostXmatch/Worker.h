/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#pragma once
#ifndef WORKER_H
#define WORKER_H
#include "JobManager.h"

#include <cstdint>
#include <iostream>

#include <boost/filesystem.hpp>

namespace xmatch
{	
	class Worker
	{    		
		uint32_t id;
		JobPtr oldjob;
		JobManagerPtr jobman;
		boost::filesystem::path outpath;

	public:
		Worker(uint32_t id, JobManagerPtr jobman, boost::filesystem::path prefix);
		void operator()();		
	};
}
#endif /* WORKER_H */