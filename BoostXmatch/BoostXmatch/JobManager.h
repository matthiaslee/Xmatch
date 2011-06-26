/*
 *   ID:          $Id: $
 *   Revision:    $Rev: $
 */
#pragma once
#ifndef JOBMANAGER_H
#define JOBMANAGER_H

#include <cstdint>
#include <iostream>

#include <boost/thread/mutex.hpp>
#include "Segment.h"
#include "Job.h"

namespace xmatch
{	
	class JobManager
	{
		boost::mutex mtx;
		JobVec jobs;

	public:
		JobManager(const SegmentVec& segA, const SegmentVec& segB, bool swap);

		//	Prefers jobs with segments that the worker already holds
		JobPtr NextPreferredJob(JobPtr oldjob);

		// TODO: Is this a potential race-condition problem with Next...() 
		void SetStatus(JobPtr job, JobStatus status);
	};

	typedef boost::shared_ptr<JobManager> JobManagerPtr;
}
#endif /* JOBMANAGER_H */