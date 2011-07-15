/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#pragma once
#ifndef JOBMANAGER_H
#define JOBMANAGER_H
#include "Job.h"

#pragma warning(push)
#pragma warning(disable: 4005)      // BOOST_COMPILER macro redefinition
#include <boost/thread/mutex.hpp>
#pragma warning(pop)


namespace xmatch
{	
	class JobManager
	{
		boost::mutex mtx;
		JobVec jobs;

	public:
		JobManager(const SegmentVec& segA, const SegmentVec& segB, double sr_deg);

		//	Prefers jobs with segments that the worker already holds
		JobPtr NextPreferredJob(JobPtr oldjob);

		// TODO: Is this a potential race-condition problem with Next...() 
		void SetStatus(JobPtr job, JobStatus status);
	};

	typedef boost::shared_ptr<JobManager> JobManagerPtr;
}
#endif /* JOBMANAGER_H */

