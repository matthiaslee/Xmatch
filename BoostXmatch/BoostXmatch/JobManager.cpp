#include "JobManager.h"

namespace xmatch
{
	JobManager::JobManager(const SegmentVec& segA, const SegmentVec& segB, bool swap) 
	{
		for (SegmentVec::size_type iA=0; iA<segA.size(); iA++)
		for (SegmentVec::size_type iB=0; iB<segB.size(); iB++)
		{
			JobPtr job(new Job(segA[iA],segB[iB],swap));
			jobs.push_back(job);
		}		
	}

	//	Prefers jobs with segments that the worker already holds
	JobPtr JobManager::NextPreferredJob(JobPtr oldjob)
	{
		boost::mutex::scoped_lock lock(mtx);
		JobPtr nextjob((Job*)NULL);
		for (JobVec::size_type i=0; i<jobs.size(); i++)
		{
			JobPtr job = jobs[i];
			if (job->status == PENDING)
			{
				if (nextjob == NULL) 
				{
					nextjob = job;
					if (oldjob == NULL) 
						break;
				}
				// override if find preloaded segments
				if (job->segA->mId == oldjob->segA->mId || 
					job->segB->mId == oldjob->segB->mId ) 
				{
					nextjob = job;
					break;
				}
			}
		}		
		if (nextjob != NULL) nextjob->status = RUNNING;
		return nextjob;
	}

	// TODO: Is this a potential race-condition problem with Next...() 
	void JobManager::SetStatus(JobPtr job, JobStatus status)
	{
		boost::mutex::scoped_lock lock(mtx);
		job->status = status;
	}
}

