/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#pragma once
#ifndef JOB_H
#define JOB_H
#include "Segment.h"

#include <cstdint>
#include <iostream>


namespace xmatch
{
	enum JobStatus { PENDING, RUNNING, FINISHED };

	class Job
	{
	public:
		JobStatus status;
		SegmentPtr segA, segB;
		bool swap;
		double sr_deg;

		Job(SegmentPtr a, SegmentPtr b, bool swap, double sr_deg) : segA(a), segB(b), swap(swap), sr_deg(sr_deg), status(PENDING) { }
		uint32_t ShareSegment(const Job &rJob);
		friend std::ostream& operator<< (std::ostream &rOs, const Job &rJob);
	};

	typedef boost::shared_ptr<Job> JobPtr;
	typedef std::vector<JobPtr> JobVec;
}


#endif /* JOB_H */