/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#pragma once
#ifndef JOB_H
#define JOB_H

#include <cstdint>
#include <iostream>

#include "Segment.h"

namespace xmatch
{
	enum JobStatus { PENDING, RUNNING, FINISHED };

	class Job
	{
	public:
		JobStatus status;
		SegmentPtr segA, segB;
		bool swap;

		Job(SegmentPtr a, SegmentPtr b, bool swap) : segA(a), segB(b), swap(swap), status(PENDING) { }

		uint32_t ShareSegment(const Job &rJob);

		std::string ToString() const;

		friend std::ostream& operator<< (std::ostream &rOs, const Job &rJob);
	};

	typedef boost::shared_ptr<Job> JobPtr;
	typedef std::vector<JobPtr> JobVec;
}


#endif /* JOB_H */