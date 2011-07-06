#include "Job.h"

namespace xmatch
{
	uint32_t Job::ShareSegment(const Job &rJob)
	{
		int n = 0;
		if (segA->mId == rJob.segA->mId) n++;
		if (segB->mId == rJob.segB->mId) n++;
		return n;
	}

	std::ostream& operator<<(std::ostream &rOs, const Job &rJob)
	{
		rOs << "Job-" << rJob.segA->mId << "x" << rJob.segB->mId; 
		return rOs;
	}
}