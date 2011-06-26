#include "Job.h"
#include <sstream>

namespace xmatch
{
	std::string Job::ToString() const
	{
		std::stringstream ss;
		ss << "Job " << segA->mId << "x" << segB->mId; // << ":" << status; 
		return ss.str();
	}

	std::ostream& operator<<(std::ostream &rOs, const Job &rJob)
	{
		rOs << rJob.ToString();
		return rOs;
	}
}