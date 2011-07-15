#include "SegmentManager.h"

namespace xmatch
{
	SegmentPtr SegmentManager::Next()
	{
		if (index < seg.size())
		{
			boost::mutex::scoped_lock lock(mtx);
			return seg[index++];
		}
		else
		{
			return SegmentPtr((Segment*)NULL);
		}
	}
}

