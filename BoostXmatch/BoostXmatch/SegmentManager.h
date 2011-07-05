/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#pragma once
#ifndef SEGMENTMANAGER_H
#define SEGMENTMANAGER_H
#include "Segment.h"


#pragma warning(push)
#pragma warning(disable: 4005)      // BOOST_COMPILER macro redefinition
#include <boost/thread/mutex.hpp>
#pragma warning(pop)


namespace xmatch
{
	class SegmentManager
	{
		boost::mutex mtx;
		SegmentVec seg;
		uint32_t index;

	public:
		SegmentManager(SegmentVec &rSegments) : seg(rSegments), index(0) {}
		SegmentPtr Next();
	};

	typedef boost::shared_ptr<SegmentManager> SegmentManagerPtr;
}
#endif /* SEGMENTMANAGER_H */