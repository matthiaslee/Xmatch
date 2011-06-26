/*
 *   ID:          $Id: $
 *   Revision:    $Rev: $
 */
#pragma once
#ifndef SEGMENTMANAGER_H
#define SEGMENTMANAGER_H

#include <cstdint>
#include <iostream>

#include <boost/thread/mutex.hpp>

#include "Segment.h"

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