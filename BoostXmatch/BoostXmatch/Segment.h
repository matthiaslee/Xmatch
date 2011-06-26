/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#pragma once
#ifndef SEGMENT_H
#define SEGMENT_H

#include <cstdint>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "Obj.h"

namespace xmatch
{
	class Segment
	{
	public:
		uint32_t mId;
		uint64_t mNum;
		bool mSorted;
		Obj *mObj;

		Segment(uint32_t mId, uint64_t mNum);
		~Segment();

		std::string ToString(std::string sep) const;	
		std::string ToString() const;

		friend std::ostream& operator<< (std::ostream &rOs, const Segment &rSegment);
	};

	typedef boost::shared_ptr<Segment> SegmentPtr;
	typedef std::vector<SegmentPtr> SegmentVec;
}

#endif /* SEGMENT_H */