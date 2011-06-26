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
#include <iostream>

#include <boost/shared_ptr.hpp>
#include <thrust/host_vector.h>

#include "Obj.h"

namespace xmatch
{
	class Segment
	{
		thrust::host_vector<Obj> dObj;

	public:
		uint32_t mId;
		uint64_t mNum;
		bool mSorted;
		Obj *mObj;

		Segment(uint32_t id, uint64_t num);
		~Segment();

		void Load(std::istream &rIs);

		std::string ToString(std::string sep) const;	
		std::string ToString() const;

		friend std::ostream& operator<< (std::ostream &rOs, const Segment &rSegment);
	};

	typedef boost::shared_ptr<Segment> SegmentPtr;
	typedef std::vector<SegmentPtr> SegmentVec;
}

#endif /* SEGMENT_H */