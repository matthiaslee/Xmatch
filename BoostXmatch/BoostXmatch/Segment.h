/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#pragma once
#ifndef SEGMENT_H
#define SEGMENT_H
#include "Obj.h"

#include <cstdint>
#include <string>
#include <vector>
#include <iostream>

#include <boost/shared_ptr.hpp>
#include <thrust/host_vector.h>


namespace xmatch
{
	class Segment
	{
		// main() reads here but clear()-ed later
		thrust::host_vector<Obj> hObj; 
		// pre-sorted data for processing
		thrust::host_vector<int64_t> hId;
		thrust::host_vector<double2> hRaDec;
		thrust::host_vector<int> hZoneBegin, hZoneEnd;


	public:
		uint32_t mId;
		uint64_t mNum;

		Segment(uint32_t id, uint64_t num) : mId(id), mNum(num) { std::cout << hId.size() << " <= " << hId.capacity() << std::endl; };

		void Load(std::istream &rIs);
		void Sort(double degZoneHeight);
		void Work(const Segment &rSegment) const;

		std::string ToString(std::string sep) const;	
		std::string ToString(void) const;

		friend std::ostream& operator<< (std::ostream &rOs, const Segment &rSegment);
	};

	typedef boost::shared_ptr<Segment> SegmentPtr;
	typedef std::vector<SegmentPtr> SegmentVec;
}

#endif /* SEGMENT_H */