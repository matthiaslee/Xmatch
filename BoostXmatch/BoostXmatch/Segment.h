/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#pragma once
#ifndef SEGMENT_H
#define SEGMENT_H
#include "Obj.h"

#include <vector>

#pragma warning(push)
#pragma warning(disable: 4996)      // Thrust's use of strerror
#pragma warning(disable: 4005)      // BOOST_COMPILER macro redefinition
#include <boost/shared_ptr.hpp>
#include <thrust/host_vector.h>
#pragma warning(pop)


namespace xmatch
{
	class Segment
	{
	public:
		uint32_t mId;
		uintmax_t mNum;
		double mZoneHeightDegree;

		// main() reads here but clear()-ed later
		thrust::host_vector<Obj> vObj; 
		// pre-sorted data for processing
		thrust::host_vector<int64_t> vId;
		thrust::host_vector<dbl2> vRadec;
		thrust::host_vector<int> vZoneBegin, vZoneEnd;

		Segment(uint32_t id, uintmax_t num); 
		void Load(std::istream &rIs);

		friend std::ostream& operator<< (std::ostream &rOs, const Segment &rSegment);
	};

	typedef boost::shared_ptr<Segment> SegmentPtr;
	typedef std::vector<SegmentPtr> SegmentVec;
}

#endif /* SEGMENT_H */