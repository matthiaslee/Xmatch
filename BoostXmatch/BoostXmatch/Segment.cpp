/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#include "Segment.h"


namespace xmatch
{
	Segment::Segment(uint32_t id, uintmax_t num) : mId(id), mNum(num), mZoneHeightDegree(0)
	{
	}

	void Segment::Load(std::istream &rIs)
	{
		if (mNum > 0)
		{
			vObj.resize(mNum);
			Obj *o = thrust::raw_pointer_cast(&vObj[0]);
			rIs.read( (char*)o, mNum * sizeof(Obj));
		}
	}
	
	std::ostream& operator<< (std::ostream &o, const Segment &s)
	{
		o << "Segment-" << s.mId;
		return o;
	}
}

