/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#include "Obj.h"

namespace xmatch
{
	__host__ __device__	
	Obj::Obj() : mId(-1), mRa(99), mDec(99) {}

	__host__ __device__
	Obj::Obj(int64_t id, double ra, double dec) : mId(id), mRa(ra), mDec(dec) {}


	__host__ __device__
	int Obj::GetZoneId(double dec_deg, double height)
	{
		return (int) rint( (dec_deg + 90) / height );
	}

	std::ostream& operator<< (std::ostream& out, const Obj& rObj) 
	{
		out << rObj.mId << " " << rObj.mRa << " " << rObj.mDec;
		return out;
	}
}