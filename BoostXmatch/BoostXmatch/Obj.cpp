/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#include "Obj.h"

namespace xmatch
{
	std::ostream& operator<< (std::ostream& out, const Obj& rObj) 
	{
		out << rObj.mId << " " << rObj.mRa << " " << rObj.mDec;
		return out;
	}
}