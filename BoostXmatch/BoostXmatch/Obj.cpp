/*
 *   ID:          $Id: Obj.cu 6985 2011-06-29 16:33:33Z budavari $
 *   Revision:    $Rev: 6985 $
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