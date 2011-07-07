/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#pragma once
#ifndef OBJ_H
#define OBJ_H

#include <iostream>

#pragma warning(push)
#pragma warning(disable: 4005)      // BOOST_COMPILER macro redefinition
#include <boost/cstdint.hpp>
#pragma warning(pop)


#define RAD2DEG 57.295779513082323

namespace xmatch
{
	struct dbl2 { double x, y; };

	struct Obj
	{
		int64_t mId;
		double mRa, mDec;
	};

	std::ostream& operator<< (std::ostream &rOs, const Obj &rObj);
}
#endif /* OBJ_H */