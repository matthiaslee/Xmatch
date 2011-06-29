/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#pragma once
#ifndef OBJ_H
#define OBJ_H

#include <cstdint>
#include <iostream>

namespace xmatch
{
	class Obj
	{
	public:
		int64_t mId;
		double mRa, mDec;

		Obj();
		Obj(int64_t id, double ra, double dec);

		static int32_t GetZoneId(double dec_deg, double height);

		friend std::ostream& operator<< (std::ostream &rOs, const Obj &rObj);
	};
}
#endif /* OBJ_H */