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
	struct dbl2 { double x, y; };

	struct Obj
	{
		int64_t mId;
		double mRa, mDec;

#ifdef NOT_HERE
		__device__ __host__
		Obj() : mId(0), mRa(0), mDec(0) { }

		__device__ __host__
		Obj(int64_t id, double ra, double dec) : mId(id), mRa(ra), mDec(dec) { } 

		__device__ __host__
		Obj(const Obj& o) : mId(o.mId), mRa(o.mRa), mDec(o.mDec) { }
#endif
	};

	std::ostream& operator<< (std::ostream &rOs, const Obj &rObj);
}
#endif /* OBJ_H */