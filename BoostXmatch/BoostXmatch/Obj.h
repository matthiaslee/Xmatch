/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#pragma once
#ifndef OBJ_H
#define OBJ_H

#include <cstdint>
#include <iostream>
#include <math.h>

namespace xmatch
{
	class Obj
	{
	public:
		int64_t mId;
		double mRa, mDec;

#ifdef __CUDACC__
		__host__ __device__
#endif	
		Obj() : mId(-1), mRa(99), mDec(99) { }

#ifdef __CUDACC__
		__host__ __device__
#endif	
		Obj(int64_t id, double ra, double dec) : mId(id), mRa(ra), mDec(dec) { }

#ifdef __CUDACC__
		__host__ __device__
#endif	
		int32_t GetZoneId(double height) const { return (int32_t) floor( (mDec + 90) / height ); }

		friend 
		std::ostream& operator<< (std::ostream &rOs, const Obj &rObj);
	};
}
#endif /* OBJ_H */