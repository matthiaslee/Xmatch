/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#pragma once
#ifndef SORTER_H
#define SORTER_H
#include "SegmentManager.h"

namespace xmatch
{	
	class Sorter
	{    		
		uint32_t id;
		double zh_deg;
		SegmentManagerPtr segman;

		void Sort(SegmentPtr seg);

	public:
		Sorter(uint32_t id, SegmentManagerPtr segman, double zh_deg) : id(id), segman(segman), zh_deg(zh_deg) {}
		void operator()();
	};
}
#endif /* SORTER_H */