/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#pragma once
#ifndef SORTER_H
#define SORTER_H
#include "SegmentManager.h"
#include "CudaManager.h"

namespace xmatch
{	
	class Sorter
	{    		
		uint32_t id;
		double zh_deg;
		SegmentManagerPtr segman;
		CudaManagerPtr cuman;

		void Sort(SegmentPtr seg);

	public:
		Sorter(CudaManagerPtr cuman, uint32_t id, SegmentManagerPtr segman, double zh_deg) : cuman(cuman), id(id), segman(segman), zh_deg(zh_deg) {}
		void operator()();
	};
}
#endif /* SORTER_H */