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
		CudaManagerPtr cuman;
		SegmentManagerPtr segman;
		double zh_deg, sr_deg;
		int verbosity;

		void Sort(SegmentPtr seg);

	public:
		Sorter(CudaManagerPtr cuman, uint32_t id, SegmentManagerPtr segman, double sr_deg, double zh_deg, int verbosity) 
			: cuman(cuman), id(id), segman(segman), zh_deg(zh_deg), sr_deg(sr_deg), verbosity(verbosity) 
		{ }

		void operator()();
	};
}
#endif /* SORTER_H */

