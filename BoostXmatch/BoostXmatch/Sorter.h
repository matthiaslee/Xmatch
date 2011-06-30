/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#pragma once
#ifndef SORTER_H
#define SORTER_H

#include <cstdint>
#include <iostream>

#include "SegmentManager.h"

namespace xmatch
{
	
	class Sorter
	{    		
		uint32_t id;
		double zh_arcsec;
		SegmentManagerPtr segman;

		void Sort(SegmentPtr seg) const;

	public:
		Sorter(uint32_t id, SegmentManagerPtr segman, double zh_arcsec) : id(id), segman(segman), zh_arcsec(zh_arcsec) {}
		void operator()();
	};
}
#endif /* SORTER_H */