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
		double degZoneHeight;
		SegmentManagerPtr segman;

		void Log(std::string msg) const;
		void Sort(SegmentPtr seg) const;

	public:
		Sorter(uint32_t id, SegmentManagerPtr segman, double degZoneHeight) : id(id), segman(segman), degZoneHeight(degZoneHeight) {}
		void operator()();
	};
}
#endif /* SORTER_H */