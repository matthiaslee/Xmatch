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
		SegmentManagerPtr segman;

		void Log(std::string msg);

	public:
		Sorter(uint32_t id, SegmentManagerPtr segman) : id(id), segman(segman) {}
		void operator()();
	};
}
#endif /* SORTER_H */