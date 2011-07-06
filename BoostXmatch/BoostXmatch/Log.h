/*
 *   ID:          $Id: Worker.h 7016 2011-07-06 01:35:38Z budavari $
 *   Revision:    $Rev: 7016 $
 */
#pragma once
#ifndef LOG_H
#define LOG_H
#include <sstream>

namespace xmatch
{	
	class Log
	{
		 std::ostringstream buffer;
		 int level;

    public:
		Log(int level);
		~Log();

		template <typename T>
		Log& operator<< (const T& t)
		{
			buffer << t;
			return *this;
		}
	};
}

// Trick?
// #define xlog(level,verbosity) if( (level) < (verbosity) ) /* pass */; else Logger(level)

#define xlog_1 Log(1)
#define xlog_2 Log(2)
#define xlog_3 Log(3)

#endif /* LOG_H */