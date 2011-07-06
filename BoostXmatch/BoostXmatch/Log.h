/*
 *   ID:          $Id: $
 *   Revision:    $Rev: $
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
		 std::ostream& os;

    public:
		Log(std::ostream& os) : os(os) { }
		~Log();
		std::ostream& Get(int level);
	};
}

// Trick?
// #define xlog(level,verbosity) if( (level) < (verbosity) ) /* pass */; else Logger(level)

#define xlog_1 Log(std::cout).Get(1)
#define xlog_2 Log(std::cout).Get(2)
#define xlog_3 Log(std::cout).Get(3)

#endif /* LOG_H */