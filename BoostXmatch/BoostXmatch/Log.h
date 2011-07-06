/*
 *   ID:          $Id$
 *   Revision:    $Rev$
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

#define clog(cond) if(!(cond)) /* pass */; else Log(std::cout).Get(level)
#define xlog(level) if( (level) > (verbosity) ) /* pass */; else Log(std::cout).Get(level)



#endif /* LOG_H */