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
	enum LogLevel { ERROR, WARNING, INFO, PROGRESS, TIMING, DEBUG, DEBUG2, DEBUG3 };

	class Log
	{
		std::ostringstream buffer;
		std::ostream& os;

    public:
		Log(std::ostream& os) : os(os) { }
		~Log();
		std::ostream& Get(LogLevel level);
	};
}

//#define CLOG(cond) if(!(cond)) /* pass */; else Log(std::cout).Get(level)
#define XLOG(level) if( (level) > (verbosity) ) /* pass */; else Log(std::cout).Get(level)

#define LOG_ERR  XLOG(ERROR)
#define LOG_WRN  XLOG(WARNING)
#define LOG_INF  XLOG(INFO)
#define LOG_PRG  XLOG(PROGRESS)
#define LOG_TIM  XLOG(TIMING)
#define LOG_DBG  XLOG(DEBUG)
#define LOG_DBG2 XLOG(DEBUG2)
#define LOG_DBG3 XLOG(DEBUG3)

#endif /* LOG_H */

