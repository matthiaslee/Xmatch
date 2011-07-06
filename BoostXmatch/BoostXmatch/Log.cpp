#include "Log.h"

#include "boost/date_time/posix_time/posix_time.hpp"
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>

extern boost::mutex mtx_cout;

namespace xmatch
{
	Log::Log(int level) : level(level)
	{ 
		buffer << level << "> " << boost::posix_time::second_clock().local_time() << " : " << boost::this_thread::get_id() << " -- ";
	}

	Log::~Log()
	{
		boost::mutex::scoped_lock lock(mtx_cout);
		std::cout << buffer.str() << std::endl;
	}
}