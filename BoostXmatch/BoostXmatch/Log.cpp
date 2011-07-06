#include "Log.h"

#include "boost/date_time/posix_time/posix_time.hpp"
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>

extern boost::mutex mtx_cout;

namespace xmatch
{
	std::ostream& Log::Get(int level) 
	{
		this->buffer << boost::this_thread::get_id() << "  " << boost::posix_time::second_clock().local_time() << "  <" << level << ">  ";
		return this->buffer;
	}

	Log::~Log()
	{
		boost::mutex::scoped_lock lock(mtx_cout);
		this->os << this->buffer.str();
		this->os.flush();
	}
}