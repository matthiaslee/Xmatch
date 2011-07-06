#include "Log.h"

#pragma warning(push)
#pragma warning(disable: 4005)      // BOOST_COMPILER macro redefinition
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#pragma warning(pop)


namespace xmatch
{
	// global mutex
	extern boost::mutex mtx_log;

	std::ostream& Log::Get(LogLevel level) 
	{
		this->buffer 
			<< level << "> "
			<< boost::this_thread::get_id() << "  " 
			<< boost::posix_time::microsec_clock().local_time()
			<< "  ";
		return this->buffer;
	}

	Log::~Log()
	{
		boost::mutex::scoped_lock lock(mtx_log);
		this->os << this->buffer.str();
		this->os.flush();
	}
}