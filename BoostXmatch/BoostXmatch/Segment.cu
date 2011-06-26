/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#include <sstream>

#include "Segment.h"
#include "Obj.h"

namespace xmatch
{
	Segment::Segment(uint32_t id, uint64_t num) : mId(id), mNum(num), mSorted(false), dObj(num)
	{
		mObj = new Obj[num];
		//Log("new-ed");		
	}

	Segment::~Segment()
	{
		if (mObj != NULL)
		{
			delete[] mObj;
			mObj = NULL;
			//Log("delete[]-ed");
		}
		else 
		{
			//Log("empty");
		}
	}
	/*
	void Log(const char* msg) const
	{
		std::string str(msg);
		Log(str);
	}

	void Log(std::string msg) const
	{
		boost::mutex::scoped_lock lock(mtx_cout);
		std::cout << "Segment " << *this << " " << msg << std::endl;
	}
	*/

	void Segment::Load(std::istream &rIs)
	{
		rIs.read( (char*)mObj, mNum * sizeof(Obj));
		//dObj[0].mRa = 111;
	}

	std::string Segment::ToString(const std::string sep) const
	{
		std::stringstream ss;
		ss << mId; // << sep << sorted;
		return ss.str();
	}
	
	std::string Segment::ToString() const
	{
		return ToString(std::string(":"));
	}

	std::ostream& operator<<(std::ostream &o, const Segment &s)
	{
		o << s.ToString();
		return o;
	}
}