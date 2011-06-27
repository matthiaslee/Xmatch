#include "Sorter.h"

namespace xmatch
{
	void Sorter::Log(std::string msg) const
	{
		// boost::mutex::scoped_lock lock(mtx_cout);
		//if (id!=0) return;
		std::cout 
			<< "Sorter " 
			<< id << ":"
			//<< " [" << boost::this_thread::get_id() << "] " 
			<< " \t" << msg << std::endl;
	}

	void Sorter::operator()()
	{   
		bool keepProcessing = true;

		while(keepProcessing)  
		{  
			try  
			{  
				SegmentPtr seg = segman->Next();

				if (seg == NULL) 
				{
					Log("-");
					keepProcessing = false;
				}
				else
				{
					// do the work
					Log(seg->ToString());
					seg->Sort(degZoneHeight);
				}
			}  
			// Catch specific exceptions first 

			// Catch general so it doesn't go unnoticed
			catch (std::exception& exc)  
			{  
				Log("Uncaught exception: " + std::string(exc.what()));  
			}  
		}  
	}



}
