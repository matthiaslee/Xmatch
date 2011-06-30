#include "Sorter.h"

namespace xmatch
{
	void Sorter::operator()()
	{   
		bool keepProcessing = true;

		while(keepProcessing)  
		{  
			//try  
			{  
				SegmentPtr seg = segman->Next();

				if (seg == NULL) 
				{
					// Log("-");
					keepProcessing = false;
				}
				else
				{
					// do the work
					seg->Sort(zh_arcsec);
				}
			}  
			// Catch specific exceptions first 
			// ...
			// Catch general so it doesn't go unnoticed
			//catch (std::exception& exc)  {  std::cout << exc.what() << std::endl;	}  
		}  
	}



}
