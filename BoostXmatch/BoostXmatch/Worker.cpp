#include "Worker.h"

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

namespace fs = boost::filesystem;

namespace xmatch
{
	Worker::Worker(uint32_t id, JobManagerPtr jobman, fs::path prefix) : id(id), jobman(jobman), outpath(prefix), oldjob((Job*)NULL)
	{
		std::ostringstream oss; 
		oss << "." << id;
		outpath.replace_extension(oss.str());
	}


	void Worker::operator()()
	{   
		// open the output file
		fs::ofstream outfile(outpath, std::ios::out | std::ios::binary);
		bool keepProcessing = true;

		while(keepProcessing)  
		{  
			try  
			{  
				JobPtr job = jobman->NextPreferredJob(oldjob);

				if (job == NULL) 
				{
					//Log("Job - - -");
					keepProcessing = false;
				}
				else
				{
					// some info for debug
					{
						if (oldjob==NULL) { 
							//Log(job->ToString() + " \t[null]");
						} else if (job->ShareSegment(*oldjob)) { 
							//Log(job->ToString() + " \t[cached]");
						} else { 
							//Log(job->ToString() + " \t[new]");
						}
					}
					// do the work 
					Segment *sB = job->segB.get();
					job->segA->Match(*sB, job->sr_deg);

					// done
					jobman->SetStatus(job,FINISHED);

					// saved what's loaded on the "gpu" now
					oldjob = job;
				}
			}  
			// Catch specific exceptions first 

			// Catch general so it doesn't go unnoticed
			catch (std::exception& exc)  
			{  
				// Log("Uncaught exception: " + std::string(exc.what()));  
				std::cout << exc.what() << std::endl;
			}  
		}  
	}
}