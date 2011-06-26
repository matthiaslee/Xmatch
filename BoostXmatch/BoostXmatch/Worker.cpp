#include "Worker.h"

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

namespace fs = boost::filesystem;

namespace xmatch
{
	void Worker::Log(std::string msg)
	{
		//boost::mutex::scoped_lock lock(mtx_cout);
		//if (id!=0) return;
		std::cout 
			<< "Worker " 
			<< id << ":"
			//<< " [" << boost::this_thread::get_id() << "] " 
			<< " \t" << msg << std::endl;
	}

	
	Worker::Worker(uint32_t id, JobManagerPtr jobman, fs::path prefix) : id(id), jobman(jobman), outpath(prefix), oldjob((Job*)NULL)
	{
		std::stringstream ss; 
		ss << "." << id;
		outpath.replace_extension(ss.str());
	}


	void Worker::operator()()
	{   
		// open the output file
		fs::ofstream outfile(outpath, std::ios::out | std::ios::app); //|std::ios::binary);
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
						if (oldjob==NULL)
							Log(job->ToString() + " \t[null]");
						else if (job->ShareSegment(*oldjob))
							Log(job->ToString() + " \t[cached]");
						else
							Log(job->ToString() + " \t[new]");
					}
					// do the work
					//boost::this_thread::sleep(boost::posix_time::milliseconds(job->segA->mNum * job->segB->mNum / 1000 + gRand.Uni(1000)));

					for (uint32_t iA=0; iA<job->segA->mNum; iA++)
					for (uint32_t iB=0; iB<job->segB->mNum; iB++)
					{
						Obj a = job->segA->mObj[iA];
						Obj b = job->segB->mObj[iB];

						// math
						if (a.mId == b.mId)
						{
							outfile << a.mId << " " << b.mId << std::endl;
						}
					}

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
				Log("Uncaught exception: " + std::string(exc.what()));  
			}  
		}  
	}
}