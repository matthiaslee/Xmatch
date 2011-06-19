#include <iostream>
#include <string>
#include <list>
#include <memory> // shared_ptr in TR1/C++0x

#include <boost/thread.hpp>
#include <boost/date_time.hpp>  
#include <boost/filesystem.hpp>
//#include <boost/filesystem/fstream.hpp>
#include <boost/program_options.hpp>

#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_int.hpp>
//#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

// This is a typedef for a random number generator.
// Try boost::mt19937 or boost::ecuyer1988 instead of boost::minstd_rand
typedef boost::minstd_rand base_generator_type;

namespace fs = boost::filesystem;
namespace po = boost::program_options;

namespace xmatch
{
	boost::mutex mtx_cout; // for std::cout dumps

	// dump a vector by elements
	template<class T>
	std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
	{
		copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " ")); 
		return os;
	};

	// Calls the provided work function and returns the number of milliseconds 
	// that it takes to call that function.
	template <class Function>
	__int64 time_call(Function&& f)
	{
	   __int64 begin = GetTickCount();
	   f();
	   return GetTickCount() - begin;
	}

	class Random
	{
		base_generator_type *generator;
		boost::mutex mtx;

	public:
		Random(const uint32_t &seed)
		{
			generator = new base_generator_type(seed);
		}

		~Random()
		{
			if (generator != NULL)
			{
				delete generator;
				generator = NULL;
			}
		}

		int Uni()
		{
			boost::mutex::scoped_lock lock(mtx);
			boost::uniform_int<> uni_dist(0,5);
			boost::variate_generator<base_generator_type&, boost::uniform_int<> > uni(*generator, uni_dist);		
			return uni();
		}
	};
	// global for testing
	Random gRand(42u);

	class Segment
	{
	public:
		int id;
		bool sorted;
		float *data;

		Segment(int id, bool sorted) : id(id), sorted(sorted)
		{
			const int dim = 100;
			data = new float[dim];
			//Log("created");
		}

		~Segment()
		{
			if (data == NULL)
			{
				//Log("empty");
			}
			else
			{
				delete[] data;
				//Log("delete[]ed");
			}		
		}

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

		std::string ToString(const std::string &sep) const
		{
			std::stringstream ss;
			ss << id; // << sep << sorted;
			return ss.str();
		}
	
		std::string ToString() const
		{
			return ToString(std::string(":"));
		}

		friend std::ostream& operator<<(std::ostream &o, const Segment &s)
		{
			o << s.ToString();
			return o;
		}
	};
	typedef boost::shared_ptr<Segment> SegmentPtr;
	typedef std::vector<SegmentPtr> SegmentVector;

	typedef enum { pending, running, finished } JobStatus;

	class Job
	{
	public:
		SegmentPtr segA, segB;
		JobStatus status;

		Job(SegmentPtr a, SegmentPtr b) : segA(a), segB(b), status(pending)	{ }

		std::string ToString() const
		{
			std::stringstream ss;
	//		ss << "Job " << segA->id << "x" << segB->id; // << " -> " << status;
			ss << segA->id << " " << segB->id; // << " -> " << status;
			return ss.str();
		}

		friend std::ostream& operator<<(std::ostream &o, const Job &blk)
		{
			o << blk.ToString();
			return o;
		}
	};
	typedef boost::shared_ptr<Job> JobPtr;
	typedef std::vector<JobPtr> JobVector;


	class JobManager
	{
	private:
		boost::mutex mtx;
		JobVector jobs;

	public:
		JobManager(const SegmentVector& segA, const SegmentVector& segB) 
		{
			for (SegmentVector::size_type iA=0; iA<segA.size(); iA++)
			for (SegmentVector::size_type iB=0; iB<segB.size(); iB++)
			{
				JobPtr job(new Job(segA[iA],segB[iB]));
				jobs.push_back(job);
			}		
		}

		//	Prefers jobs with segments that the worker already holds
		JobPtr NextPreferredJob(JobPtr oldjob)
		{
			boost::mutex::scoped_lock lock(mtx);
			JobPtr nextjob((Job*)NULL);
			for (JobVector::size_type i=0; i<jobs.size(); i++)
			{
				JobPtr job = jobs[i];
				if (job->status == pending)
				{
					if (nextjob == NULL) 
					{
						nextjob = job;
						if (oldjob == NULL) 
							break;
					}
					// override if find preloaded segments
					if (job->segA->id == oldjob->segA->id || 
						job->segB->id == oldjob->segB->id ) 
					{
						nextjob = job;
						break;
					}
				}
			}		
			if (nextjob != NULL) nextjob->status = running;
			return nextjob;
		}

		// Race condition with Next...() ???????????????????????
		void SetStatus(JobPtr job, JobStatus status)
		{
			boost::mutex::scoped_lock lock(mtx);
			job->status = status;
		}
	};
	typedef boost::shared_ptr<JobManager> JobManagerPtr;


	class Worker
	{    
	public:
		unsigned id;
		JobManagerPtr jobman;
		JobPtr oldjob;

		Worker(unsigned id, JobManagerPtr jobman) : id(id), jobman(jobman), oldjob((Job*)NULL)
		{
		}

		void Log(std::string msg)
		{
			boost::mutex::scoped_lock lock(mtx_cout);
			//if (id!=0) return;
			std::cout 
				// << "Worker " 
				<< id 
				//<< " [" << boost::this_thread::get_id() << "] " 
				<< " " << msg << std::endl;
		}

		void operator()()
		{   
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
						if (oldjob==NULL)
							Log(job->ToString() + " matching [new] (null)");
	//						Log(job->ToString() + " 0");
						else if (job->segA->id==oldjob->segA->id || job->segB->id==oldjob->segB->id)
							Log(job->ToString() + " matching [cached]");
	//						Log(job->ToString() + " 1");
						else
							Log(job->ToString() + " matching [new]");
	//						Log(job->ToString() + " 2");
						boost::this_thread::sleep(boost::posix_time::milliseconds(1+gRand.Uni()));
						jobman->SetStatus(job,finished);

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
	};

	/*
	Outline:
		1) Main thread: Load all segments from file A 
		2) Pre-process: Sort all segments and save in host mem
		3) Main thread: Load segment for all GPUs from file B
		4) Pre-process: Sort segments
		5) Worker thrd: Loop on A from own queue, and try to steal jobs if done
	*/
	int _main(int argc, char* argv[])
	{
		// parse command line
		po::options_description options("Allowed options");
		po::variables_map vm;
		int num_threads;
		try
		{
			// visible options
			options.add_options()
				("help,h", "produce help message")
				("nthreads,n", po::value<int>(&num_threads)->default_value(-1), "number of threads")
				("verbose,v", po::value<int>()->implicit_value(1), "enable verbosity (optionally specify level)")
				// ("input-file", po::value< std::vector<std::string> >(), "input file")
			;
			// hidden options
			po::options_description opt_hidden("Hidden options");        
			opt_hidden.add_options()
				("input-file", po::value< std::vector<std::string> >(), "input file")
			;        
			// compose options
			po::options_description opt_cmd("Command options");
			opt_cmd.add(options).add(opt_hidden);

			// input files without switch
			po::positional_options_description p;
			p.add("input-file", -1);

			// parse
			po::store(po::command_line_parser(argc, argv).options(opt_cmd).positional(p).run(), vm);
			po::notify(vm);
		}
		catch (std::exception& exc)
		{
            std::cout << "Usage: " << argv[0] << " [options] file(s)" << std::endl;
            std::cout << options;
			std::cout << "Error: " << std::endl;
			std::cout << exc.what() << std::endl;
            return 1;
		}

        if (vm.count("help")) 
		{
            std::cout << "Usage: " << argv[0] << " [options] file(s)" << std::endl;
            std::cout << options;
            return 0;
        }
        if (vm.count("input-file"))
        {
            std::cout << "Input file(s): " << vm["input-file"].as< std::vector<std::string> >() << std::endl;
        }
        if (vm.count("verbose")) 
		{
            std::cout << "Verbosity enabled: Level " << vm["verbose"].as<int>() << std::endl;
        }

        std::cout << "Number of threads: " << num_threads << std::endl;                
    
		return 0;

		//std::cout << "RAM" << std::endl;
		

		// load segments from file A
		SegmentVector segmentsRam;
		for (size_t i=0; i<5; i++) 
		{
			Segment *s = new Segment(i, true); // sorted for now
			SegmentPtr sp(s); 
			segmentsRam.push_back(sp);
		}	
		// sort segments of A in parallel [tbd]

		// 
		// loop on file B
		//
		int Nloop = 1;
		for (int loop=0; loop<Nloop; loop++)
		{
			//std::cout << "FILE" << std::endl;

			// load new segments from file B
			SegmentVector segmentsFile;
			size_t num_seg = num_threads;
			for (SegmentVector::size_type i=0; i<num_seg; i++)
			{
				Segment *s = new Segment(i+loop*num_seg,true); // sorted for now
				SegmentPtr sp(s); 
				segmentsFile.push_back(sp);
			}
			// sort segments of B in parallel [tbd]

			// job are provided by the manager
			JobManagerPtr jobman(new JobManager(segmentsRam,segmentsFile));

			//std::cout << "THREADS" << std::endl;

			// create new worker threads
			boost::thread_group threads;	
			for(unsigned id=0; id<num_threads; id++) 
			{
				Worker w(id,jobman);
				threads.create_thread(w);
			}
			// anything to do in main?
			boost::this_thread::yield();

			// wait for threads to finish
			threads.join_all();
		
			//std::cout << "JOIN" << std::endl;
		}

		return 0;
	}

} // namespace xmatch


// entry point
int main(int argc, char* argv[])
{
	return xmatch::_main(argc, argv);
}