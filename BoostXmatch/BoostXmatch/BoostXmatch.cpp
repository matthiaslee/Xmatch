/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */

/*
Copyright (c) 2011 Tamas Budavari <budavari.at.deleteme.jhu.edu>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
/* MIT license http://www.opensource.org/licenses/mit-license.php */


#include <iostream>
#include <string>
#include <list>

#include <math.h>

#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/date_time.hpp>  
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/program_options.hpp>

#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_int.hpp>
//#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
// This is a typedef for a random number generator; try boost::mt19937 or boost::ecuyer1988 instead of boost::minstd_rand
typedef boost::minstd_rand base_generator_type;

namespace fs = boost::filesystem;
namespace po = boost::program_options;

#define RAD2DEG 57.295779513082323
#define THREADS_PER_BLOCK 512

namespace xmatch
{
	boost::mutex mtx_cout; // for std::cout dumps

	// dump a vector by elements
	template<class T>
	std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
	{
		copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout," ")); 
		return os;
	};

	// Calls the provided work function and returns the number of milliseconds 
	// that it takes to call that function.
	template <class Function>
	int64_t time_call(Function&& f)
	{
	   int64_t begin = GetTickCount();
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


	struct Object
	{
		int64_t id;
		double ra, dec;

		//__host__ __device__	
		Object() : id(-1), ra(0), dec(0) {}

		//__host__ __device__
		Object(int64_t id, double ra, double dec) : id(id), ra(ra), dec(dec) {}

		//__host__ __device__
		int zoneid(double height) const
		{
			return 0; // (int) rint( (dec+90)/height );
		}

		friend std::ostream& operator<< (std::ostream& out, const Object& o) 
		{
			out << o.id << " " << o.ra << " " << o.dec;
			return out;
		}
	};
/*
	typedef boost::shared_ptr<Object> ObjectPtr;
	typedef std::vector<ObjectPtr> ObjectVec;
*/

	class Segment
	{
	public:
		int id, num;
		bool sorted;
		Object *obj;

		Segment(int id, int num) : id(id), num(num), sorted(false) //, obj((Object*)NULL)
		{
			obj = new (std::nothrow) Object[num];
		}

		~Segment()
		{
			if (obj != NULL)
			{
				delete[] obj;
				obj = NULL;
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
	typedef std::vector<SegmentPtr> SegmentVec;


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
			ss << segA->id << "x" << segB->id << ":" << status; 
			return ss.str();
		}

		friend std::ostream& operator<<(std::ostream &o, const Job &blk)
		{
			o << blk.ToString();
			return o;
		}
	};
	typedef boost::shared_ptr<Job> JobPtr;
	typedef std::vector<JobPtr> JobVec;


	class JobManager
	{
	private:
		boost::mutex mtx;
		JobVec jobs;

	public:
		JobManager(const SegmentVec& segA, const SegmentVec& segB) 
		{
			for (SegmentVec::size_type iA=0; iA<segA.size(); iA++)
			for (SegmentVec::size_type iB=0; iB<segB.size(); iB++)
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
			for (JobVec::size_type i=0; i<jobs.size(); i++)
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

		// TODO: Is this a potential race-condition problem with Next...() 
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
		int id;
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
		1) Main thread: Load all segments from smaller file 
		2) Pre-process: Sort all segments and save in host mem
		3) Main thread: Load enough segments for GPUs from the larger file
		4) Pre-process: Sort them
		5) Worker thrd: Do the job(s)
	*/
	int _main(int argc, char* argv[])
	{
		// parse command line
		std::vector<std::string> ifiles;
		std::string ofile;
		double zh_arcsec, sr_arcsec;
		int num_threads, num_obj;

		po::options_description options("Options");
		po::variables_map vm;
		try
		{
			options.add_options()
				("out,o", po::value(&ofile)->implicit_value("out"), "pathname prefix for output(s)")
				("radius,r", po::value<double>(&sr_arcsec)->default_value(5), "search radius in arcsec, default is 5\"")
				("zoneheight,z", po::value<double>(&zh_arcsec)->default_value(0), "zone height in arcsec, defaults to radius")
				("threads,t", po::value<int>(&num_threads)->default_value(1), "number of threads")
				("nobject,n", po::value<int>(&num_obj)->default_value(0), "number of objects in a segment, defaults to full set")
				("verbose,v", po::value<int>()->implicit_value(1), "enable verbosity (optionally specify level)")
				("help,h", "print help message")
			;
			// hidden 
			po::options_description opt_hidden("Hidden options");        
			opt_hidden.add_options()
				("input", po::value< std::vector<std::string> >(&ifiles), "input file")
			;        
			// all options
			po::options_description opt_cmd("Command options");
			opt_cmd.add(options).add(opt_hidden);

			// use input files without the switch
			po::positional_options_description p;
			p.add("input", -1);

			// parse...
			po::store(po::command_line_parser(argc,argv).options(opt_cmd).positional(p).run(),vm);
			po::notify(vm);
		}
		catch (std::exception& exc)
		{
            std::cout << "Usage: " << argv[0] << " [options] file(s)" << std::endl << options;
			std::cout << "Error: " << std::endl << "   " << exc.what() << std::endl;
            return 1;
		}
		if (vm.count("help")) 
		{
            std::cout << "Usage: " << argv[0] << " [options] file(s)" << std::endl << options;
			std::cout << "Subversion: $Rev$" << std::endl ;
            return 0;
        }
		if (!vm.count("input") || ifiles.size() != 2)
		{
			std::cout << "Usage: " << argv[0] << " [options] file(s)" << std::endl << options;
			std::cout << "Error: " << std::endl << "   Input 2 files!" << std::endl ;
			std::cout << "Got this: " << std::endl << ifiles << std::endl;
            return 2;
		}
		// default zone height is radius
		if (zh_arcsec == 0) zh_arcsec = sr_arcsec;

		int verbose = 0;
		if (vm.count("verbose")) verbose = vm["verbose"].as<int>();

		fs::path opath(ofile);

		//std::cout << "Input file(s): " << vm["input-file"].as< std::vector<std::string> >() << std::endl;
		if (verbose)
		{
			std::cout << " -1- Input file(s): " << ifiles << std::endl;
			if (!ofile.empty()) std::cout << " -1- Output file: " << opath << std::endl;
			std::cout << " -1- Search radius: " << sr_arcsec << std::endl;                
			std::cout << " -1- Zone height: " << zh_arcsec << std::endl;                
			std::cout << " -1- Verbosity: " << vm["verbose"].as<int>() << std::endl;
			std::cout << " -1- # of threads: " << num_threads << std::endl;                
			std::cout << " -1- # of obj/seg: " << num_obj << std::endl;                
		}

		// input files
		fs::path inApath = ifiles[0];
		fs::path inBpath = ifiles[1];

		uintmax_t inAsize, inBsize, inAlen, inBlen;
		try
		{
			inAsize = fs::file_size(inApath);  
			inBsize = fs::file_size(inBpath);  
			inAlen = inAsize / sizeof(Object);
			inBlen = inBsize / sizeof(Object);
		}
		catch (std::exception& exc)
		{
			std::cout << "Usage: " << argv[0] << " [options] file(s)" << std::endl << options;
			std::cout << "Error: " << std::endl << "   File not found! " << std::endl ;
			return 3;
		}
		// swap if B is smaller
		bool swap = false;
		if (inBlen < inAlen)
		{
			if (verbose > 1) std::cout << " -2- Swapping order of files" << std::endl;
			swap = true;
			std::swap(inAlen,inBlen);
			std::swap(inApath,inBpath);
		}
		if (verbose>1) 
			std::cout << " -2- # of objects for RAM: " << inAlen << std::endl
					  << " -2- # of objects for FIL: " << inBlen << std::endl;		

		// load segments from file A
		if (verbose>1) std::cout << " -2- Reading smaller file" << std::endl;
		SegmentVec segmentsRam;
		{
			fs::ifstream file(inApath, std::ios::in | std::ios::binary);
			int id = 0;
			int len = inAlen;
			while (len > 0)
			{
				int num = (len > num_obj) ? num_obj : len;
				if (verbose>2) std::cout << " -3- id:" << id << " num:" << num << std::endl;
				len -= num;
				// create and load segment
				Segment *s = new Segment(id++, num); 
				file.read((char*)s->obj,num*sizeof(Object));
				std::cout << ":: " << s->obj[0] << std::endl;
				segmentsRam.push_back(SegmentPtr(s));
			}	
		}
		// sort segments of A in parallel [tbd]

		return 0;

		// 
		// loop on file B
		//
		int Nloop = 1;
		for (int loop=0; loop<Nloop; loop++)
		{
			//std::cout << "FILE" << std::endl;

			// load new segments from file B
			SegmentVec segmentsFile;
			size_t num_seg = num_threads;
			for (SegmentVec::size_type i=0; i<num_seg; i++)
			{
				Segment *s = NULL; //new Segment(i+loop*num_seg,true); // sorted for now
				SegmentPtr sp(s); 
				segmentsFile.push_back(sp);
			}
			// sort segments of B in parallel [tbd]

			// job are provided by the manager
			JobManagerPtr jobman(new JobManager(segmentsRam,segmentsFile));

			//std::cout << "THREADS" << std::endl;

			// create new worker threads
			boost::thread_group threads;	
			for(int id=0; id<num_threads; id++) 
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