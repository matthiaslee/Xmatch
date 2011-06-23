/*
 *   ID:          $Id: BoostXmatch.cpp 6909 2011-06-23 04:37:18Z budavari $
 *   Revision:    $Rev: 6909 $
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
#include <fstream>
#include <cstdlib>

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

/*
	// Calls the provided work function and returns the number of milliseconds 
	// that it takes to call that function.
	template <class Function>
	int64_t time_call(Function&& f)
	{
	   int64_t begin = GetTickCount();
	   f();
	   return GetTickCount() - begin;
	}
*/

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

		int Uni(int max)
		{
			boost::mutex::scoped_lock lock(mtx);
			boost::uniform_int<> uni_dist(0,max);
			boost::variate_generator<base_generator_type&, boost::uniform_int<> > uni(*generator, uni_dist);		
			return uni();
		}
	};
	Random gRand(42u);

	
	struct Cartesian
	{
		double x, y, z;
		Cartesian() : x(0), y(0), z(0) {}
		Cartesian(double x, double y, double z) : x(x), y(y), z(z) {}

		friend std::ostream& operator<< (std::ostream& out, const Cartesian& o) 
		{
			out << o.x << " " << o.y << " " << o.z;
			return out;
		}
	};


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


	class Segment
	{
	public:
		uint32_t id, num;
		bool sorted;
		Object *obj;

		Segment(uint32_t id, uint32_t num) : id(id), num(num), sorted(false) 
		{
			obj = new Object[num];
			Log("new-ed");
		}

		~Segment()
		{
			if (obj != NULL)
			{
				delete[] obj;
				obj = NULL;
				Log("delete[]-ed");
			}
			else 
			{
				Log("empty");
			}
		}
		///*
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
		//*/

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


	enum JobStatus { pending, running, finished };


	class Job
	{
	public:
		JobStatus status;
		SegmentPtr segA, segB;
		bool swap;

		Job(SegmentPtr a, SegmentPtr b, bool swap) : segA(a), segB(b), swap(swap), status(pending) { }

		std::string ToString() const
		{
			std::stringstream ss;
			ss << "Job " << segA->id << "x" << segB->id; // << ":" << status; 
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
		boost::mutex mtx;
		JobVec jobs;

	public:
		JobManager(const SegmentVec& segA, const SegmentVec& segB, bool swap) 
		{
			for (SegmentVec::size_type iA=0; iA<segA.size(); iA++)
			for (SegmentVec::size_type iB=0; iB<segB.size(); iB++)
			{
				JobPtr job(new Job(segA[iA],segB[iB],swap));
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
		uint32_t id;
		JobPtr oldjob;
		JobManagerPtr jobman;
		fs::path outpath;

		void Log(std::string msg)
		{
			boost::mutex::scoped_lock lock(mtx_cout);
			//if (id!=0) return;
			std::cout 
				<< "Worker " 
				<< id << ":"
				//<< " [" << boost::this_thread::get_id() << "] " 
				<< " \t" << msg << std::endl;
		}

	public:
		Worker(uint32_t id, JobManagerPtr jobman, fs::path prefix) 
			: id(id), jobman(jobman), outpath(prefix), oldjob((Job*)NULL)
		{
			std::stringstream ss; 
			ss << "." << id;
			outpath.replace_extension(ss.str());
		}

		void operator()()
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
							if (oldjob==NULL)
								Log(job->ToString() + " \t[null]");
							else if (job->segA->id==oldjob->segA->id || job->segB->id==oldjob->segB->id)
								Log(job->ToString() + " \t[cached]");
							else
								Log(job->ToString() + " \t[new]");
						}
						// do the work
						// boost::this_thread::sleep(boost::posix_time::milliseconds(job->segA->num * job->segB->num / 1000 + gRand.Uni(1000)));

						for (uint32_t iA=0; iA<job->segA->num; iA++)
						for (uint32_t iB=0; iB<job->segB->num; iB++)
						{
							Object a = job->segA->obj[iA];
							Object b = job->segB->obj[iB];

							// math
							if (a.id == b.id)
							{
								outfile << a.id << " " << b.id << std::endl;
							}
						}

						// done
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
		uint32_t num_threads, num_obj;

		po::options_description options("Options");
		po::variables_map vm;
		try
		{
			options.add_options()
				("out,o", po::value(&ofile)->implicit_value("out"), "pathname prefix for output(s)")
				("radius,r", po::value<double>(&sr_arcsec)->default_value(5), "search radius in arcsec, default is 5\"")
				("zoneheight,z", po::value<double>(&zh_arcsec)->default_value(0), "zone height in arcsec, defaults to radius")
				("threads,t", po::value<uint32_t>(&num_threads)->default_value(1), "number of threads")
				("nobject,n", po::value<uint32_t>(&num_obj)->default_value(0), "number of objects in a segment, defaults to full set")
				("verbose,v", po::value<uint32_t>()->implicit_value(1), "enable verbosity (optionally specify level)")
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
			std::cout << "Subversion: $Rev: 6909 $" << std::endl ;
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

		uint32_t verbose = 0;
		if (vm.count("verbose")) verbose = vm["verbose"].as<uint32_t>();

		fs::path opath(ofile);

		//std::cout << "Input file(s): " << vm["input-file"].as< std::vector<std::string> >() << std::endl;
		if (verbose)
		{
			std::cout << " -1- Input file(s): " << ifiles << std::endl;
			if (!ofile.empty()) std::cout << " -1- Output file: " << opath << std::endl;
			std::cout << " -1- Search radius: " << sr_arcsec << std::endl;                
			std::cout << " -1- Zone height: " << zh_arcsec << std::endl;                
			std::cout << " -1- Verbosity: " << verbose << std::endl;
			std::cout << " -1- # of threads: " << num_threads << std::endl;                
			std::cout << " -1- # of obj/seg: " << num_obj << std::endl;                
		}

		// input files
		fs::path inApath = ifiles[0];
		fs::path inBpath = ifiles[1];

		uintmax_t inAsize, inBsize;
		uint32_t inAlen, inBlen;
		try
		{
			inAsize = fs::file_size(inApath);  
			inBsize = fs::file_size(inBpath);  
			inAlen = (uint32_t) (inAsize / sizeof(Object));
			inBlen = (uint32_t) (inBsize / sizeof(Object));
		}
		catch (std::exception& exc)
		{
			std::cout << "Usage: " << argv[0] << " [options] file(s)" << std::endl << options;
			//std::cout << "Error: " << std::endl << "   File not found! " << std::endl ;
			std::cout << "Error: " << std::endl << "   " << exc.what() << std::endl;
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
			SegmentVec::size_type sid = 0;
			uint32_t len = inAlen;
			while (len > 0)
			{
				uint32_t num = (len > num_obj) ? num_obj : len;
				Segment *s = new Segment(sid++, num); 
				if (verbose>2) std::cout << " -3- id:" << sid << " num:" << num << std::endl;
				file.read( (char*)s->obj, s->num * sizeof(Object));
				segmentsRam.push_back(SegmentPtr(s));
				len -= num;
			}	
		}
		// sort segments of A in parallel [tbd]

		// 
		// loop on file B
		//
		fs::ifstream file(inBpath, std::ios::in | std::ios::binary);
		SegmentVec::size_type sid = 0;
		uint32_t len = inBlen;
		uint32_t wid = 0;
		while (len > 0)
		{
			// load new segments from file
			SegmentVec segmentsFile;
			for (SegmentVec::size_type i=0; i<num_threads; i++)
			{
				uint32_t num = (len > num_obj) ? num_obj : len;
				Segment *s = new Segment(sid++, num); 
				if (verbose>2) std::cout << " -3- sid:" << sid << " num:" << num << std::endl;				
				file.read( (char*)s->obj, s->num * sizeof(Object));
				SegmentPtr sptr(s);
				if (num > 0) segmentsFile.push_back(sptr);
				len -= num;				
			}
			// sort segments in parallel [tbd]

			// job are provided by the manager
			JobManagerPtr jobman(new JobManager(segmentsRam,segmentsFile,swap));

			// create new worker threads
			boost::thread_group threads;	
			for(uint32_t it=0; it<num_threads; it++) 
			{
				Worker w(wid++, jobman, fs::path(ofile));
				threads.create_thread(w);
			}
			// anything to do in main?
			boost::this_thread::yield();

			// wait for threads to finish
			threads.join_all();
		}

		return 0;
	}

} // namespace xmatch


// entry point
int main(int argc, char* argv[])
{
	return xmatch::_main(argc, argv);
}


