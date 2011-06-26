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

//#include <thrust/host_vector.h>


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


struct Object;
class Segment;


/*
__host__ __device__ func()
{
#if __CUDA_ARCH__ == 100
    // Device code path for compute capability 1.0
#elif __CUDA_ARCH__ == 200
    // Device code path for compute capability 2.0
#elif !defined(__CUDA_ARCH__)
    // Host code path
#endif
}*/

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

/* DOES NOT WORK ON LINUX *

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

	/*
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
	*/
	
	typedef boost::shared_ptr<Segment> SegmentPtr;
	typedef std::vector<SegmentPtr> SegmentVec;


	class SegmentManager
	{
		boost::mutex mtx;
		SegmentVec seg;
		uint32_t index;

	public:
		SegmentManager(SegmentVec& seg) : seg(seg), index(0) {}

		SegmentPtr Next()
		{
			if (index < seg.size())
			{
				boost::mutex::scoped_lock lock(mtx);
				return seg[index++];
			}
			else
			{
				return SegmentPtr((Segment*)NULL);
			}
		}
	};
	typedef boost::shared_ptr<SegmentManager> SegmentManagerPtr;


	class Sorter
	{    		
		uint32_t id;
		SegmentManagerPtr segman;

		void Log(std::string msg)
		{
			boost::mutex::scoped_lock lock(mtx_cout);
			//if (id!=0) return;
			std::cout 
				<< "Sorter " 
				<< id << ":"
				//<< " [" << boost::this_thread::get_id() << "] " 
				<< " \t" << msg << std::endl;
		}

	public:
		Sorter(uint32_t id, SegmentManagerPtr segman) : id(id), segman(segman) {}

		void operator()()
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
						boost::this_thread::sleep(boost::posix_time::milliseconds(gRand.Uni(1000)));
						seg->sorted = true;
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
		Worker(uint32_t id, JobManagerPtr jobman, fs::path prefix) : id(id), jobman(jobman), outpath(prefix), oldjob((Job*)NULL)
		{
			std::stringstream ss; 
			ss << "." << id;
			outpath.replace_extension(ss.str());
		}

		void operator()()
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
							else if (job->segA->id==oldjob->segA->id || job->segB->id==oldjob->segB->id)
								Log(job->ToString() + " \t[cached]");
							else
								Log(job->ToString() + " \t[new]");
						}
						// do the work
						boost::this_thread::sleep(boost::posix_time::milliseconds(job->segA->num * job->segB->num / 1000 + gRand.Uni(1000)));

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

	struct FileDesc
	{
		fs::path path;
		uintmax_t size;

		FileDesc() : path(""), size(0) {}

		FileDesc(fs::path path) : path(path), size(0)
		{
			if (fs::is_regular_file(path))
				size = fs::file_size(path);
		}
			
		friend std::ostream& operator<< (std::ostream& out, const FileDesc& fd) 
		{
			out << fd.path;
			return out;
		}
	};


	struct Parameter
	{
		uint32_t num_threads, num_obj, verbose;
		double zh_arcsec, sr_arcsec;
		FileDesc fileA, fileB;
		fs::path outpath;
		int error;
		
		//Parameter() : error(0), verbose(0), num_threads(0), num_obj(0), zh_arcsec(0), sr_arcsec(0), fileA(""), fileB(""), outpath("") {}
		Parameter(int argc, char* argv[]) : error(0), verbose(0)
		{
			po::options_description options("Options");
			po::variables_map vm;
			std::string ofile;
			std::vector<std::string> ifiles;
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
				error = 1;
				return;
			}
			if (vm.count("help")) 
			{
				std::cout << "Usage: " << argv[0] << " [options] file(s)" << std::endl << options;
				std::cout << "Subversion: $Rev$" << std::endl ;
				error = 1;
				return;
			}
			if (!vm.count("input") || ifiles.size() != 2)
			{
				std::cout << "Usage: " << argv[0] << " [options] file(s)" << std::endl << options;
				std::cout << "Error: " << std::endl << "   Input 2 files!" << std::endl ;
				std::cout << "Got this: " << std::endl << ifiles << std::endl;
				error = 2;
				return;
			}
			// default zone height is radius
			if (zh_arcsec == 0) zh_arcsec = sr_arcsec;
			if (vm.count("verbose")) verbose = vm["verbose"].as<uint32_t>();
			fileA = FileDesc(ifiles[0]);
			fileB = FileDesc(ifiles[1]);
			if (fileA.size == 0 || fileB.size == 0)
			{
				std::cout << "Usage: " << argv[0] << " [options] file(s)" << std::endl << options;
				std::cout << "Error: " << std::endl << "   File not found! " << std::endl ;
				error = 3;
				return;
			}
			outpath = ofile;
		}
		
		friend std::ostream& operator<< (std::ostream& out, const Parameter& p) 
		{
			out << " -1- Input file(s): " << p.fileA << " " << p.fileB << std::endl;
			out << " -1- Output file: " << p.outpath << std::endl;
			out << " -1- Search radius: " << p.sr_arcsec << std::endl;                
			out << " -1- Zone height: " << p.zh_arcsec << std::endl;                
			out << " -1- Verbosity: " << p.verbose << std::endl;
			out << " -1- # of threads: " << p.num_threads << std::endl;                
			out << " -1- # of obj/seg: " << p.num_obj;                
			return out;
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
		Parameter pmt(argc,argv);
		if (pmt.error) return pmt.error;
		if (pmt.verbose) std::cout << pmt << std::endl;

		// swap if B is smaller
		bool swap = false;
		if (pmt.fileB.size < pmt.fileA.size)
		{
			if (pmt.verbose > 1) std::cout << " -2- Swapping order of files" << std::endl;
			swap = true;
			std::swap(pmt.fileA, pmt.fileB);
		}
		if (pmt.verbose>1) 
			std::cout << " -2- # of objects for RAM: " << pmt.fileA.size / sizeof(Object) << std::endl
					  << " -2- # of objects for FIL: " << pmt.fileB.size / sizeof(Object) << std::endl;		

		//
		// load segments from small file
		// 
		if (pmt.verbose>1) std::cout << " -2- Reading file" << std::endl;
		SegmentVec segmentsRam;
		{
			fs::ifstream file(pmt.fileA.path, std::ios::in | std::ios::binary);
			uint64_t len = pmt.fileA.size / sizeof(Object);
			uint32_t sid = 0;
			// load segments
			while (len > 0)
			{
				uint64_t num = (len > pmt.num_obj) ? pmt.num_obj : len;
				Segment *s = new Segment(sid++, num); 
				if (pmt.verbose>2) std::cout << " -3- id:" << sid << " num:" << num << std::endl;
				file.read( (char*)s->obj, s->num * sizeof(Object));
				segmentsRam.push_back(SegmentPtr(s));
				len -= num;
			}	
			// sort segments
			if (pmt.verbose>1) std::cout << " -2- Sorting segments" << std::endl;
			{
				SegmentManagerPtr segman(new SegmentManager(segmentsRam));
				boost::thread_group sorters;	
				for (uint32_t it=0; it<pmt.num_threads; it++) 
					sorters.create_thread(Sorter(it, segman));
				sorters.join_all();
			}
		}

		// 
		// loop on larger file
		//
		fs::ifstream file(pmt.fileB.path, std::ios::in | std::ios::binary);
		uint64_t len = pmt.fileB.size / sizeof(Object);
		uint32_t wid = 0;
		uint32_t sid = 0;
		while (len > 0)
		{
			// load segments
			SegmentVec segmentsFile;
			for (uint64_t i=0; i<pmt.num_threads; i++)
			{
				uint64_t num = (len > pmt.num_obj) ? pmt.num_obj : len;
				Segment *s = new Segment(sid++, num); 
				if (pmt.verbose>2) std::cout << " -3- sid:" << sid << " num:" << num << std::endl;				
				file.read( (char*)s->obj, s->num * sizeof(Object));
				if (num > 0) segmentsFile.push_back(SegmentPtr(s));
				len -= num;				
			}
			// sort segments
			if (pmt.verbose>1) std::cout << " -2- Sorting segments" << std::endl;
			{
				SegmentManagerPtr segman(new SegmentManager(segmentsFile));
				boost::thread_group sorters;	
				for (uint32_t it=0; it<pmt.num_threads; it++) 
					sorters.create_thread(Sorter(it, segman));
				sorters.join_all();
			}
			// process jobs
			if (pmt.verbose>1) std::cout << " -2- Processing jobs" << std::endl;
			{
				JobManagerPtr jobman(new JobManager(segmentsRam,segmentsFile,swap));
				boost::thread_group workers;	
				for (uint32_t it=0; it<pmt.num_threads; it++) 
					workers.create_thread(Worker(wid++, jobman, pmt.outpath));
				workers.join_all();
			}
		}
		return 0;
	}

} // namespace xmatch


// entry point
int main(int argc, char* argv[])
{
	return xmatch::_main(argc, argv);
}
