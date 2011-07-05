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
#include <sstream>

#include "Sorter.h"
#include "Worker.h"

#pragma warning(push)
//#pragma warning(disable: 4996)      // Thrust's use of strerror
//#pragma warning(disable: 4251)      // STL class exports
#pragma warning(disable: 4005)      // BOOST_COMPILER macro redefinition
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/date_time.hpp>  
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/program_options.hpp>
#pragma warning(pop)


namespace fs = boost::filesystem;
namespace po = boost::program_options;


namespace xmatch
{
	// dump a vector by elements
	template<class T>
	std::ostream& operator<< (std::ostream& os, const std::vector<T>& v)
	{
		copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout," ")); 
		return os;
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
		uint32_t num_threads, num_obj, verbose, n_zones, nz;
		double zh_arcsec, zh_deg;
		double sr_arcsec, sr_deg, sr_rad, sr_dist2;
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
			std::cout << " -2- # of objects for RAM: " << pmt.fileA.size / sizeof(Obj) << std::endl
					  << " -2- # of objects for FIL: " << pmt.fileB.size / sizeof(Obj) << std::endl;		

		//
		// load segments from small file
		// 
		if (pmt.verbose>1) std::cout << " -2- Reading file" << std::endl;
		SegmentVec segmentsRam;
		{
			fs::ifstream file(pmt.fileA.path, std::ios::in | std::ios::binary);
			uint64_t len = pmt.fileA.size / sizeof(Obj);
			uint32_t sid = 0;

			// load segments
			while (len > 0)
			{
				uint64_t num = (len > pmt.num_obj) ? pmt.num_obj : len;
				Segment *s = new Segment(sid++, num); 
				if (pmt.verbose>2) std::cout << " -3- id:" << sid << " num:" << num << std::endl;
				s->Load(file);
				segmentsRam.push_back(SegmentPtr(s));
				len -= num;
			}	

			if (pmt.verbose>1) std::cout << " -2- Sorting segments" << std::endl;
			// sort segments
			{
				SegmentManagerPtr segman(new SegmentManager(segmentsRam));
				boost::thread_group sorters;	
				for (uint32_t it=0; it<pmt.num_threads; it++) 
					sorters.create_thread(Sorter(it, segman, pmt.zh_arcsec/3600));
				sorters.join_all();
			}

			for (uint32_t it=0; it<segmentsRam.size(); it++) 
				std::cout << segmentsRam[it]->vId[0] << std::endl;
		}

		// 
		// loop on larger file
		//
		fs::ifstream file(pmt.fileB.path, std::ios::in | std::ios::binary);
		uint64_t len = pmt.fileB.size / sizeof(Obj);
		uint32_t sid = 0;
		uint32_t wid = 0;
		while (len > 0)
		{
			SegmentVec segmentsFile;
			// load segments
			for (uint64_t i=0; i<pmt.num_threads; i++)
			{
				uint64_t num = (len > pmt.num_obj) ? pmt.num_obj : len;
				Segment *s = new Segment(sid++, num); 
				if (pmt.verbose>2) std::cout << " -3- sid:" << sid << " num:" << num << std::endl;				
				s->Load(file);
				if (num > 0) segmentsFile.push_back(SegmentPtr(s));
				len -= num;				
			}

			if (pmt.verbose>1) std::cout << " -2- Sorting segments" << std::endl;
			// sort segments
			{
				SegmentManagerPtr segman(new SegmentManager(segmentsFile));
				boost::thread_group sorters;	
				for (uint32_t it=0; it<pmt.num_threads; it++) 
					sorters.create_thread(Sorter(it, segman, pmt.zh_arcsec/3600));
				sorters.join_all();
			}

			if (pmt.verbose>1) std::cout << " -2- Processing jobs" << std::endl;
			// process jobs
			{
				JobManagerPtr jobman(new JobManager(segmentsRam,segmentsFile,swap,pmt.sr_arcsec/3600));
				boost::thread_group workers;	
				for (uint32_t it=0; it<pmt.num_threads; it++) 
				{
					std::ostringstream oss;	     
					oss << "." << wid++;
					Worker w = Worker(it, jobman, pmt.outpath.string() + oss.str());
					workers.create_thread(w);
				}
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
