///
// BinConv.c
//
#include <stdafx.h>

#include <iostream>
#include <fstream>
#include <cstdlib>

#ifdef __GNUC__
#ifndef __int64
#define __int64 long long
#endif
#ifndef __uint64
#define __uint64 unsigned long long
#endif
#endif

/*
	Usage
*/
void usage(int argc, char* argv[], const char* msg, int code)
{
	std::cerr << "Syntax:" << std::endl;
	std::cerr << "  " << argv[0] << " idradec1.bin n1 idradec2.bin n2 zh_arcsec sr_arcsec flag out.bin num" << std::endl;
	if (msg != NULL)
		std::cerr << "ERROR: " << msg << std::endl;
	exit(code);
}

/*
	Objects
*/
struct object
{
	__int64 id;
	double ra, dec;

	__host__ __device__
	object() : id(-1), ra(0), dec(0) {}

	__host__ __device__
	object(__int64 objid, double ra_deg, double dec_deg)
		: id(objid), ra(ra_deg), dec(dec_deg) {}

	__host__ __device__
	int zoneid(double height) const
	{
		return (int) rint( (dec+90)/height );
	}
};

/*
thrust::host_vector<object> load_objects_bin(const char *fname)
{
	object* o = thrust::raw_pointer_cast(&obj[0]); // ptr to array
	std::ifstream myfile (fname, std::ios::in | std::ios::binary);

	// get size of file
	myfile.seekg(0,ifstream::end);
    size_t size=myfile.tellg();
	myfile.seekg(0);

    thrust::host_vector<object>
	if (myfile.is_open())
		myfile.read( (char*)o, obj.size() * sizeof(object) );
	myfile.close();
}

std::ostream& operator<< (std::ostream& out, const object& o)
{
	out << o.id << " " << o.ra << " " << o.dec;
	return out;
}
*/

/*
	Entry point
*/
int main(int argc, char * argv[])
{
	// command line params
	if (argc < 3)
    {
       usage(argc, argv, NULL, 1);
    }
	int c = 1;
	// input file
	char * f_obj = argv[c++];
	int n_o = atoi(argv[c++]);

	// output file
	char * f_out = argv[c++];
	int n_x = atoi(argv[c++]);

	// read and dump
	std::ifstream infile (f_obj, std::ios::in | std::ios::binary);
	std::ifstream outfile(f_out, std::ios::out);

	// get size of file
	infile.seekg(0, std::ifstream::end);
    size_t size = infile.tellg();
	infile.seekg(0);

	// timers
	clock_t clk_start, clk_stop;
	clk_start = clock();

    object o;
    char sep = ',';
    size_t cur = 0;
    size_t objsize = sizeof(object);

	if (infile.is_open() && outfile.is_open())
	{
        while (cur < size)
        {
		      infile.read((char*)&o, objsize);
              cur += objsize;
		      outfile << o.id << sep << o.ra << sep << o.dec << std::endl;
        }
	}
	infile.close();
	outfile.close();

	std::cerr << "[tmr] Dump: " << std::fixed
	 		  << (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl;

	return 0;
}
