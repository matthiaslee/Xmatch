//
// CuXmatch.cu 
//

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
#include <fstream>
#include <cstdlib>

// CUDA and Thrust
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#define RAD2DEG 57.295779513082323
#define THREADS_PER_BLOCK 512

/*
	Objects
*/
struct object
{
	long long id;
	double ra, dec;

	__host__ __device__	
	object() : id(-1), ra(0), dec(0) {}

	__host__ __device__
	object(long long objid, double ra_deg, double dec_deg)
		: id(objid), ra(ra_deg), dec(dec_deg) {}

	__host__ __device__
	int zoneid(double height) const
	{
		return (int) rint( (dec+90)/height );
	}
};

// dummy dump
std::ostream& operator<< (std::ostream& out, const object& o) 
{
	out << o.id << " " << o.ra << " " << o.dec;
	return out;
}

// read binary files
void load_objects_bin(thrust::host_vector<object>& obj, const char * fname)
{
	object* o = thrust::raw_pointer_cast(&obj[0]); // ptr to array
	std::ifstream myfile (fname, std::ios::in | std::ios::binary);
    if (myfile.is_open()) 
		myfile.read( (char*)o, obj.size() * sizeof(object) );
    myfile.close();    
}

// functor for sorting objects
struct less_zonera
{
	double h;

	__host__ __device__
	less_zonera(double height) : h(height) {}

	__host__ __device__ 
	bool operator()(object lhs, object rhs) const
	{
		int lz = lhs.zoneid(h);
		int rz = rhs.zoneid(h);
		if (lz < rz) return true;
		if (lz > rz) return false;
		return lhs.ra < rhs.ra;
	}
}; 

// functor for finding zone boundaries
struct less_zone
{
	double h;

	__host__ __device__
	less_zone(double height) : h(height) {}

	__host__ __device__ 
	bool operator()(const object &lhs, const object &rhs) const
	{
		int lz = lhs.zoneid(h);
		int rz = rhs.zoneid(h);
		return lz < rz;
	}

	__host__ __device__ 
	bool operator()(const object &o, int zone) const
	{
		int z = o.zoneid(h);
		return z < zone;
	}

	__host__ __device__ 
	bool operator()(int zone, const object &o) const
	{
		int z = o.zoneid(h);
		return zone < z;
	}

}; 

// functor for splitting object list
struct get_id
{
	__host__ __device__
	long long operator()(const object& o) const
	{
		return o.id;
	}
};

// functor for splitting object list
struct get_radec_radian
{
	__host__ __device__
	double2 operator()(const object& o) const
	{
		return make_double2(o.ra/RAD2DEG, o.dec/RAD2DEG);
	}
};


// convert ra,dec to xyz
__host__ __device__
void radec2xyz(double2 radec, double *x, double *y, double *z)
{
	double sr, cr, sd, cd;
	sincos(radec.x, &sr, &cr);
	sincos(radec.y, &sd, &cd);
	*x = cd * cr;
	*y = cd * sr;
	*z = sd;
}

// convert ra,dec to xyz (gpu w/ single point and less precise "fast math")
/*
__device__
void radec2xyz(double2 radec, float *x, float *y, float *z)
{
	float sr, cr, sd, cd;
	__sincosf(radec.x, &sr, &cr);
	__sincosf(radec.y, &sd, &cd);
	*x = cd * cr;
	*y = cd * sr;
	*z = sd;
}
*/


// Gray, Nieto & Szalay 2007 (arXiv:cs/0701171v1)
__host__ __device__
double calc_alpha(double theta, double abs_dec)
{
	if ( abs_dec+theta > 89.99 ) return 180;
	return abs( atan(sin(theta / RAD2DEG))
		        / sqrt( abs( cos((abs_dec-theta) / RAD2DEG) 
					       * cos((abs_dec+theta) / RAD2DEG) )) 
			);
}

__host__ __device__
double calc_alpha(double theta, double zh_deg, int zone)
{
	double dec  = abs( zone    * zh_deg - 90);
	double dec2 = abs((zone+1) * zh_deg - 90);
	dec = (dec > dec2 ? dec : dec2);
	return calc_alpha(theta, dec + 1e-3); // with overshoot
}


/*
	CUDA kernel
*/
static __shared__ uint2 s_idx[THREADS_PER_BLOCK];				// indices of objects in pair
static __shared__ bool s_found[THREADS_PER_BLOCK];				// found flags
static __shared__ unsigned short s_sum_gap[THREADS_PER_BLOCK];	// prefix sum of gaps
static __shared__ unsigned short s_sum_fnd[THREADS_PER_BLOCK];  // prefix sum of founds
static __shared__ unsigned short s_tmp[2 * THREADS_PER_BLOCK];  // tmp array for prefix sum
static __shared__ unsigned int s_index;

__global__
void xmatch_kernel(	const double2* p_radec1, int i1s, int i1e, 
					const double2* p_radec2, int i2s, int i2e,
					double sr_rad, double alpha_rad, double sr_dist2,
					uint2 *m_idx,
					unsigned int m_size, unsigned int *m_end)
{
	unsigned int tid =  blockDim.x * threadIdx.y + threadIdx.x;
	unsigned int i1 = blockIdx.x * blockDim.x + threadIdx.x + i1s;
	unsigned int i2 = blockIdx.y * blockDim.y + threadIdx.y + i2s;

	// save pairs
	uint2 idx = make_uint2(i1,i2);
	s_idx[tid] = idx;

	// test
	bool found = false;
	if (i1 < i1e && i2 < i2e) 
	{
		double2 radec1 = p_radec1[i1];
		double2 radec2 = p_radec2[i2];

		if (abs(radec1.x-radec2.x) < alpha_rad && abs(radec1.y-radec2.y) < sr_rad) 
		{
			double x,y,z, x2,y2,z2, dist2;
			radec2xyz(radec1,&x,&y,&z);
			radec2xyz(radec2,&x2,&y2,&z2);
			x -= x2;
			y -= y2;
			z -= z2;
			dist2 = x*x + y*y + z*z;
			found = (dist2 < sr_dist2 ? 1 : 0);
		}
	}
	s_found[tid] = found;

	// count matches in the block
	int count = __syncthreads_count(found);	

	// quit if none found
	if (count < 1) 
		return;

	// claim part of output vector
	if (tid == 0)
	{
		s_index = atomicAdd(m_end, count);
	}	
	__syncthreads(); // make sure it's ready for all
	unsigned int index = s_index;
	unsigned int n = THREADS_PER_BLOCK;

	// check allocated memory size
	if (index + count > m_size) 
		return;

	// check if need to fiddle before write
	if (count == n)
	{
		m_idx[index + tid] = idx;
		return;
	}

	// prefix scan for gaps
	int pout = 0, pin = 1;
	s_tmp[tid] = (s_found[tid] ? 0 : 1);
	__syncthreads();
	for (int offset = 1; offset < n; offset *= 2)
	{
		pout = 1 - pout; // swap double buffer indices
		pin = 1 - pin;
		unsigned short val = s_tmp[pin*n+tid];

		if (tid >= offset)
		{
			unsigned short inc = s_tmp[pin*n+tid - offset];
			s_tmp[pout*n+tid] = val + inc;
		}
		else
		{
			s_tmp[pout*n+tid] = val;
		}
		__syncthreads();
	}
	s_sum_gap[tid] = s_tmp[pout*n+tid]; // copy gaps
	__syncthreads();

	// prefix sum of founds
	pout = 0, pin = 1;
	s_tmp[pout*n + tid] = (tid < count || !s_found[tid]) ? 0 : 1;
	__syncthreads();
	for (int offset = 1; offset < n; offset *= 2)
	{
		pout = 1 - pout; // swap double buffer indices
		pin = 1 - pin;
		if (tid >= offset)
			s_tmp[pout*n+tid] = s_tmp[pin*n+tid] + s_tmp[pin*n+tid - offset];
		else
			s_tmp[pout*n+tid] = s_tmp[pin*n+tid];
		__syncthreads();
	}
	s_sum_fnd[tid] = s_tmp[pout*n+tid]; // copy but could be spared w/ sync
	__syncthreads();

	// find index of gaps
	int lo = -1;
	int hi = count-1;
	int mid;
	while (hi-lo > 1)
	{
		mid = (hi+lo)/2;
		if (tid+1 > s_sum_gap[mid])
			lo = mid;
		else
			hi = mid;
	}
	int idx_to = hi;

	// find index of remainder founds
	lo = count-1;
	hi = n-1;
	while (hi-lo > 1)
	{
		mid = (hi+lo)/2;
		if (tid+1 > s_sum_fnd[mid])
			lo = mid;
		else
			hi = mid;
	}
	int idx_from = hi;
	
	// number of gaps to fill
	int n_gaps = s_sum_gap[count-1];

	if (tid < n_gaps)
	{
		s_idx[idx_to] = s_idx[idx_from];
	}
	__syncthreads();
	
	// parallel global write
	if (tid < count)
	{
		m_idx[index + tid] = s_idx[tid];
	}

}

void load_n_sort( const char* f_obj, double zh_deg, std::ostream& out,
				  thrust::host_vector<long long>& h_id, 
				  thrust::host_vector<double2>& h_radec, 
				  thrust::host_vector<int>& h_zone_begin, 
				  thrust::host_vector<int>& h_zone_end  )
{
	thrust::device_vector<object> d_obj(h_id.size()); 
	thrust::host_vector<object> h_obj(d_obj.size());
	clock_t clk_start, clk_stop;

	// object* p_obj = thrust::raw_pointer_cast(&h_obj[0]); // debug

	// load objects
	clk_start = clock();
	load_objects_bin(h_obj, f_obj);
	clk_stop = clock();
	out << "[tmr] Load: " << std::fixed 
		<< (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl;	
	out.flush();

	// copy to gpu
	clk_start = clock();
	thrust::copy(h_obj.begin(), h_obj.end(), d_obj.begin()); 
	clk_stop = clock();
	out << "[tmr] Copy: " << std::fixed 
		<< (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl; 	
	out.flush();

	// sort objects
	clk_start = clock();
	thrust::sort(d_obj.begin(), d_obj.end(), less_zonera(zh_deg));
	clk_stop = clock();
	out << "[tmr] Sort: " << std::fixed 
		<< (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl;	
	out.flush();

	// zone limits on gpu
	thrust::device_vector<int> d_zone_begin(h_zone_begin.size());
	thrust::device_vector<int> d_zone_end(d_zone_begin.size());
	clk_start = clock();
	thrust::counting_iterator<size_t> search_begin(0); // used to produce integers in the range [0, n_zones)
	thrust::counting_iterator<size_t> search_end = search_begin + d_zone_begin.size();
	thrust::lower_bound( d_obj.begin(),d_obj.end(), search_begin,search_end, d_zone_begin.begin(), less_zone(zh_deg));
	thrust::upper_bound( d_obj.begin(),d_obj.end(), search_begin,search_end, d_zone_end.begin(), less_zone(zh_deg));
	clk_stop = clock();
	// copy to host
	thrust::copy(d_zone_begin.begin(), d_zone_begin.end(), h_zone_begin.begin());
	thrust::copy(d_zone_end.begin(), d_zone_end.end(), h_zone_end.begin());
	out << "[tmr] Lmts: "  << std::fixed
		<< (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl;
	out.flush();

	// copy back
	clk_start = clock();
	thrust::copy(d_obj.begin(), d_obj.end(), h_obj.begin());
	clk_stop = clock();
	out << "[tmr] Back: " << std::fixed
		<< (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl;
	out.flush();

	// split on host
	clk_start = clock();
	thrust::transform(h_obj.begin(), h_obj.end(), h_id.begin(), get_id());
	thrust::transform(h_obj.begin(), h_obj.end(), h_radec.begin(), get_radec_radian());
	clk_stop = clock();
	out << "[tmr] Splt: " << std::fixed
		<< (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl;
	out.flush();
}


void load_n_sort2( const char* f_obj, double zh_deg, std::ostream& out,
				  thrust::host_vector<long long>& h_id,
				  thrust::host_vector<double2>& h_radec,
				  thrust::host_vector<int>& h_zone_begin,
				  thrust::host_vector<int>& h_zone_end  )
{
	thrust::device_vector<object> d_obj(h_id.size());
	clock_t clk_start, clk_stop;

	// load to gpu
	{
		thrust::host_vector<object> h_obj(d_obj.size());
		// load objects
		clk_start = clock();
		load_objects_bin(h_obj, f_obj);
		clk_stop = clock();
		out << "[tmr] Load: " << std::fixed
			<< (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl;
		out.flush();
	
		// copy to gpu
		clk_start = clock();
		thrust::copy(h_obj.begin(), h_obj.end(), d_obj.begin());
		clk_stop = clock();
		out << "[tmr] Copy: " << std::fixed
			<< (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl;
		out.flush();
	}

	// sort objects
	clk_start = clock();
	thrust::sort(d_obj.begin(), d_obj.end(), less_zonera(zh_deg));
	clk_stop = clock();
	out << "[tmr] Sort: " << std::fixed
		<< (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl;
	out.flush();

	// alloc on device id and radec
	thrust::device_vector<long long> d_id(d_obj.size());
	thrust::device_vector<double2> d_radec(d_obj.size());

   	// split
	clk_start = clock();
	thrust::transform(d_obj.begin(), d_obj.end(), d_id.begin(), get_id());
	thrust::transform(d_obj.begin(), d_obj.end(), d_radec.begin(), get_radec_radian());
	clk_stop = clock();
	out << "[tmr] Splt: " << std::fixed
		<< (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl;
	out.flush();

	// copy back
	clk_start = clock();
	thrust::copy(d_id.begin(), d_id.end(), h_id.begin());
	thrust::copy(d_radec.begin(), d_radec.end(), h_radec.begin());
	clk_stop = clock();
	out << "[tmr] Back: " << std::fixed
		<< (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl;
	out.flush();

	// zone limits on gpu
	thrust::device_vector<int> d_zone_begin(h_zone_begin.size());
	thrust::device_vector<int> d_zone_end(d_zone_begin.size());
	clk_start = clock();
	thrust::counting_iterator<size_t> search_begin(0); // used to produce integers in the range [0, n_zones)
	thrust::counting_iterator<size_t> search_end = search_begin + d_zone_begin.size();
	thrust::lower_bound(d_obj.begin(),d_obj.end(), search_begin,search_end, d_zone_begin.begin(), less_zone(zh_deg));
	thrust::upper_bound(d_obj.begin(),d_obj.end(), search_begin,search_end, d_zone_end.begin(), less_zone(zh_deg));
	// copy limits to host
	thrust::copy(d_zone_begin.begin(), d_zone_begin.end(), h_zone_begin.begin());
	thrust::copy(d_zone_end.begin(), d_zone_end.end(), h_zone_end.begin());
	clk_stop = clock();
	out << "[tmr] Lmts: "  << std::fixed
		<< (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl;
	out.flush();

    /*
	// copy data to host
	clk_start = clock();
	thrust::copy(d_obj.begin(), d_obj.end(), h_obj.begin());
	clk_stop = clock();
	out << "[tmr] Back: " << std::fixed
		<< (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl;
	out.flush();
    */

    /*
	// split on host
	clk_start = clock();
	thrust::transform(h_obj.begin(), h_obj.end(), h_id.begin(), get_id());
	thrust::transform(h_obj.begin(), h_obj.end(), h_radec.begin(), get_radec_radian());
	clk_stop = clock();
	out << "[tmr] Splt: " << std::fixed
		<< (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl;
	out.flush();
   */
}


/*
	Usage
*/
void usage(int argc, char* argv[], const char* msg, int code)
{
	std::cerr << "Syntax:" << std::endl;
	std::cerr << "  " << argv[0] << " idradec1.bin n1 idradec2.bin n2 zh_arcsec sr_arcsec {x.bin} {nX}" << std::endl;
	if (msg != NULL)
		std::cerr << "ERROR: " << msg << std::endl;
	exit(code);
}


/* 
	Entry point 
*/
int main(int argc, char * argv[])
{
	// params
	if (argc < 7) { usage(argc, argv, NULL, 1); }
	int c = 1;
	// input file 1
	char * f1_obj = argv[c++];      
	int n1_obj = atoi(argv[c++]);
	// input file 2
	char * f2_obj = argv[c++];      
	int n2_obj = atoi(argv[c++]);
	// search
	double zh_arcsec = atof(argv[c++]); // zone height in "
	double sr_arcsec = atof(argv[c++]); // search radius
	// output
	char * f_out = NULL;
	int nX = 0;
	if (argc > c) f_out = argv[c++];
	if (argc > c) nX = atoi(argv[c++]);	

	// timers
	clock_t clk_start, clk_stop;
	
	// derivatives
	double zh_deg = zh_arcsec / 3600;
	double sr_deg = sr_arcsec / 3600;
	double sr_rad = sr_deg / RAD2DEG;
	double sr_dist2 = 2.0 * sin( sr_deg / 2.0 / RAD2DEG );
	sr_dist2 *= sr_dist2;
	// number of zones
	int n_zones = (int) ceil(180/zh_deg);
	std::cerr << "[dbg] Nzns: " << n_zones << std::endl;
	int nz = (int) ceil (sr_arcsec / zh_arcsec);
	std::cerr << "[dbg] dZns: " << nz << std::endl;
	
	// 1
	thrust::host_vector<long long> h1_id(n1_obj);
	thrust::host_vector<double2> h1_radec(n1_obj);
	thrust::host_vector<int> h1_zone_begin(n_zones);
	thrust::host_vector<int> h1_zone_end(n_zones);
	std::cerr << std::endl << "[dat] 1" << std::endl;
	load_n_sort2( f1_obj, zh_deg, std::cerr, h1_id, h1_radec, h1_zone_begin, h1_zone_end);

	// 2
	thrust::host_vector<long long> h2_id(n2_obj);
	thrust::host_vector<double2> h2_radec(n2_obj);
	thrust::host_vector<int> h2_zone_begin(n_zones);
	thrust::host_vector<int> h2_zone_end(n_zones);
	std::cerr << std::endl << "[dat] 2" << std::endl;
	load_n_sort2( f2_obj, zh_deg, std::cerr, h2_id, h2_radec, h2_zone_begin, h2_zone_end);	

	// copy to gpu
	std::cerr << std::endl;
	clk_start = clock();
	thrust::device_vector<double2> d1_radec = h1_radec;
	thrust::device_vector<double2> d2_radec = h2_radec;
	clk_stop = clock();
	std::cerr << "[tmr] Cop2: " << std::fixed 
		<< (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl; 	

	// xmatch
	unsigned int n_match_alloc = nX;
	thrust::host_vector<uint2> h_match_idx(n_match_alloc);
	double2* p1_radec = thrust::raw_pointer_cast(&d1_radec[0]);
	double2* p2_radec = thrust::raw_pointer_cast(&d2_radec[0]);
	// alloc
	thrust::device_vector<unsigned int> d_match_num(1);
	thrust::device_vector<uint2> d_match_idx(h_match_idx.size());
	// ptr
	unsigned int * p_match_num =  thrust::raw_pointer_cast(&d_match_num[0]);
	uint2 * p_match_idx =  thrust::raw_pointer_cast(&d_match_idx[0]);

	// loop
	clk_start = clock();
	for (int zid1 = 0; zid1 < n_zones; zid1++)
	{
		int i1s = h1_zone_begin[zid1];
		int i1e = h1_zone_end[zid1];
		int n1 = i1e - i1s;

		if (n1 < 1) continue;

		double alpha_rad = calc_alpha(sr_deg, zh_deg, zid1); // in radians

		int zid2start = max(0,zid1-nz);
		int zid2end = min(n_zones-1,zid1+nz);

		for (int zid2=zid2start; zid2<=zid2end; zid2++)
		{
			int i2s = h2_zone_begin[zid2];
			int i2e = h2_zone_end[zid2];
			int n2 = i2e - i2s;

			if (n2 < 1) continue;

			dim3 dimBlock(16, THREADS_PER_BLOCK / 16);
			dim3 dimGrid( (n1+dimBlock.x-1) / dimBlock.x, 
						  (n2+dimBlock.y-1) / dimBlock.y );  				
			// launch
			xmatch_kernel<<<dimGrid,dimBlock>>>( 
				p1_radec, i1s, i1e,  
				p2_radec, i2s, i2e, 
				sr_rad, alpha_rad, sr_dist2, 
				p_match_idx, n_match_alloc, p_match_num);
		}
	}
	cudaThreadSynchronize();	
	clk_stop = clock();

	// fetch number of matches from gpu
	unsigned int match_num = d_match_num[0];

	std::cerr << "[tmr] Mtch: " 
		<< (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl;
	
	clk_start = clock();
	thrust::copy(d_match_idx.begin(), d_match_idx.end(), h_match_idx.begin());
	clk_stop = clock();
	std::cerr << "[tmr] Ftch: " << std::fixed 
		<< (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl; 	

	std::cerr << std::endl;		
	std::cerr << "[sum] Numb: " << match_num << std::endl;
	std::cerr << std::endl;
	// top 10
	{
		int n_out = min(10, n_match_alloc);
		std::cerr << "[inf] Top " << n_out << std::endl;
		for (int i=0; i<n_out; i++)
		{
			std::cerr << h1_id[h_match_idx[i].x] << " " << h2_id[h_match_idx[i].y] << std::endl;
		}
	}
	std::cerr << std::endl;
	
	// write binary
	if (f_out != NULL && nX > 0)
	{
		std::ofstream outfile (f_out, std::ios::out | std::ios::binary);
		clk_start = clock();
		if (outfile.is_open())
		{
			int n_out = min(n_match_alloc, match_num);
			// outfile.write( (char*) p_match_iden, n_out  * sizeof(longlong2) );
			for (int i=0; i<n_out; i++)
			{
				outfile.write( (char*) &h1_id[h_match_idx[i].x], sizeof(long long) );
				outfile.write( (char*) &h2_id[h_match_idx[i].y], sizeof(long long) );
			}
		}
	    outfile.close();
		clk_stop = clock();

		std::cerr << "[tmr] Dump: " << std::fixed
			<< (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl;

		if (n_match_alloc < match_num)
		{
			std::cerr << std::endl;
			std::cerr << "[wrn] Truncated output!" << std::endl;
		}
	}

	return 0;
}
