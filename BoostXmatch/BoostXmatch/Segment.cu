/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#include "Segment.h"

#include <sstream>
#include <ctime>

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

namespace xmatch
{
	void Segment::Load(std::istream &rIs)
	{
		if (mNum > 0)
		{
			hObj.resize(mNum);
			Obj *o = thrust::raw_pointer_cast(&hObj[0]);
			rIs.read( (char*)o, mNum * sizeof(Obj));
		}
	}

	// functor for sorting objects
	__host__ __device__
	struct less_zonera
	{
		double h;

		__host__ __device__
		less_zonera(double height) : h(height) {}

		__host__ __device__ 
		bool operator()(const Obj& lhs, const Obj& rhs) const
		{
			int lz = Obj::GetZoneId(lhs.mDec, h);
			int rz = Obj::GetZoneId(rhs.mDec, h);
			if (lz < rz) return true;
			if (lz > rz) return false;
			return lhs.mRa < rhs.mRa;
		}
	}; 

	struct get_id
	{
		__host__ __device__
		int64_t operator()(const Obj& o) const { return o.mId; }
	};

	// functor for splitting object list
	struct get_radec_radian
	{
		__host__ __device__
		double2 operator()(const Obj& o) const
		{
			return make_double2(o.mRa/RAD2DEG, o.mDec/RAD2DEG);
		}
	};

	// functor for finding zone boundaries
	__host__ __device__
	struct less_zone
	{
		double h;

		__host__ __device__
		less_zone(double height) : h(height) {}

		__host__ __device__ 
		bool operator()(const Obj& lhs, const Obj& rhs) const
		{
			int lz = Obj::GetZoneId(lhs.mDec,h);
			int rz = Obj::GetZoneId(rhs.mDec,h);
			return lz < rz;
		}

		__host__ __device__ 
		bool operator()(const Obj& o, int zone) const
		{
			int z = Obj::GetZoneId(o.mDec,h);
			return z < zone;
		}

		__host__ __device__ 
		bool operator()(int zone, const Obj& o) const
		{
			int z = Obj::GetZoneId(o.mDec,h);
			return zone < z;
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


	void Segment::Sort(double zh_arcsec)
	{
		zh_deg = zh_arcsec / 3600;
		clock_t clk_start, clk_stop;

		thrust::device_vector<Obj> dObj(hObj.size());
		// copy to gpu
		{
			clk_start = clock();
			thrust::copy(hObj.begin(), hObj.end(), dObj.begin());
			clk_stop = clock();
			// out << "[tmr] Copy: " << std::fixed	<< (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl;
			// out.flush();
		}

		// sort objects
		{
			clk_start = clock();
			thrust::sort(dObj.begin(), dObj.end(), less_zonera(zh_deg));
			clk_stop = clock();
			// out << "[tmr] Sort: " << std::fixed	<< (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl;
			//out.flush();
		}
		// alloc on device id and radec
		thrust::device_vector<int64_t> dId(dObj.size());
		thrust::device_vector<double2> dRaDec(dObj.size());

   		// split
		{
			clk_start = clock();
			thrust::transform(dObj.begin(), dObj.end(), dId.begin(), get_id());
			thrust::transform(dObj.begin(), dObj.end(), dRaDec.begin(), get_radec_radian());
			clk_stop = clock();
			// out << "[tmr] Splt: " << std::fixed	<< (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl;
			//out.flush();
		}
		
		hId.resize(dId.size());
		hRaDec.resize(dRaDec.size());
		// copy back -- could be async
		{
			clk_start = clock();
			thrust::copy(dId.begin(), dId.end(), hId.begin());
			thrust::copy(dRaDec.begin(), dRaDec.end(), hRaDec.begin());
			clk_stop = clock();
			// out << "[tmr] Back: " << std::fixed	<< (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl;
			// out.flush();
		}

		// zone limits on gpu
		{
			int n_zones = (int) ceil(180/zh_deg);
			thrust::device_vector<int> d_zone_begin(n_zones);
			thrust::device_vector<int> d_zone_end(d_zone_begin.size());
			hZoneBegin.resize(d_zone_begin.size());
			hZoneEnd.resize(d_zone_end.size());
			clk_start = clock();
			thrust::counting_iterator<size_t> search_begin(0); // used to produce integers in the range [0, n_zones)
			thrust::counting_iterator<size_t> search_end = search_begin + d_zone_begin.size();
			thrust::lower_bound(dObj.begin(),dObj.end(), search_begin,search_end, d_zone_begin.begin(), less_zone(zh_deg));
			thrust::upper_bound(dObj.begin(),dObj.end(), search_begin,search_end, d_zone_end.begin(), less_zone(zh_deg));
			// copy limits to host
			thrust::copy(d_zone_begin.begin(), d_zone_begin.end(), hZoneBegin.begin());
			thrust::copy(d_zone_end.begin(), d_zone_end.end(), hZoneEnd.begin());
			clk_stop = clock();
			// out << "[tmr] Lmts: "  << std::fixed << (double)(clk_stop - clk_start) / ((double)CLOCKS_PER_SEC) << std::endl;
			// out.flush();
		}
	}

	void Segment::Match(Segment const &rSegment, double sr_arcsec) const
	{
	}

	std::string Segment::ToString(const std::string sep) const
	{
		std::stringstream ss;
		ss << mId; // << sep << sorted;
		return ss.str();
	}
	
	std::string Segment::ToString() const
	{
		return ToString(std::string(":"));
	}

	std::ostream& operator<<(std::ostream &o, const Segment &s)
	{
		o << s.ToString();
		return o;
	}
}