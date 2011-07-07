#include "Sorter.h"
#include "Log.h"

#pragma warning(push)
#pragma warning(disable: 4996)      // Thrust's use of strerror
#pragma warning(disable: 4251)      // STL class exports
//#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#pragma warning(pop)


namespace xmatch
{		
	__host__ __device__
	int32_t GetZoneId(double dec, double height)  
	{ 
		return (int32_t) floor( (dec + 90) / height ); 
	}

	// functor for sorting objects
	struct less_zonera
	{
		double h;

		__host__ __device__ 
		less_zonera(double height) : h(height) {}

		__host__ __device__ 
		bool operator()(const Obj& lhs, const Obj& rhs) const
		{
			int lz = GetZoneId(lhs.mDec, h);
			int rz = GetZoneId(rhs.mDec, h);
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
		dbl2 operator()(const Obj& o) const
		{
			dbl2 r;
			r.x = o.mRa/RAD2DEG;
			r.y = o.mDec/RAD2DEG;
			return r;
		}
	};

	// functor for finding zone boundaries
	struct less_zone
	{
		double h;

		__host__ __device__
		less_zone(double height) : h(height) {}

		__host__ __device__ 
		bool operator()(const Obj& lhs, const Obj& rhs) const
		{
			int lz = GetZoneId(lhs.mDec, h);
			int rz = GetZoneId(rhs.mDec, h);
			return lz < rz;
		}

		__host__ __device__ 
		bool operator()(const Obj& o, int zone) const
		{
			int z = GetZoneId(o.mDec,h);
			return z < zone;
		}

		__host__ __device__ 
		bool operator()(int zone, const Obj& o) const
		{
			int z = GetZoneId(o.mDec,h);
			return zone < z;
		}

	}; 

	void Sorter::Sort(SegmentPtr seg) 
	{
		LOG_TIM << "- GPU-" << id << " sorting" << std::endl;
		thrust::device_vector<Obj> dObj(seg->vObj.size());
		thrust::device_vector<int64_t> dId(seg->vObj.size());
		thrust::device_vector<dbl2> dRadec(seg->vObj.size());

		// copy and sort
		thrust::copy(seg->vObj.begin(), seg->vObj.end(), dObj.begin());
		seg->vObj.resize(0);
		thrust::sort(dObj.begin(), dObj.end(), less_zonera(zh_deg));
		
		LOG_TIM << "- GPU-" << id << " splitting" << std::endl;
		// split
		thrust::transform(dObj.begin(), dObj.end(), dId.begin(), get_id());
		thrust::transform(dObj.begin(), dObj.end(), dRadec.begin(), get_radec_radian());

		LOG_TIM << "- GPU-" << id << " copying to host" << std::endl;
		// copy back
		seg->vId = dId;
		seg->vRadec = dRadec;
		seg->mZoneHeightDegree = zh_deg;

		LOG_DBG2 << "- GPU-" << id << " zone boundaries" << std::endl;
		// zone limits on gpu
		{
			int n_zones = (int) ceil(180/zh_deg);
			thrust::device_vector<int> d_zone_begin(n_zones);
			thrust::device_vector<int> d_zone_end(n_zones);
			// searches
			thrust::counting_iterator<size_t> search_begin(0); // used to produce integers in the range [0, n_zones)
			thrust::counting_iterator<size_t> search_end = search_begin + n_zones;
			thrust::lower_bound(dObj.begin(),dObj.end(), search_begin,search_end, d_zone_begin.begin(), less_zone(zh_deg));
			thrust::upper_bound(dObj.begin(),dObj.end(), search_begin,search_end, d_zone_end.begin(), less_zone(zh_deg));
			// copy limits to host
			seg->vZoneBegin = d_zone_begin;
			seg->vZoneEnd = d_zone_end;
		}
		LOG_TIM << "- GPU-" << id << " done" << std::endl;
	}

	void Sorter::operator()()
	{   
		try  
		{
			cudaError_t err = cuman->SetDevice(this->id);
			if (err!=cudaSuccess) { LOG_ERR << "Cannot set CUDA device on GPU-" << this->id << std::endl; return;	}

			bool   keepProcessing = true;
			while (keepProcessing)  
			{  
				SegmentPtr seg = segman->Next();
				if (seg == NULL) keepProcessing = false;
				else 			 Sort(seg);
			}
		}  
		// Catch specific exceptions first 
		// ...
		// Catch general so it doesn't go unnoticed
		catch (std::exception& exc)  {  LOG_ERR << exc.what() << std::endl;	}  
		catch (...)  {  LOG_ERR << "Unknown error!" << std::endl;	}  
		// reset device
		{
			cudaError_t err = cuman->Reset(); 
			if (err!=cudaSuccess) LOG_ERR << "Cannot reset CUDA device on GPU-" << this->id << std::endl;			
		}

	} // operator()

} // xmatch
