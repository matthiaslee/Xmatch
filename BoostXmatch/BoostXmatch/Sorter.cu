#include "Sorter.h"
#include "Obj.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#pragma warning(push)
//#pragma warning(disable: 4996)      // Thrust's use of strerror
//#pragma warning(disable: 4251)      // STL class exports
//#pragma warning(disable: 4005)      // BOOST_COMPILER macro redefinition
#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#pragma warning(pop)



#define RAD2DEG 57.295779513082323
#define THREADS_PER_BLOCK 512

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
		int64_t operator()(Obj o) const { return o.mId; }
	};

	// functor for splitting object list
	struct get_radec_radian
	{
		__host__ __device__
		double2 operator()(Obj o) const
		{
			double2 r;
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
		double zh_deg = zh_arcsec / 3600;

		thrust::device_vector<Obj> dObj(seg->vObj.size());
		thrust::device_vector<int64_t> dId(seg->vObj.size());
		thrust::device_vector<double2> dRadec(seg->vObj.size());

		//std::cout << dRadec.size() << std::endl;

		// dObj.resize((thrust::device_vector<Obj>::size_type)(seg->mNum));

		// move data to gpu
		{
			thrust::copy(seg->vObj.begin(), seg->vObj.end(), dObj.begin());
		}
		// sort objects
		{
			thrust::sort(dObj.begin(), dObj.end(), less_zonera(zh_deg));
		}

		int devid;
		cudaDeviceProp devProp;

		cudaGetDevice(&devid);
        cudaGetDeviceProperties(&devProp, devid);
        //printDevProp(devProp);
		std::cout << "Sort(): CUDA Device # " << devid << std::endl;

		std::cout << seg->vObj[0].mId << std::endl;


		thrust::copy(dObj.begin(), dObj.end(), seg->vObj.begin());



		// split
		{
			//dId.resize(dObj.size());
			thrust::transform(dObj.begin(), dObj.end(), dId.begin(), get_id());
		
			//dRadec.resize(dObj.size());
			thrust::transform(dObj.begin(), dObj.end(), dRadec.begin(), get_radec_radian());
		}		


		// move back -- could be async
		{
			seg->vId = dId;
			seg->vRadec = dRadec;
		}


		std::cout << "there" << std::endl;
		#ifdef THERE


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
			//seg->vZoneBegin.resize(d_zone_begin.size());
			//seg->vZoneEnd.resize(d_zone_end.size());
			//thrust::copy(d_zone_begin.begin(), d_zone_begin.end(), seg->vZoneBegin.begin());
			//thrust::copy(d_zone_end.begin(), d_zone_end.end(), seg->vZoneEnd.begin());
		}
#endif
	}

	// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %u\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %u\n",  devProp.totalConstMem);
    printf("Texture alignment:             %u\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}

	void Sorter::operator()()
	{   
		cudaError_t err = cudaSetDevice(this->id);
		if (err) 
		{
			std::cerr << "Cannot set CUDA device " << this->id << std::endl;
			return;
		}

		bool keepProcessing = true;

		try  
		{
			while(keepProcessing)  
			{  
				SegmentPtr seg = segman->Next();

				if (seg == NULL) 
				{
					keepProcessing = false;
				}
				else
				{
					Sort(seg);
				}
			}
		}  
		// Catch specific exceptions first 
		// ...
		// Catch general so it doesn't go unnoticed
		catch (std::exception& exc)  {  std::cerr << exc.what() << std::endl;	}  
		catch (...)  {  std::cerr << "Unknown error!" << std::endl;	}  

		// reset
		{
			// cudaThreadExit() is now deprecated
			err = cudaDeviceReset();
			if (err)
			{
				std::cerr << "Cannot reset device" << std::endl;
			}
		}


	}



}
