#include "Worker.h"
#include "Log.h"

#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#pragma warning(push)
#pragma warning(disable: 4996)      // Thrust's use of strerror
#pragma warning(disable: 4251)      // STL class exports
#pragma warning(disable: 4005)      // BOOST_COMPILER macro redefinition
//#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#pragma warning(pop)


#define THREADS_PER_BLOCK 512


namespace xmatch
{

	#ifndef FASTMATH
	// convert ra,dec to xyz
	__host__ __device__
	void radec2xyz(dbl2 radec, double *x, double *y, double *z)
	{
		double sr, cr, sd, cd;
		sincos(radec.x, &sr, &cr);
		sincos(radec.y, &sd, &cd);
		*x = cd * cr;
		*y = cd * sr;
		*z = sd;
	}
	#else
	// convert ra,dec to xyz (gpu w/ single point and less precise "fast math")
	__device__
	void radec2xyz(dbl2 radec, float *x, float *y, float *z)
	{
		float sr, cr, sd, cd;
		__sincosf(radec.x, &sr, &cr);
		__sincosf(radec.y, &sd, &cd);
		*x = cd * cr;
		*y = cd * sr;
		*z = sd;
	}
	#endif

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
	void xmatch_kernel(	const dbl2* p_radec1, int i1s, int i1e, 
						const dbl2* p_radec2, int i2s, int i2e,
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
			dbl2 radec1 = p_radec1[i1];
			dbl2 radec2 = p_radec2[i2];

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

	void Worker::Match(JobPtr job, std::ofstream& outfile)
	{
		LOG_TIM << "- GPU-" << id << " " << *job << " copying to device" << std::endl;
		// copy to gpu  -- shd look 1st if already there...
		thrust::device_vector<dbl2> d1_radec = job->segA->vRadec;
		thrust::device_vector<dbl2> d2_radec = job->segB->vRadec;
	
		// xmatch alloc limit -- hack for now...
		unsigned int n_match_alloc = 2 * std::max (d1_radec.size(), d2_radec.size());
		thrust::device_vector<uint2> d_match_idx(n_match_alloc);
		thrust::device_vector<unsigned int> d_match_num(1);

		// pointers
		dbl2* p1_radec = thrust::raw_pointer_cast(&d1_radec[0]);
		dbl2* p2_radec = thrust::raw_pointer_cast(&d2_radec[0]);
		unsigned int* p_match_num = thrust::raw_pointer_cast(&d_match_num[0]);
		uint2* p_match_idx = thrust::raw_pointer_cast(&d_match_idx[0]);

		int n_zones = job->segA->vZoneBegin.size();
		double sr_deg = job->sr_deg;
		double zh_deg = job->segA->mZoneHeightDegree;
		int nz = (int) ceil (sr_deg / zh_deg);

		double sr_rad = sr_deg / RAD2DEG;
		double sr_dist2 = 2.0 * sin( sr_rad / 2.0 );
		sr_dist2 *= sr_dist2;

		LOG_TIM << "- GPU-" << id << " " << *job << " kernel launches" << std::endl;

		// loop
		for (int zid1 = 0; zid1 < n_zones; zid1++)
		{
			int i1s = job->segA->vZoneBegin[zid1];
			int i1e = job->segA->vZoneEnd[zid1];
			int n1 = i1e - i1s;

			if (n1 < 1) continue;

			double alpha_rad = calc_alpha(sr_deg, zh_deg, zid1); // in radians

			// deal with RA wraparound here or after the kernel launch

			int zid2start = std::max(0,zid1-nz);
			int zid2end = std::min(n_zones-1,zid1+nz);

			for (int zid2=zid2start; zid2<=zid2end; zid2++)
			{
				int i2s = job->segB->vZoneBegin[zid2];
				int i2e = job->segB->vZoneEnd[zid2];
				int n2 = i2e - i2s;

				if (n2 < 1) continue;

				dim3 dimBlock(16, THREADS_PER_BLOCK / 16);
				dim3 dimGrid( (n1+dimBlock.x-1) / dimBlock.x, 
							  (n2+dimBlock.y-1) / dimBlock.y );  				
				// launch
				xmatch_kernel <<<dimGrid,dimBlock>>> ( p1_radec, i1s, i1e,  p2_radec, i2s, i2e, 
					sr_rad, alpha_rad, sr_dist2,  p_match_idx, n_match_alloc, p_match_num);
			}
			// could cuda-sync here and dump (smaller) result sets on the fly...
		}
		//cudaThreadSynchronize();
		cudaDeviceSynchronize();
		//if (err != cudaSuccess) LOG_ERR  << "- GPU-" << id << " " << *job << " !! Cannnot sync !!" << std::endl; 

		// fetch number of matches from gpu
		unsigned int match_num = d_match_num[0];
		LOG_TIM << "- GPU-" << id << " " << *job << " # " << match_num << std::endl;

		// copy indices to host
		thrust::host_vector<uint2> h_match_idx = d_match_idx;

		// check if all matches fit in mem - hack guard
		if (n_match_alloc < match_num) 
		{
			LOG_ERR << "- GPU-" << id << " " << *job << " !! Truncated output !!" << std::endl;
		}

		// dump 
		{
			int n_out = std::min(n_match_alloc, match_num);
			// debug to screen: top 10
			if (true)
			{
				int n_top = std::min(n_out,10);
				for (int i=0; i<n_top; i++)
				{
					uint2 idx = h_match_idx[i];
					LOG_DBG2 << " \t" 
						<< job->segA->vId[idx.x] << " "
						<< job->segB->vId[idx.y] << std::endl;
				}
			}
			// write binary
			if (outfile.is_open())
			{
				int64_t objid;
				for (int i=0; i<n_out; i++)
				{
					uint2 idx = h_match_idx[i];
					objid = job->segA->vId[idx.x];
					outfile.write((char*)&objid, sizeof(int64_t));
					objid = job->segB->vId[idx.y];
					outfile.write((char*)&objid, sizeof(int64_t));
				}
				//outfile.flush();
			}
		}
		LOG_PRG << "- GPU-" << id << " " << *job << " done" << std::endl;
	}

	void Worker::operator()()
	{   
		cudaError_t err = cuman->SetDevice(this->id);
		if (err!=cudaSuccess) 
		{ 
			LOG_ERR << "- GPU-" << id << " !! Cannot set CUDA device !!" << std::endl; 
			return; 
		}

		std::ofstream outfile;
		try  
		{
			// open output file
			if (!outpath.empty()) 
			{
				LOG_DBG << "- GPU-" << id << " output to file " << outpath << std::endl; 
				outfile.open(outpath, std::ios::out | std::ios::binary);
				if (!outfile.is_open())
				{
					LOG_ERR << "- GPU-" << id << " !! Cannot open output file !!" << std::endl;
				}
				else
				{
					bool   keepProcessing = true;
					while (keepProcessing)  
					{  
						JobPtr job = jobman->NextPreferredJob(oldjob);
						if (job == NULL) keepProcessing = false;
						else
						{
							Match(job,outfile);
							jobman->SetStatus(job,FINISHED);
							oldjob = job;
						}				
					}  
				} 
			} /* endif */

		}
		// Catch specific exceptions first 
		// ...
		// Catch general so it doesn't go unnoticed
		catch (std::exception& exc)  
		{  
			LOG_ERR << "- GPU-" << id << " !! Error !!" << std::endl
				        << exc.what() << std::endl;	
		}  
		catch (...)  
		{  
			LOG_ERR << "- GPU-" << id << " !! Unknown error !!" << std::endl;	
		}  
		// reset device
		{
			cudaError_t err = cuman->Reset(); 
			if (err!=cudaSuccess) 
				LOG_ERR << "- GPU-" << id << " !! Cannot reset CUDA device !!" << std::endl;					
		}

		outfile.close();
	}
}