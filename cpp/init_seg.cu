#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <float.h>
using namespace std;

#define THREADS_PER_BLOCK 512

#include "init_seg.h"

#include <stdio.h>

// b = 5; 1-5 ->1, 6-10 ->2, 11-15 ->3
__host__ int myCeil(int a, int b){
    if (a%b==0) return a/b;
    else return ceil(double(a)/double(b));
}

__host__ bool saveArray( int* pdata, size_t length, const std::string& file_path )
{
    std::ofstream os(file_path.c_str(), std::ios::binary | std::ios::out);
    if ( !os.is_open() )
        return false;
    os.write(reinterpret_cast<const char*>(pdata), std::streamsize(length*sizeof(int)));
    os.close();
    return true;
}

__host__ bool loadArray( int* pdata, size_t length, const std::string& file_path)
{
    std::ifstream is(file_path.c_str(), std::ios::binary | std::ios::in);
    if ( !is.is_open() )
        return false;
    is.read(reinterpret_cast<char*>(pdata), std::streamsize(length*sizeof(int)));
    is.close();
    return true;
}



__host__ int CudaInitSeg(int* seg_cpu, int* seg_gpu, int nPts,int sz, int xdim, int ydim, bool use_hex){	
  	dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);

    int num_block_pixel = ceil(double(nPts+1) / double(THREADS_PER_BLOCK));
    dim3 BlockPerGrid_pixel(num_block_pixel,1);

	if (!use_hex){
        InitSquareSeg<<<BlockPerGrid_pixel,ThreadPerBlock>>>(seg_gpu,nPts,sz, xdim, ydim);
        cudaMemcpy(seg_cpu, seg_gpu, nPts*sizeof(int), cudaMemcpyDeviceToHost);

	}else{
        std::stringstream xdim_str, ydim_str, sz_str;
        xdim_str << xdim;
        ydim_str << ydim;
        sz_str << sz;     
        std::string file_path = xdim_str.str() + "_" + ydim_str.str() + "_" + sz_str.str() + ".bin";

        // length of each side   
        double H = sqrt( double(pow(sz, 2)) / (1.5 *sqrt(3.0)) );
        double w = sqrt(3.0) * H;
        //printf("%1f \n", H);
        //printf("%1f \n", w);

        //calculate how many hexagons are on x and y axis
 
        int max_num_sp_x = (int) floor(double(xdim)/w) + 1;
        int max_num_sp_y = (int) floor(double(ydim)/(1.5*H)) + 1;
        int max_nSPs = max_num_sp_x * max_num_sp_y;

        //printf("%d \n", max_num_sp_x);
       // printf("%d \n", max_num_sp_y);
       // printf("%d \n", max_nSPs);

        if (loadArray(seg_cpu, nPts, file_path)){
            cudaMemcpy(seg_gpu,seg_cpu,nPts*sizeof(int),cudaMemcpyHostToDevice);
        }else{

            int num_block_sp =  ceil(double(max_nSPs) /double(THREADS_PER_BLOCK));
            dim3 BlockPerGrid_sp(num_block_sp,1);

            double* centers;
            cudaMalloc((void**) &centers, 2*max_nSPs*sizeof(double));
            InitHexCenter<<<BlockPerGrid_sp,ThreadPerBlock>>>(centers, H, w, max_nSPs, max_num_sp_x, xdim, ydim); 
            cudaDeviceSynchronize();

            InitHexSeg<<<BlockPerGrid_pixel,ThreadPerBlock>>>(seg_gpu, centers, max_nSPs, nPts, xdim);
            cudaDeviceSynchronize();

            //write the seg_cpu to file     
            cudaMemcpy(seg_cpu, seg_gpu, nPts*sizeof(int), cudaMemcpyDeviceToHost);
            saveArray(seg_cpu, nPts, file_path);
            cudaFree(centers);

        }
        
	}
    int nSPs = get_max(seg_cpu, nPts)+1;
	return nSPs;
}




__global__ void InitHexCenter(double* centers, double H, double w, int max_nPts, int max_num_sp_x, int xdim, int ydim){
	//int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * num_sp_x;
	//int idx = offsetBlock + threadIdx.x + threadIdx.y * num_sp_x;
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	if (idx >= max_nPts) return;

    int x = idx % max_num_sp_x; 
    int y = idx / max_num_sp_x; 

    double xx = double(x) * w;
    double yy = double(y) * 1.5 *H; 
    
    if (y%2 == 0){
        xx = xx + 0.5*w;
    }
    
    centers[2*idx]  = xx;
    centers[2*idx+1]  = yy;    
}




__global__ void InitHexSeg(int* seg, double* centers, int K, int nPts, int xdim){
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 	
	if (idx >= nPts) return;

    int x = idx % xdim;
    int y = idx / xdim;   

    double dx,dy,d2;
    double D2 = DBL_MAX; 
    for (int j=0; j < K;  j++){
        dx = (x - centers[j*2+0]);
        dy = (y - centers[j*2+1]);
        d2 = dx*dx + dy*dy;
        if ( d2 <= D2){
              D2 = d2;  
              seg[idx]=j;
        }           
    } 
    return;	
}



// for everypixel, assign it to a superptxel
__global__ void  InitSquareSeg(int* seg, int nPts, int sz, int xdim, int ydim){
	int t = threadIdx.x + blockIdx.x * blockDim.x; 
	if (t>=nPts) return;
	
    double side = double(sz);
	//how many superpixels per col
	int sp_y = (ydim%sz == 0)? ydim/sz : ( (int)floor(ydim/side) + 1 );

	int x = t % xdim;  
	int y =  t / xdim;

	int i = (x%sz==0)? x/sz: ((int) floor(x/side)); // which col
    int j = (y%sz==0)? y/sz: ((int) floor(y/side)); //which row

	seg[t] = j + i*sp_y;  
}


/* find the maximum value in the seg_arr */
__host__ int get_max(int* seg, int nPts){
    int max_val = 0;
    for (int i = 0; i<nPts; i++){
        if (seg[i]>max_val){
            max_val = seg[i];
        }
    }
    return max_val;
}