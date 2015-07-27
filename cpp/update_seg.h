#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "my_sp_struct.h"
#endif

__host__ void CudaFindBorderPixels( const int* seg, bool* border, const int nPixels, const int xdim, const int ydim, const int single_border);
__global__  void find_border_pixels( const int* seg, bool* border, const int nPixels, const int xdim, const int ydim, const int single_border);

__host__ void update_seg(double* img, int* seg, bool* border, 
						superpixel_params* sp_params, 
						const double3 J_i, const double logdet_Sigma_i, 
						bool cal_cov, int i_std, int s_std,int nInnerIters,
						const int nPixels, const int nSPs, int xdim, int ydim);

__global__  void update_seg_subset(
    double* img, int* seg, bool* border,
    superpixel_params* sp_params, 
    const double3 J_i, const double logdet_Sigma_i,  
    bool cal_cov, int i_std, int s_std,
    const int nPts,const int nSuperpixels,
    const int xdim, const int ydim,
    const int xmod3, const int ymod3);

