#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct superpixel_params
{
    double3 mu_i;
    double3 sigma_s;
    double2 mu_s;
    double logdet_Sigma_s;
    int count;
    double log_count;
};

struct superpixel_GPU_helper{
    int3 mu_i_sum;  // with respect to nSps
    int2 mu_s_sum;
    longlong3 sigma_s_sum;
};

struct superpixel_options{
    int nPixels_in_square_side,area; 
    int i_std, s_std, prior_count;
    bool permute_seg, calc_cov, use_hex;
    int prior_sigma_s_sum;
    int nEMIters, nInnerIters;
};