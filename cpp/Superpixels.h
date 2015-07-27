#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
 #include "optionparser.h"

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "my_sp_struct.h"
#endif

 
#include "RgbLab.h"
#include "init_seg.h"
#include "sp_helper.h"
#include "update_param.h"
#include "update_seg.h"


using namespace cv;
using namespace std;


class Superpixels {
    int dim_i, dim_s;
    int dim_x, dim_y;
    int nPixels_in_square_side;
    int nSPs, nPixels;
    int nInnerIters;
    bool init_sp;
    superpixel_options sp_options;
    
    superpixel_params* sp_params;
    superpixel_params* sp_params_cpu;
    superpixel_GPU_helper* sp_gpu_helper;
    
    // since we fix covariance for color component
    double3 J_i; //fixed
    double logdet_Sigma_i; //fixed

    unsigned char* image_cpu;
    uchar3* image_gpu;
    double* image_gpu_double;

    bool* border_cpu;
    bool* border_gpu;
    int* seg_cpu;
    int* seg_gpu;


  public:
    Superpixels(int img_dimx, int img_dimy, superpixel_options spoptions);
    ~Superpixels();
    void load_img(unsigned char* imgP);
    void calc_seg();
    void gpu2cpu();
    Mat get_img_overlaid();
    Mat get_img_cartoon();
    void same_size_reinit();
};

