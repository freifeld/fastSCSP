

#include "Superpixels.h"


#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <sstream>

void throw_on_cuda_error(cudaError_t code)
{
  if(code != cudaSuccess){
    throw thrust::system_error(code, thrust::cuda_category());
  }
}


// constructor
// init the superpixels with dim_x, dim_y, dim_i and options
Superpixels::Superpixels(int img_dimx, int img_dimy, superpixel_options spoptions){
    init_sp = false;
    dim_x = img_dimx;
    dim_y = img_dimy;
    nPixels = dim_x * dim_y;

    dim_i = 3; // RGB/BGR/LAB
    dim_s = 2;

    sp_options = spoptions;
    int i_std  = sp_options.i_std;
    double half_i_std_square = double(i_std/2) * double(i_std/2); 
    double i_std_square = double(i_std) * double(i_std); 
    
    logdet_Sigma_i = log(half_i_std_square * i_std_square * i_std_square);
            
    J_i.x = 1.0/half_i_std_square;     
    J_i.y = 1.0/i_std_square; 
    J_i.z = 1.0/i_std_square; 

    //allocate memory for the cpu variables: image_cpu, seg_cpu and border_cpu
    const int sizeofint = sizeof(int);
    const int sizeofbool = sizeof(bool);
    const int sizeofuchar = sizeof(unsigned char);
    const int sizeofd = sizeof(double);
    const int sizeofuchar3 = sizeof(uchar3); 

    image_cpu = (unsigned char*) malloc(dim_i*nPixels*sizeofuchar);
    seg_cpu = (int*) malloc(nPixels*sizeofint);
    border_cpu = (bool*) malloc(nPixels * sizeofbool);

     // allocate memory for the cuda variables
    try{
        throw_on_cuda_error( cudaMalloc((void**) &image_gpu, nPixels*sizeofuchar3));
        throw_on_cuda_error( cudaMalloc((void**) &image_gpu_double, dim_i*nPixels*sizeofd));
        throw_on_cuda_error( cudaMalloc((void**) &seg_gpu, nPixels * sizeofint));
        throw_on_cuda_error( cudaMalloc((void**) &border_gpu, nPixels*sizeofbool));
    }
    catch(thrust::system_error &e){
        std::cerr << "CUDA error after cudaMalloc: " << e.what() << std::endl;
        cudaSetDevice(0);
    }
    
    if (dim_x>0){
        cout << "dim_x:" << dim_x << endl;
        cout << "dim_y:" << dim_y << endl;
        cout << "i_std:" << sp_options.i_std << endl;
        cout << "nPixels_in_square_side:" << sp_options.nPixels_in_square_side << endl;
    
    }
    // initialize the gpu variables: seg_gpu, border_gpu, sp_params
    nSPs = CudaInitSeg(seg_cpu, seg_gpu, nPixels, sp_options.nPixels_in_square_side, dim_x, dim_y, sp_options.use_hex);
    if (dim_x>0){
        cout << "nSPs:" << nSPs << endl;
    }


    const int sofsparams = sizeof(superpixel_params);
    const int sofsphelper = sizeof(superpixel_GPU_helper);
    sp_params_cpu = (superpixel_params*) malloc(nSPs*sofsparams);

    try{
        throw_on_cuda_error(cudaMalloc((void**) &sp_params, nSPs*sofsparams));
        throw_on_cuda_error(cudaMalloc((void**) &sp_gpu_helper, nSPs*sofsphelper));
    }
    catch(thrust::system_error &e)    {
        std::cerr << "CUDA error after cudaMalloc: " << e.what() << std::endl;
        cudaSetDevice(0);
    }

    CudaFindBorderPixels(seg_gpu, border_gpu, nPixels, dim_x, dim_y, 0);

    CudaInitSpParams(sp_params, sp_options.s_std, sp_options.i_std, nSPs);  

    init_sp = true;
}

void Superpixels::same_size_reinit(){
    nSPs = CudaInitSeg(seg_cpu, seg_gpu,  nPixels, sp_options.nPixels_in_square_side, dim_x, dim_y, sp_options.use_hex);
    CudaFindBorderPixels(seg_gpu, border_gpu, nPixels, dim_x, dim_y, 0);
    CudaInitSpParams(sp_params, sp_options.s_std, sp_options.i_std, nSPs); 
}


//read an rgb image, set the gpu copy, set the float_gpu to be the lab image
void Superpixels::load_img(unsigned char* imgP){       
    memcpy(image_cpu, imgP, dim_i*nPixels*sizeof(unsigned char));  
    cudaMemcpy(image_gpu,image_cpu,dim_i*nPixels*sizeof(unsigned char),cudaMemcpyHostToDevice);
    Rgb2Lab(image_gpu, image_gpu_double, nPixels);  
}


// update seg_gpu, sp_gpu_helper and sp_params
void Superpixels::calc_seg(){
    int prior_sigma_s = sp_options.area * sp_options.area;
    int prior_count = sp_options.prior_count;


    bool cal_cov = sp_options.calc_cov;
    int i_std =  sp_options.i_std;
    int s_std = sp_options.s_std;
    int nInnerIters = sp_options.nInnerIters;

    for (int i = 0 ; i<sp_options.nEMIters; i++){
        // "M step"
        update_param( image_gpu_double, seg_gpu, sp_params, sp_gpu_helper, nPixels, nSPs, dim_x, dim_y,prior_sigma_s,prior_count);
        
        //"(Hard) E step"
        update_seg( image_gpu_double, seg_gpu, border_gpu, sp_params, J_i, logdet_Sigma_i, cal_cov, i_std, s_std, nInnerIters, nPixels, nSPs, dim_x, dim_y);
    }    
    CudaFindBorderPixels(seg_gpu, border_gpu, nPixels, dim_x, dim_y, 1); 
}



//Set the pixels on the superpixel boundary to red: 
Mat Superpixels::get_img_overlaid(){  
    unsigned char* image_border_cpu = (unsigned char*)malloc(dim_i*nPixels*sizeof(unsigned char));
    CUDA_get_image_overlaid(image_gpu, border_gpu, nPixels,dim_x);
    cudaMemcpy(image_border_cpu,image_gpu,dim_i*nPixels*sizeof(unsigned char),cudaMemcpyDeviceToHost);   
    Mat img_border(dim_y, dim_x, CV_8UC3, image_border_cpu);   
    return img_border;  
}



//replace pixel color by superpixel mean
Mat Superpixels::get_img_cartoon(){ 
    // fill in image_mean_gpu with superpixel mean in 
    uchar3* image_mean_gpu;
    unsigned char* image_mean_cpu = (unsigned char*)malloc(dim_i*nPixels*sizeof(unsigned char));
    cudaMalloc((void**) &image_mean_gpu, dim_i*nPixels*sizeof(uchar3));       
    CUDA_get_image_cartoon(image_mean_gpu, seg_gpu, sp_params, nPixels);
    cudaMemcpy(image_mean_cpu,image_mean_gpu,dim_i*nPixels*sizeof(unsigned char),cudaMemcpyDeviceToHost);
    Mat img_mean(dim_y, dim_x, CV_8UC3, image_mean_cpu);
    return img_mean; 
}



void Superpixels::gpu2cpu(){
    Lab2Rgb(image_gpu, image_gpu_double, nPixels);
    cudaMemcpy(seg_cpu, seg_gpu, nPixels*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(border_cpu, border_gpu, nPixels*sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(sp_params_cpu, sp_params, nPixels*sizeof(superpixel_params), cudaMemcpyDeviceToHost);
}



Superpixels::~Superpixels()
{       
    if (init_sp){
        free(sp_params_cpu);
        cudaFree(sp_params);
        cudaFree(sp_gpu_helper);
        //cout << "free sp_params..." << endl;

        free(image_cpu);
        //cout << "free image_cpu" << endl;
        cudaFree(image_gpu);
        //cout << "free image_gpu" << endl;
        cudaFree(image_gpu_double);
        //cout << "free image_gpu_double" << endl;

        free(border_cpu);
        cudaFree(border_gpu);
        //cout << "free border..." << endl;

        free(seg_cpu);
        cudaFree(seg_gpu);
        //cout << "free seg..." << endl;

    }else{
        cout << "init_sp = false" << endl;
    }
    init_sp = false;
    
   // cout << "Object is being deleted" << endl;
}
 
