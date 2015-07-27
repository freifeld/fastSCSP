#ifndef OUT_OF_BOUNDS_LABEL
#define OUT_OF_BOUNDS_LABEL -1
#endif

#ifndef BAD_TOPOLOGY_LABEL 
#define BAD_TOPOLOGY_LABEL -2
#endif

#ifndef NUM_OF_CHANNELS 
#define NUM_OF_CHANNELS 3
#endif


#ifndef USE_COUNTS
#define USE_COUNTS 1
#endif


#ifndef OUT_OF_BOUNDS_LABEL
#define OUT_OF_BOUNDS_LABEL -1
#endif

#define THREADS_PER_BLOCK 1024


#include "update_seg.h"
#include "sp.h"

#include <stdio.h>


__host__ void CudaFindBorderPixels(const int* seg, bool* border, const int nPixels, const int xdim, const int ydim, const int single_border){   
    int num_block = ceil( double(nPixels) / double(THREADS_PER_BLOCK) ); 
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,1);
    find_border_pixels<<<BlockPerGrid,ThreadPerBlock>>>(seg,border,nPixels, xdim, ydim, single_border);
}


__global__  void find_border_pixels(const int* seg, bool* border, const int nPixels, const int xdim, const int ydim, const int single_border){   
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  
    if (idx>=nPixels) return; 

    border[idx]=0;  // init        
    
    int x = idx % xdim;
    int y = idx / xdim;

    int C = seg[idx]; // center 
    int N,S,E,W; // north, south, east,west            
    N=S=W=E=OUT_OF_BOUNDS_LABEL; // init 
    
    if (y>1){
        N = seg[idx-xdim]; // above
    }          
    if (x>1){
        W = seg[idx-1];  // left
    }
    if (y<ydim-1){
        S = seg[idx+xdim]; // below
    }   
    if (x<xdim-1){
        E = seg[idx+1];  // right
    }       
   
    // If the nbr is different from the central pixel and is not out-of-bounds,
    // then it is a border pixel.
    if ( (N>=0 && C!=N) || (S>=0 && C!=S) || (E>=0 && C!=E) || (W>=0 && C!=W) ){
        if (single_border){
            if (N>=0 && C>N) border[idx]=1; 
            if (S>=0 && C>S) border[idx]=1;
            if (E>=0 && C>E) border[idx]=1;
            if (W>=0 && C>W) border[idx]=1;
        }else{   
            border[idx]=1;  
        }
    }

    return;        
}



__host__ void update_seg(double* img, int* seg, bool* border, 
                        superpixel_params* sp_params, 
                        const double3 J_i, const double logdet_Sigma_i, 
                        bool cal_cov, int i_std, int s_std,
                        int nInnerIters,
                        const int nPixels, const int nSPs, int xdim, int ydim){
    int num_block = ceil( double(nPixels) / double(THREADS_PER_BLOCK) ); 
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,1);

    int single_border = 0 ;
    for (int iter = 0 ; iter < nInnerIters; iter++){
        for (int xmod3 = 0 ; xmod3 <3; xmod3++){
            for (int ymod3 = 0; ymod3 <3; ymod3++){
                //find the border pixels
                find_border_pixels<<<BlockPerGrid,ThreadPerBlock>>>(seg, border, nPixels, xdim, ydim, single_border);
                update_seg_subset<<<BlockPerGrid,ThreadPerBlock>>>(img, seg, border, sp_params, J_i, logdet_Sigma_i,  cal_cov, i_std, s_std, nPixels, nSPs,xdim, ydim, xmod3, ymod3);

            }
        }
    }
}


/*
* Update the superpixel labels for pixels 
* that are on the boundary of the superpixels
* and on the (xmod3, ymod3) position of 3*3 block
*/
__global__  void update_seg_subset(
    double* img, int* seg, bool* border,
    superpixel_params* sp_params, 
    const double3 J_i, const double logdet_Sigma_i,  
    bool cal_cov, int i_std, int s_std, 
    const int nPts,const int nSuperpixels,
    const int xdim, const int ydim,
    const int xmod3, const int ymod3)
{   
    int idx = threadIdx.x + blockIdx.x*blockDim.x; 
    if (idx>=nPts)  return;

    if (border[idx]==0) return;   

    int x = idx % xdim;  
    if (x % 3 != xmod3) return;  
    int y = idx / xdim;   
    if (y % 3 != ymod3) return;   
    
    const bool x_greater_than_1 = (x>1);
    const bool y_greater_than_1 = (y>1);
    const bool x_smaller_than_xdim_minus_1 = x<(xdim-1);
    const bool y_smaller_than_ydim_minus_1 = y<(ydim-1);
    
    int C = seg[idx]; // center 
    int N,S,E,W; // north, south, east,west        

    N = S = W = E = OUT_OF_BOUNDS_LABEL; // init to out-of-bounds 
    
    bool nbrs[8];
            
    bool isNvalid = 0;
    bool isSvalid = 0;
    bool isEvalid = 0;
    bool isWvalid = 0;
     
    // In the implementation below, if the label of the center pixel is
    // different from the labels of all its (4-conn) nbrs -- that is, it is 
    // a single-pixel superpixel -- then we allow it to die off inspite the fact
    // that this "changes the connectivity" of this superpixel. 

    if (x_greater_than_1){
        N = seg[idx-xdim]; // the label, above
        set_nbrs(idx,xdim,ydim,x_greater_than_1,y_greater_than_1,x_smaller_than_xdim_minus_1,y_smaller_than_ydim_minus_1,seg,nbrs,N);
        isNvalid = ischangbale_by_nbrs(nbrs);   
        if (!isNvalid){              
            N = BAD_TOPOLOGY_LABEL;
            if (N == C) return; // Bug fix, 03/12/2015, Oren Freifeld
        }            
    }
    

       
    if (y_greater_than_1){
        W = seg[idx-1];  // left
        set_nbrs(idx,xdim,ydim,x_greater_than_1,y_greater_than_1,x_smaller_than_xdim_minus_1,y_smaller_than_ydim_minus_1,seg,nbrs,W);
        isWvalid = ischangbale_by_nbrs(nbrs);   
        if (!isWvalid){ 
            W = BAD_TOPOLOGY_LABEL;
            if (W == C) return; // Bug fix, 03/12/2015, Oren Freifeld
        }
    }


    if (y_smaller_than_ydim_minus_1){
        S = seg[idx+xdim]; // below
        set_nbrs(idx,xdim,ydim,x_greater_than_1,y_greater_than_1,x_smaller_than_xdim_minus_1,y_smaller_than_ydim_minus_1,seg,nbrs,S);
        isSvalid = ischangbale_by_nbrs(nbrs);   
        if (!isSvalid){       
            S = BAD_TOPOLOGY_LABEL;
            if (S == C) return; // Bug fix, 03/12/2015, Oren Freifeld
        }
    }

    if (x_smaller_than_xdim_minus_1){
        E = seg[idx+1];  // right
        set_nbrs(idx,xdim,ydim,x_greater_than_1,y_greater_than_1,x_smaller_than_xdim_minus_1,y_smaller_than_ydim_minus_1,seg,nbrs,E);
        isEvalid = ischangbale_by_nbrs(nbrs);   
        if (!isEvalid){             
            E = BAD_TOPOLOGY_LABEL;
            if (E == C) return; // Bug fix, 03/12/2015, Oren Freifeld
        }      
    }           


    double* imgC = img + idx * 3;   

    double pt[2];
    pt[0]=(double)x;
    pt[1]=(double)y;


    //---------------
    // log-likelihood  (ignoring constants)
    //---------------   


    
    double resN = cal_posterior(isNvalid, imgC, x,y, pt, sp_params, N, J_i, logdet_Sigma_i, cal_cov, i_std, s_std);
    
    double resS = cal_posterior(isSvalid, imgC, x,y, pt, sp_params, S, J_i, logdet_Sigma_i, cal_cov, i_std, s_std);
    
    double resE = cal_posterior(isEvalid, imgC, x,y, pt, sp_params, E, J_i, logdet_Sigma_i, cal_cov, i_std, s_std);
    
    double resW = cal_posterior(isWvalid, imgC, x,y, pt, sp_params, W, J_i, logdet_Sigma_i, cal_cov, i_std, s_std);



    bool all_are_valid = (isNvalid || N==OUT_OF_BOUNDS_LABEL) && 
                         (isSvalid || S==OUT_OF_BOUNDS_LABEL) && 
                         (isEvalid || E==OUT_OF_BOUNDS_LABEL) && 
                         (isWvalid || W==OUT_OF_BOUNDS_LABEL);
    

    if (!all_are_valid)  return;

    //double res_max = -1; // some small negative number (use when using l)
    double res_max = log(.000000000000000001); // (use when using ll)
    
    int arg_max = C; // i.e., no change
    
    // In the tests below, the order matters: 
    // E.g., testing (res_max<resN && isNvalid) is wrong!
    // The reason is that if isNvalid, then the test max<resN has no meaning.
    // The correct test is thus isNvalid && res_max<resN. 
    
    
    if (isNvalid && res_max<resN ){ 
        res_max=resN;
        arg_max=N;
    }
    
    if (isSvalid && res_max<resS ){
        res_max=resS;
        arg_max=S;
    }

    if (isEvalid && res_max<resE){
        res_max=resE;
        arg_max=E;
    }

    if (isWvalid && res_max<resW){
        res_max=resW;
        arg_max=W;
    }     

    // update seg
    seg[idx] = arg_max;     
    return;   
}