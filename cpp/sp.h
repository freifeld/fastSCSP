#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "my_sp_struct.h"
#endif

/*
__device__ inline double calc_squared_mahal_2d(double* val,
                                        const double2 mu,
                                        const double4 J
                                        ){
    // val: value of interest                 
    // mu: mean
    // J: inverse of the covariance                     
    double J00 = J.x;
    double J01 = J.y;
    double J11 = J.w;   
    
    double x0 = val[0]-mu.x;
    double x1 = val[1]-mu.y;
    
    double res = x0*x0*J00 + x1*x1*J11 + 2*x0*x1*J01;
        
    return res;   
}


__device__ inline double calc_squared_mahal_3d(double* val,
                                        const double3 mu,
                                        const double3 J
                                        ){
    // val: value of interest                 
    // mu: mean
    // J: inverse of the covariance                     
    double J00 = J.x;
    //double J01 = 0.0;
    //double J02 = 0.0;
    double J11 = J.y;   
    //double J12 = 0.0; 
    double J22 = J.z; 
    
    double x0 = val[0]-mu.x;
    double x1 = val[1]-mu.y;
    double x2 = val[2]-mu.z;
    
    //double res = x0*x0*J00 + x1*x1*J11 + x2*x2*J22 +
    //     2*(x0*x1*J01 + x0*x2*J02 + x1*x2*J12);

    double res = x0*x0*J00 + x1*x1*J11 + x2*x2*J22;              
    return res;              

}

__device__ inline double calc_squared_eucli_2d(double* val,const double2 mu, const int std){
    // val: value of interest                                 
    // mu: mean      
    double x0 = val[0]-mu.x;
    double x1 = val[1]-mu.y;
    double res = (x0*x0 + x1*x1)/double(std)/double(std);      
    return res;   
}

__device__ inline double calc_squared_eucli_3d(double* val, const double3 mu, const int std){
    // val: value of interest                                 
    // mu: mean
    double x0 = val[0]-mu.x;
    double x1 = val[1]-mu.y;
    double x2 = val[2]-mu.z;     
    double res = (x0*x0 + x1*x1 + x2*x2)/double(std)/double(std);                 
    return res;              
}
*/

    
__device__ bool lut[256] = {0,0,1,1,0,0,1,1,1,1,0,1,0,0,0,1,1,0,0,0,1,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,1,0,0,0,1,0,1,1,1,1,0,1,1,1,1,1};
    
    
__device__ inline bool ischangbale_by_nbrs(bool* nbrs){
  // This function does the following:
  // 1) converts the arrray of binary labels of the 8 nbrs into an integer  btwn 0 and 255
  // 2) does a lookup check on the resulting function using the ischangbale_by_num function.
    /*int num=(nbrs[7]+    // SE
             nbrs[6]*2+  // S
             nbrs[5]*4+  // SW
             nbrs[4]*8+  // E
             nbrs[3]*16+ // W
             nbrs[2]*32+ // NE
             nbrs[1]*64+ // N
             nbrs[0]*128); // NW  
    */
    int num = 0;
#pragma unroll
   for (int i=7; i>=0; i--){    
      num <<= 1;
      if (nbrs[i]) num++;
   } 
    if (num == 0)
    return 0;
  else     
    return lut[num];  
    //return ischangbale_by_num(num);
    }
    
   
/*
* Set the elements in nbrs "array" to 1 if corresponding neighbor pixel has the same superpixel as "label"
*/
__device__ inline void set_nbrs(int idx, int xdim, int ydim,
                                const bool x_greater_than_1,
                                const bool y_greater_than_1,
                                const bool x_smaller_than_xdim_minus_1,
                                const bool y_smaller_than_ydim_minus_1,
                                int* seg, bool* nbrs,int label){
    // init            
    nbrs[0]=nbrs[1]=nbrs[2]=nbrs[3]=nbrs[4]=nbrs[5]=nbrs[6]=nbrs[7]=0;
    
    if (x_greater_than_1 && y_greater_than_1){// NW        
        nbrs[0] = (label == seg[idx-xdim-1]);  
    }
    if (y_greater_than_1){// N        
        nbrs[1] = (label == seg[idx-xdim]);  
    }
    if (x_smaller_than_xdim_minus_1 && y_greater_than_1){// NE         
        nbrs[2] = (label == seg[idx-xdim+1]);  
    }
    if (x_greater_than_1){// W
        nbrs[3] = (label == seg[idx-1]);        
    }
    if (x_smaller_than_xdim_minus_1){// E
       nbrs[4] = (label == seg[idx+1]);  
    }
    if (x_greater_than_1 && y_smaller_than_ydim_minus_1){// SW
       nbrs[5] = (label == seg[idx+xdim-1]);  
    }
    if (y_smaller_than_ydim_minus_1){// S 
       nbrs[6] = (label == seg[idx+xdim]);  
    }    
    if (x_smaller_than_xdim_minus_1 && y_smaller_than_ydim_minus_1){// SE
       nbrs[7] = (label == seg[idx+xdim+1]);  
    }
    return;
}   


__device__ inline double cal_posterior(
    bool isValid,
    double* imgC, 
    int x, int y, double* pt,
    superpixel_params* sp_params,  
     int seg,
    double3 J_i, double logdet_Sigma_i, 
    bool cal_cov, int i_std, int s_std)
{
      double res = -100000000; // some large negative number  
      if (isValid){
        const double3 mu_i = sp_params[seg].mu_i;
        const double3 sigma_s = sp_params[seg].sigma_s;
        const double2 mu_s = sp_params[seg].mu_s;
        
        double x0 = imgC[0]-mu_i.x;
        double x1 = imgC[1]-mu_i.y;
        double x2 = imgC[2]-mu_i.z;

        double d0 = x - mu_s.x;
        double d1 = y - mu_s.y;

        const double logdet_Sigma_s = sp_params[seg].logdet_Sigma_s;
        const double log_count = sp_params[seg].log_count;

        if (true){   
            //color component         
            res = - (x0*x0*J_i.x + x1*x1*J_i.y + x2*x2*J_i.z);   //res = -calc_squared_mahal_3d(imgC,mu_i,J_i);            
            res -= logdet_Sigma_i;               
         
            //space component
            res -= d0*d0*sigma_s.x;
            res -= d1*d1*sigma_s.z;
            res -= 2*d0*d1*sigma_s.y;            // res -= calc_squared_mahal_2d(pt,mu_s,J_s);              
            res -= logdet_Sigma_s; 

        }else{        
       //     res = -calc_squared_eucli_3d(imgC, mu_i, sp_options.i_std);
              res = (x0*x0 + x1*x1 + x2*x2)/double(i_std)/double(i_std); 
     //       res -= calc_squared_eucli_2d(pt, mu_s, sp_options.s_std); 
              res -= ((d0*d0 + d1*d1)/double(s_std)/double(s_std));     
        }
       
        //add in prior prob
#if USE_COUNTS 
        const double prior_weight = 0.5;
        res *= (1-prior_weight);
        double prior = prior_weight * log_count;
        res += prior;
#endif    
      }
      return res;
}
  