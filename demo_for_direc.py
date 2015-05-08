#!/usr/bin/env python
"""
Created on Wed Oct 15 09:28:58 2014

Authors:
1)
Oren Freifeld
Email: freifeld@csail.mit.edu

2)
Yixin Li
Email: liyixin@mit.edu

Example usage of the superpixels code on all files in a directory
"""
import os
from sys import path
if '.'+os.sep not in path:
    path.insert(0,'.'+os.sep)
    
from scipy import misc
import scipy.io as sio

from superpixels.SuperpixelsWrapper import SuperpixelsWrapper
from of.utils import *


def get_list_of_all_imgs_dir(dirname,
                             exts = ['.jpg','.jpeg','.png','.bmp','.ppm']):
    L = os.listdir(dirname) 
    exts = exts + [ item.upper()  for item in exts]     
    L = [ x for x in L if os.path.splitext(x)[-1] in exts]     
    L.sort()    
    return L
    
def main(image_direc = 'image',
        nPixels_on_side = 15,
        i_std = 15, # std dev for color Gaussian
        imgs_of_the_same_size=None
        ):
    if image_direc is None:
        raise ValueError("image_direc cannot be None")
    image_direc = os.path.expanduser(image_direc)
    image_direc = os.path.abspath(image_direc)
    FilesDirs.raise_if_dir_does_not_exist(image_direc)
    imgs = get_list_of_all_imgs_dir(image_direc)     
    if not imgs:
        raise Exception("\n\nDirectory \n{}\ncontains no images.\n".format(image_direc))
    
    # Results will be saved in the /results folder under input directory. You can change it.
    save_path_root = os.path.join(image_direc , 'result')
    print "I am going to save results into",save_path_root
    FilesDirs.mkdirs_if_needed(save_path_root)
    
    # Part 1: Specify the parameters:
    prior_count_const = 5  # the weight of Inverse-Wishart prior of space covariance(ex:1,5,10)
    use_hex = True # toggle between hexagons and squares (for the init)
    prior_weight = 0.5 # in the segmentation, 
                       # we do argmax w * log_prior + (1-w) *log_likelihood.
                       # Keeping w (i.e., prior_weight) at 0.5 means we are trying 
                       # to maximize the true posterior. 
                       # We keep the paramter here in case the user will like 
                       # to tweak it. 
                        
    calc_s_cov = True # If this is False, then we avoid estimating the spatial cov.
    num_EM_iters = nPixels_on_side
    num_inner_iters = 10
    sp_size = nPixels_on_side*nPixels_on_side 

    
    
    
    
    

    for i,img_filename in enumerate(imgs): 

        # Part 2 : prepare for segmentation
        print img_filename
        fullfilename = os.path.join(image_direc,img_filename)
        FilesDirs.raise_if_file_does_not_exist(fullfilename)
        img = misc.imread(fullfilename)
        
        dimx=img.shape[1]
        dimy=img.shape[0]
        
        if not((img.ndim in [1,3]) and (img.shape[2] in [1,3])):
            raise ValueError("\n\nProblem with {0}\n\nI was expecting 3 channels, but img.shape = {1}\n\n".format(fullfilename,img.shape))
        
        tic = time.clock()
        if (imgs_of_the_same_size and i !=0):
            sw.initialize_seg()
            # you can use the same SuperpixelsWrapper object with different imgs and/or, 
            # i_std, s_std, prior_count. 
            # Just call sw.set_img(new_img), sw.initialize_seg(), and/or
            # sw.set_superpixels(i_std=..., s_std = ..., prior_count = ...)
            # again and recompute the seg.
        else:
            sw = SuperpixelsWrapper(dimy=dimy,dimx=dimx, nPixels_in_square_side=nPixels_on_side,
                                 i_std = i_std , s_std = nPixels_on_side, 
                                 prior_count = prior_count_const*sp_size,
                                 use_hex = use_hex 
                                )      
        
        toc = time.clock()
        print 'init time = ', toc-tic
        print 'nSuperpixels =', sw.nSuperpixels    
        sw.set_img(img) 



        # Part 3: Do the superpixel segmentation

        tic  = time.clock() 
        #actual work
        sw.calc_seg(nEMIters=num_EM_iters, nItersInner=num_inner_iters, calc_s_cov=calc_s_cov, prior_weight=prior_weight)
        # Copy the parameters from gpu to cpu
        sw.gpu2cpu() 
        toc  = time.clock()
        print 'superpixel calculation time = ',toc-tic


        # Part 4: Save the mean/boundary image and the resulting parameters

        img_overlaid = sw.get_img_overlaid()   # get the boundary image                      
        img_cartoon = sw.get_cartoon()         # get the cartoon image
        grid = ['square','hex'][sw.use_hex]

        root_slash_img_num = os.path.splitext(img_filename)[0]
        img_num = os.path.split(root_slash_img_num)[1]
        #fname_res_border = '_'.join([img_num , 'std', str(i_std), 'border', grid+'.png'])
        #fname_res_cartoon = '_'.join([img_num , 'std', str(i_std), 'mean', grid+'.png'])
        
        fname_res_border =  '_'.join([img_num,'n','{0:03}'.format(nPixels_on_side), 'std', '{0:02}'.format(i_std), 'border', grid+'.png'])
        fname_res_cartoon = '_'.join([img_num, 'n','{0:03}'.format(nPixels_on_side), 'std', '{0:02}'.format(i_std), 'mean', grid+'.png'])
        
        
        fname_res_border = os.path.join(save_path_root,fname_res_border)
        fname_res_cartoon = os.path.join(save_path_root,fname_res_cartoon)

        print 'saving',fname_res_border
        misc.imsave(fname_res_border, img_overlaid)
        print 'saving',fname_res_cartoon
        misc.imsave(fname_res_cartoon, img_cartoon)

        #save the resulting parameters to MATLAB .mat file 
        #mat_filename = os.path.join(save_path_root , img_num + '_std_'+str(i_std)+'.mat')
        mat_filename = os.path.join(save_path_root#, img_num + '_std_'+str(i_std)+'.mat')
                                              ,'_'.join([img_num,'n','{0:03}'.format(nPixels_on_side), 'std', '{0:02}'.format(i_std)+'.mat']))
        print 'Saving params to ',mat_filename 

        pm = sw.superpixels.params                  
        sio.savemat(mat_filename,{'pixel_to_super':sw.seg.cpu, 'count':pm.counts.cpu,
                                  'mu_i': pm.mu_i.cpu, 'mu_s':pm.mu_s.cpu, 
                                  'sigma_s':pm.Sigma_s.cpu, 'sigma_i':pm.Sigma_i.cpu})
        if (not imgs_of_the_same_size):
            del sw


if __name__ == "__main__":  
    import argparse
     
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--image_direc", help="the directory to work on",
                        nargs='?',
                        const='image',default='image')
    parser.add_argument("-n","--nPixels_on_side", type=int, help="the desired number of pixels on the side of a superpixel",
                        nargs='?',
                        const=10,default=10)
    parser.add_argument("--i_std", type=int, help="std dev for color Gaussians, in [5,40]. A smaller value leads to more irregular superpixels",
                        nargs='?',
                        const=15,default=15)
    
    # If not passed, the value below defaults to False
    parser.add_argument("--imgs_of_the_same_size",
                        help="If passed, it will be assumed that all images in the directory have the same size. This will save some time.",
                        action="store_true")

    
    args = parser.parse_args()    
    main(**args.__dict__)    
    
