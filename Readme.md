







# fastSCSP: A Fast Method for Inferring High-Quality Simply-Connected Superpixels


Authors: 
Oren Freifeld  (email: freifeld@csail.mit.edu)
Yixin Li (email: liyixin@mit.edu)

An early and partial version of this code was written by Oren Freifeld. 
The rest of the code was then completed by Yixin Li.

This software is released under the MIT License (included with the software).
Note, however, that using this code (and/or the results of running it) 
to support any form of publication (e.g.,a book, a journal paper, 
a conference paper, a patent application, etc.) requires you to cite
the following paper:

	@incollection{Freifeld:ICIP:2015,
	  title={A Fast Method for Inferring High-Quality Simply-Connected Superpixels},
	  author={Freifeld, Oren and Li, Yixin and Fisher III, John W},
	  booktitle={International Conference on Image Processing},
	  year={2015},
	}


Versions
--------
05/08/2015: Version 1.01 Minor bug fixes  (directory tree; relative paths; some windows-vs-linux issue) 

05/07/2015: Version 1.0  First release)




Programming Language
--------
The current implementation is essentially a python wrapper around CUDA.
In the future (hopefully by the end of summer 2015), we intend to release two alternative wrappers that do no require python (one for Matlab, and one for C++). 

Requirements 
-------------

CUDA (version >= 5.5)

Numpy (version: developed/tested on 1.8. Some older versions should *probably* be fine too)

Scipy (version: developed/tested on 0.13.  Some older versions should *probably* be fine too)

matplotlib (version: developed/tested on 1.3.1.  Some older versions should *probably* be fine too)

pycuda (version: >= 2013.1.1)

OS: 

Developed/tested on Ubuntu 12.04 64-bit and Ubuntu 14.04 64-bit. 

Also tested on Windows 7 Professional 64-bit.

Instructions
------------

See demo.py for an example of running the algorithm on a single image.
See demo_for_direc.py for an example of running the algorithm on all files in a directory (this assumes that all files are in valid image format).

See the end of this README file for options the user can choose to speed up the initialization.





To run the algorithm on the default image (image/1.jpg) with default parameters:

	 python demo.py

To run on a user-specified image with default parameters:

	 python demo.py -i <img_filename>

For help:

	 python demo.py -h

To run on a user-specified image with user-specified parameters:

	 python demo.py -i <img_filename> -n <nPixels_on_side> --i_std <i_std>

In the initialization, the area of each superpixel is, more or less, nPixels_on_side is^2. 
Let K denote the number of superpixels. High nPixels_on_side means small K and vice versa.
The i_std controls the tradeoff between spatial and color features. A small i_std means a small standard deviation for the color features
(and thus making their effect more significant). In effect, small i_std = less regular boundaries.

To run the algorithm on all files in a user-specified directory:
Replace "demo.py" with "demo_for_direc.py" and replace "-i <img_filename>" with "-d <directory_name>"
The rest is the same as above.

Example 1: 
To run superpixel code on all images under default directory (./image):

	 python demo_for_direc.py
	 
Example 2: 

	 python demo_for_direc.py -d <directory_name>
  
Example 3: 

	 python demo_for_direc.py -d <directory_name> -n <nPixels_on_side> --i_std <i_std>
	 
Example 4: If all the images in the directory have the same size, you can save computing time by using

         python demo_for_direc.py -d <directory_name> --imgs_of_the_same_size 
         
(with or without the options for nPixels_on_side  and i_std mentioned above)

For help:

	 python demo_for_direc.py -h



The main functions in demo.py and demo_for_direc.py are:

1. Construct the SuperpixelsWrapper object (internally, this also initializes the segmentation according to the number of superpixels and image size):

	 sw = SuperpixelsWrapper(...)
	 
2. set the input image
  
	 sw.set_img(img)

3. compute the superpixels segmentation
  
	 sw.calc_seg(...)

4. copy parameters from gpu to cpu
  
	 sw.gpu2cpu()


Speeding up the initialization step
-----------------------------------

By default, we use an hexagonal honeycomb tiling, computed (on the GPU) using brute force. When K and/or the image is very large, this can be a bit slow.
Setting use_hex=False will use squares instead of hexagons. This will be faster, but less visually pleasing
(that said, in case you are obssesed with benchmarks, note we found that in comparison to hexagons, squares give slighly better results on benchmarks, although we didn't bother to include the square-based results in the paper). 
We thus suggest to stay with hexagons. To speed up the hexagonal initialization, 
we first try to load a precomputed initialization since it does not depend on the actual image. It depends only 
on the number of superpixels and the image size. If it doesn't exist, we compute and save it for a future use. Thus,
the next time the user calls the algorithm with the same image size and same K, it will be loaded instead of being computed from scratch.

As an aside remark, we actually played with several smarter options for computing the honeycomb tiling.
E.g., we used Warfield's K-DT algorithm (With K=1; his K does not have the same meaning as ours). 
However, while the complexity was fixed wrt (our) K, it was still slow for large images. Since loading precomputed results was much simpler and faster, we decided to go with that option.

Another step that takes some time but does not depend on the actual image is the construction of the SuperpixelWrapper object
(this will be faster once we move to a C++ wrapper). However, when running the algorithm on 
many images in the same directioy where all images have the same size, this object needs to be constructed only once.  This is what the --imgs_of_the_same_size option mentioned above does, leading to some speedups.

