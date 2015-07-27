#include "Superpixels.h"



static void show_usage(std::string name)
{
    std::cerr << "Usage of " << name << ":\n"
              << "\t-h, --help\t\tShow this help message\n"
              << "\t-i, --img_filename \tSpecify the path of the image to work on\n"
              << "\t-n, --nPixels_on_side \tthe desired number of pixels on the side of a superpixel\n"
              << "\t--i_std \tstd dev for color Gaussians, should be 5<= value <=40. A smaller value leads to more irregular superpixels\n"
              << std::endl;
}



static superpixel_options get_sp_options(int nPixels_in_square_side, int i_std){
    superpixel_options opt;
    opt.nPixels_in_square_side = nPixels_in_square_side; 
    opt.i_std = i_std;

    opt.area = opt.nPixels_in_square_side*opt.nPixels_in_square_side;   
    opt.s_std = opt.nPixels_in_square_side;
    opt.prior_count = 5 * opt.area ;
    opt.calc_cov = true;
    opt.use_hex = true;

    opt.nEMIters = opt.nPixels_in_square_side;
    opt.nInnerIters = 10;
    return opt;
}


//./Sp_demo -i <img_filename> -n <nPixels_on_side> --i_std <i_std>
int main( int argc, char** argv )
{
    
    // get the image filename, nPixels_in_square_side and i_std
    // the defaults
    String filename = "image/1.jpg";
    int nPixels_in_square_side = 15;
    int i_std = 20;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) {
            show_usage(argv[0]);
            return 0;
        } 
        else if ((arg == "-i") || (arg == "--img_filename")) {
            if (i + 1 < argc) { 
                i++;
                filename = argv[i];
            } else {
                std::cerr << "--img_filename option requires one argument." << std::endl;
                return 1;
            }  
        } 
        else if ((arg == "-n") || (arg == "--nPixels_on_side")) {
            if (i + 1 < argc) { 
                i++;
                nPixels_in_square_side = atoi (argv[i]);
                if (nPixels_in_square_side<3) {
                    std::cerr << "--nPixels_in_square_side option requires nPixels_in_square_side >= 3." << std::endl;
                    return 1;
                }

            } else {
                std::cerr << "--nPixels_on_side option requires one argument." << std::endl;
                return 1;
            }  
        }
        else if (arg == "--i_std") {
            if (i + 1 < argc) { 
                i++;
                i_std = atoi (argv[i]);
                
                if (i_std<5 || i_std>40) {
                    std::cerr << "--i_std option requires 5<= value <=40." << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "--i_std option requires a number." << std::endl;
                return 1;
            }  
        }else{  

        }
    }

    cout << "finish reading the arguments" << endl;

    Mat image = imread(filename, CV_LOAD_IMAGE_COLOR);
    // Check for invalid input
    if(! image.data ){
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    cout << "finish reading the image" << endl;
    //Part 1: Specify the parameters:
    superpixel_options spoptions = get_sp_options(nPixels_in_square_side, i_std);

    // Part 2 : prepare for segmentation
    int dimy = image.rows;
    int dimx = image.cols;
    Superpixels sp = Superpixels(dimx, dimy, spoptions);
    //cout << "finish init sp" << endl;
    cudaDeviceSynchronize();
    sp.load_img((unsigned char*)(image.data));

   //cout << "finish loading the image" << endl;

    // Part 3: Do the superpixel segmentation
    clock_t start,finish;
    start = clock();
    sp.calc_seg();
    cudaDeviceSynchronize();

    sp.gpu2cpu();
    
    cudaError_t err_t = cudaDeviceSynchronize();
    if (err_t){
        std::cerr << "CUDA error after cudaDeviceSynchronize. " << std::endl;
        return 0;
    }
    finish = clock();
    cout<< "Segmentation takes " << ((double)(finish-start)/CLOCKS_PER_SEC) << " sec" << endl;

    // Part 4: Save the mean/boundary image 
    
    Mat border_img = sp.get_img_overlaid();
    String fname_res_border = "image/result/img_border.png";
    imwrite(fname_res_border, border_img);
    cout << "saved " << fname_res_border << endl;
    

    Mat mean_img = sp.get_img_cartoon();
    String fname_res_mean = "image/result/img_mean.png";
    imwrite(fname_res_mean, mean_img);
    cout << "saving " << fname_res_mean << endl;

                
    return 0;
}