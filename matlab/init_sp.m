function [sp, params, gpu_helper, option] = init_sp(dimx, dimy, nPixels_in_square_side, i_std)
    sp.nPts = dimx * dimy;
    sp.dim_s = 2;
    sp.dim_i = 3;
    sp.dimy = dimy;
    sp.dimx = dimx;
    
    %reset(gpuDevice);
    sp.threads_per_block = 512;
    sp.block =[sp.threads_per_block,1,1]; 
    num_block = ceil( sp.nPts / sp.threads_per_block);
    sp.grid = [num_block,1,1]; 

    option = get_sp_options(nPixels_in_square_side, i_std);
    %%
    % init seg, border and params
    kernel_honeycomb = parallel.gpu.CUDAKernel('honeycomb.ptx', 'honeycomb.cu');
    kernel_honeycomb.ThreadBlockSize = sp.block;
    kernel_honeycomb.GridSize = sp.grid;

    M = option.nPixels_in_square_side;
    if option.use_hex
        % first test if we already have the segmentation
        seg_filename = strcat(num2str(dimx),'_',num2str(dimy), '_', num2str(M), '.mat');
        if exist(seg_filename,'file')
            x = load(seg_filename);
            sp.seg_cpu = x.seg;
            sp.seg_gpu = gpuArray(sp.seg_cpu);
            sp.nSps = x.nSps;
        else
            disp('Did not find a precomputed seg initialization... Recompute and save.')
            sp.seg_gpu = gpuArray(zeros(dimy,dimx, 'int32'));
            H = sqrt(M^2 / (1.5 *sqrt(3)));
            W = sqrt(3)*H;
            [YY,XX] = meshgrid(0:W:dimx, 0:(1.5*H):dimy);
            YY(1:2:end,:) = YY(1:2:end,:) + W*0.5;
            X = reshape(XX,1,[]);
            Y = reshape(YY,1,[]);
            centers = vertcat(X,Y);
            [~,sp.nSps] = size(X);
            sp.centers_cpu = reshape(centers, [],1);
            sp.centers_gpu = gpuArray(sp.centers_cpu);
            sp.seg_gpu = feval(kernel_honeycomb, sp.seg_gpu, sp.centers_gpu,sp.nSps, sp.nPts, dimy, dimx); 
            sp.seg_cpu = gather(sp.seg_gpu);
            seg = sp.seg_cpu;
            nSps = sp.nSps;
            save(seg_filename,'seg','nSps');
        end
    else
            sp.seg_cpu = zeros(dimy,dimx, 'int32');
            [yy,xx] = meshgrid(0:1:dimx-1,0:1:dimy-1); 
            xx = double(xx);
            yy = double(yy);
            nTimesInX = ceil(dimx/M) ;
            nTimesInY = ceil(dimy/M);
            sp.nSps = nTimesInX * nTimesInY;
            sp.seg_cpu = floor(yy/M)*nTimesInX + floor(xx/M);
            sp.seg_gpu = gpuArray( int32(sp.seg_cpu));
    end

    imwrite(double(sp.seg_cpu)/sp.nSps,'seg_init.png');

    % init border
    kernel_find_border = parallel.gpu.CUDAKernel('update_seg.ptx', 'update_seg.cu', 'find_border_pixels');
    kernel_find_border.ThreadBlockSize = sp.block;
    kernel_find_border.GridSize = sp.grid;

    border_init_gpu =  gpuArray(false([dimy, dimx]));
    [~, sp.border_gpu] = feval(kernel_find_border, sp.seg_gpu, border_init_gpu, sp.nPts, dimy, dimx, 0);
    %imwrite(gather(sp.border_gpu), 'border_init.png');
    
    % reshape border and seg into gpu 
    sp.seg_gpu = reshape(sp.seg_gpu, [],1);
    sp.border_gpu = reshape(sp.border_gpu, [],1);
    
    [params, gpu_helper] = init_params(sp.dim_i, sp.dim_s, option.i_std, option.s_std, sp.nSps);
    
end


function [option] = get_sp_options(nPixels_in_square_side, i_std)
    % set the superpixel option
    option.nPixels_in_square_side = nPixels_in_square_side; 
    option.i_std = i_std;
    option.area = option.nPixels_in_square_side* option.nPixels_in_square_side;
    option.s_std = option.nPixels_in_square_side;
    option.calc_cov = true;
    option.use_hex = true;
    option.prior_count = 5 * option.area;
    option.nEMIters = option.nPixels_in_square_side;
    option.nInnerIters = 10;
    option.prior_prob_weight =  0.5;
end


function [params, gpu_helper] = init_params(dim_i, dim_s, i_std, s_std, nSps)
    % init sp_params
    params.mu_i_gpu = gpuArray( zeros(nSps * dim_i, 1,'double')); 
    params.mu_s_gpu = gpuArray( zeros(nSps * dim_s, 1,'double')); 
    params.Sigma_s_gpu = gpuArray( zeros(nSps * dim_s * dim_s, 1));
    params.Sigma_i_gpu = gpuArray( zeros(nSps * dim_i * dim_i, 1));
    params.prior_sigma_s_sum_gpu = gpuArray( zeros(nSps * dim_s * dim_s, 1));
    params.J_s_gpu = gpuArray( zeros(nSps * dim_s * dim_s, 1));
    params.J_i_gpu = gpuArray( zeros(nSps * dim_i * dim_i, 1));
    params.logdet_Sigma_i_gpu = gpuArray( zeros(nSps, 1));
    params.logdet_Sigma_s_gpu = gpuArray( zeros(nSps, 1));
    params.counts_gpu = gpuArray( zeros(nSps, 1, 'int32')); 

    half_i_std_square = (i_std/2)^2;
    i_std_square = i_std^2;
    s_std_square = s_std ^ 2;
    % Covariance for each superpixel is a diagonal matrix
    params.Sigma_s_gpu(1:4:end) = s_std_square;  
    params.Sigma_s_gpu(4:4:end) = s_std_square; 
    params.prior_sigma_s_sum_gpu(1:4:end) = s_std_square * s_std_square;  
    params.prior_sigma_s_sum_gpu(4:4:end) = s_std_square * s_std_square;

    params.Sigma_i_gpu(1:9:end) = half_i_std_square ;  % To account for scale differences between the L,A,B
    params.Sigma_i_gpu(5:9:end) = i_std_square ; 
    params.Sigma_i_gpu(9:9:end) = i_std_square; 

    %calculate the inverse of covariance
    params.J_s_gpu(1:4:end) = 1 / s_std_square;  
    params.J_s_gpu(4:4:end) = 1 / s_std_square; 
    params.J_i_gpu(1:9:end) = 1 / half_i_std_square;
    params.J_i_gpu(5:9:end) = 1 / i_std_square; 
    params.J_i_gpu(9:9:end) = 1 / i_std_square; 

    %calculate the log of the determinant of covriance
    params.logdet_Sigma_i_gpu(:) = log(half_i_std_square * i_std_square * i_std_square);
    params.logdet_Sigma_s_gpu(:) = log(s_std_square * s_std_square);

    gpu_helper.mu_i_sum = gpuArray(zeros(nSps*dim_i, 1, 'int32'));
    gpu_helper.mu_s_sum = gpuArray(zeros(nSps*dim_s, 1,'int32'));  
    gpu_helper.prior_sigma_s = params.prior_sigma_s_sum_gpu;
    gpu_helper.sigma_s_sum = gpuArray(zeros(nSps*3, 1,'uint64'));
    gpu_helper.log_count = gpuArray(zeros(nSps, 1, 'double'));
end


