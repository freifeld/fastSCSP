%% load all the kernels here
function [kernel_lab_to_rgb, kernel_rgb_to_lab, kernel_find_border,...
    kernel_clear_fields, kernel_sum_by_label, kernel_calculate_mu_and_sigma,...
    kernel_clear_fields2, kernel_sum_by_label2, kernel_calculate_mu,...
    kernel_update_seg_subset, kernel_get_cartoon]...
        = load_all_kernels(nSps, threads_per_block, block, grid) 
    % grid size for parallization over superpixels
    num_block_sp = ceil( nSps / threads_per_block);
    grid_sp = [num_block_sp,1,1]; 

    kernel_lab_to_rgb  = parallel.gpu.CUDAKernel('lab_to_rgb.ptx', 'lab_to_rgb.cu');
    kernel_lab_to_rgb.ThreadBlockSize = block;
    kernel_lab_to_rgb.GridSize = grid;

    kernel_rgb_to_lab = parallel.gpu.CUDAKernel('rgb_to_lab.ptx', 'rgb_to_lab.cu');
    kernel_rgb_to_lab.ThreadBlockSize = block;
    kernel_rgb_to_lab.GridSize = grid;

    kernel_find_border = parallel.gpu.CUDAKernel('update_seg.ptx', 'update_seg.cu', 'find_border_pixels');
    kernel_find_border.ThreadBlockSize = block;
    kernel_find_border.GridSize = grid;

    kernel_clear_fields = parallel.gpu.CUDAKernel('update_param.ptx', 'update_param.cu', 'clear_fields');
    kernel_clear_fields.ThreadBlockSize = block;
    kernel_clear_fields.GridSize = grid_sp;

    kernel_sum_by_label = parallel.gpu.CUDAKernel('update_param.ptx', 'update_param.cu', 'sum_by_label');
    kernel_sum_by_label.ThreadBlockSize = block;
    kernel_sum_by_label.GridSize = grid;

    kernel_calculate_mu_and_sigma = parallel.gpu.CUDAKernel('update_param.ptx', 'update_param.cu', 'calculate_mu_and_sigma');
    kernel_calculate_mu_and_sigma.ThreadBlockSize = block;
    kernel_calculate_mu_and_sigma.GridSize = grid_sp;

    kernel_clear_fields2 = parallel.gpu.CUDAKernel('update_param.ptx', 'update_param.cu', 'clear_fields_2');
    kernel_clear_fields2.ThreadBlockSize = block;
    kernel_clear_fields2.GridSize = grid_sp;

    kernel_sum_by_label2 = parallel.gpu.CUDAKernel('update_param.ptx', 'update_param.cu', 'sum_by_label_2');
    kernel_sum_by_label2.ThreadBlockSize = block;
    kernel_sum_by_label2.GridSize = grid;

    kernel_calculate_mu = parallel.gpu.CUDAKernel('update_param.ptx', 'update_param.cu', 'calculate_mu');
    kernel_calculate_mu.ThreadBlockSize = block;
    kernel_calculate_mu.GridSize = grid_sp;

    kernel_update_seg_subset = parallel.gpu.CUDAKernel('update_seg.ptx', 'update_seg.cu', 'update_seg_subset');
    kernel_update_seg_subset.ThreadBlockSize = block;
    kernel_update_seg_subset.GridSize = grid;
    
       
    kernel_get_cartoon = parallel.gpu.CUDAKernel('get_cartoon.ptx', 'get_cartoon.cu');
    kernel_get_cartoon.ThreadBlockSize = block;
    kernel_get_cartoon.GridSize = grid;
end
