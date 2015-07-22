function [sp, params] = update_param(lab_image_gpu, option,params, gpu_helper,...
                                    sp,...
                                    kernel_clear_fields, kernel_sum_by_label,kernel_calculate_mu_and_sigma, ...
                                    kernel_clear_fields2, kernel_sum_by_label2, kernel_calculate_mu)  
% calculate sp parameters
    if option.calc_cov
        [params.counts_gpu, gpu_helper.log_count, ...
            gpu_helper.mu_i_sum, gpu_helper.mu_s_sum,...
            params.mu_i_gpu, params.mu_s_gpu,...
            gpu_helper.sigma_s_sum] = ...
        feval(kernel_clear_fields, params.counts_gpu, gpu_helper.log_count,...
                gpu_helper.mu_i_sum, gpu_helper.mu_s_sum, ...
                params.mu_i_gpu, params.mu_s_gpu,...
                gpu_helper.sigma_s_sum,...
                sp.dim_i, sp.nSps);

       [~, sp.seg_gpu, params.counts_gpu, ...
           gpu_helper.mu_i_sum, gpu_helper.mu_s_sum, gpu_helper.sigma_s_sum] = ... 
       feval(kernel_sum_by_label, lab_image_gpu, sp.seg_gpu, params.counts_gpu, ...
              gpu_helper.mu_i_sum, gpu_helper.mu_s_sum, gpu_helper.sigma_s_sum,...
              sp.dimy, sp.dimx, sp.dim_i, sp.nPts);

       [ params.counts_gpu, gpu_helper.log_count,...
            gpu_helper.mu_i_sum, gpu_helper.mu_s_sum,...
            params.mu_i_gpu, params.mu_s_gpu, gpu_helper.sigma_s_sum, ...
            params.prior_sigma_s_sum_gpu,...
            params.Sigma_s_gpu, params.logdet_Sigma_s_gpu, params.J_s_gpu] =...
       feval( kernel_calculate_mu_and_sigma, params.counts_gpu, gpu_helper.log_count,...
            gpu_helper.mu_i_sum, gpu_helper.mu_s_sum,...
            params.mu_i_gpu, params.mu_s_gpu, gpu_helper.sigma_s_sum, ...
            params.prior_sigma_s_sum_gpu,...
            params.Sigma_s_gpu, params.logdet_Sigma_s_gpu, params.J_s_gpu, ...
            option.prior_count, sp.dim_i, sp.nSps);
        
    else
         [params.counts_gpu, gpu_helper.mu_i_sum, gpu_helper.mu_s_sum] = ...
         feval(kernel_clear_fields2, params.counts_gpu,...
             gpu_helper.mu_i_sum, gpu_helper.mu_s_sum,...
             sp.dim_i, sp.nSps);         
         
        [~, ~, params.counts_gpu, ...
             gpu_helper.mu_i_sum, gpu_helper.mu_s_sum] = ...
        feval(kernel_sum_by_label2, lab_image_gpu, seg_gpu, params.counts_gpu, ...
             gpu_helper.mu_i_sum, gpu_helper.mu_s_sum,...
             sp.dimy, sp.dimx, sp.dim_i, sp.nPts);
       
         [ params.counts_gpu,gpu_helper.mu_i_sum, gpu_helper.mu_s_sum,...
            params.mu_i_gpu, params.mu_s_gpu] = ... 
        feval(kernel_calculate_mu, params.counts_gpu, ...
            gpu_helper.mu_i_sum, gpu_helper.mu_s_sum,...
            params.mu_i_gpu, params.mu_s_gpu,...
            sp.dim_i, sp.nSps); 
    end
end