function [image_overlaid] = get_img_overlaid(sp, params, ...
                            kernel_lab_to_rgb,kernel_get_cartoon)

    img = gpuArray(zeros(3,sp.nPts,'int32'));
    mu_i_rgb_gpu = feval(kernel_lab_to_rgb,params.mu_i_gpu, sp.nSps);
    [~,~, img] = feval(kernel_get_cartoon, sp.seg_gpu, mu_i_rgb_gpu, img, 3, sp.nPts);
    img_cpu = gather(img);

    image_overlaid = zeros(sp.dimy, sp.dimx, 3);
    % TODO: change this...
    for i = 1:sp.dimy % this row
        for j = 1:sp.dimx %this column
            index = sp.dimy*(j-1) + i;
            image_overlaid(i,j,1) = img_cpu(1,index) ;
            image_overlaid(i,j,2) = img_cpu(2,index) ;
            image_overlaid(i,j,3) = img_cpu(3,index) ;
        end
    end
    image_overlaid = uint8(image_overlaid);
end