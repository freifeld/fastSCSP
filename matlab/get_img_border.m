function image_border = get_img_border (sp, image)
    sp.border_cpu = gather(sp.border_gpu);
    sp.border_cpu = reshape(sp.border_cpu,sp.dimy,sp.dimx);
    [r,c] = find(sp.border_cpu==1);
    R = image(:, :, 1);
    G = image(:, :, 2);
    B = image(:, :, 3);
    indices = sub2ind(size(R), r, c);
    R(indices) = 255;
    G(indices) = 0;
    B(indices) = 0;
    image_border = cat(3,R,G,B);
end