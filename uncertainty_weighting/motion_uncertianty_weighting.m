
function uncertainty_map = motion_uncertianty_weighting(original_img, global_motion, smap)

% smap = imread('./DC_original_smap_norm_res/1.jpg');
smap = mat2gray(smap);
[row col] = size(smap);


%% calculate the uncertainty for the global motion and contrast.
[row col] = size(original_img);
row_blk = row/8;
col_blk = col/8;
% tmp_contrast = zeros(row_blk*8, col_blk*8);

img_col = im2col(original_img, [8 8], 'distinct');
img_std = std(double(img_col));
img_mean = mean(img_col);
tmp_contrast_one = img_std./(6.0 + img_mean);
tmp_contrast = repmat(tmp_contrast_one, 64, 1);
contrast_res = col2im(tmp_contrast, [8 8], [row_blk*8 col_blk*8], 'distinct');

if global_motion == 0
    global_motion_res = 0;
else
    global_motion_res = sqrt(global_motion(:, :, 1).^2 + global_motion(:, :, 2).^2);
end

u_global_motion = 0.5 + 0.5*log(2*pi) + log(1 + global_motion_res./0.32) + 2.5*log(1 + contrast_res./0.07);

% u_spatial_dist = mat2gray(u_spatial_dist);
% u_connectedness = mat2gray(u_connectedness);
u_global_motion = mat2gray(u_global_motion);


uncertainty_map = u_global_motion;
uncertainty_map = mat2gray(uncertainty_map);
% imshow(weighting_map);

end