
function uncertainty_map = addition_uncertainty_weighting(smap)

% smap = imread('./DC_original_smap_norm_res/1.jpg');
smap = mat2gray(smap);
[row col] = size(smap);

%% the probibility function for spatial distance is firstly calculated
%{
%% calculate the expected spatial value for row
row_array = 1:row;
row_array = row_array';
row_matrix = repmat(row_array, 1, col);
expected_row = round(sum(sum(row_matrix.*smap))/sum(sum(smap)));

%% calculate the expected spatial value for col
col_array = 1:col;
col_matrix = repmat(col_array, row, 1);
expected_col = round(sum(sum(col_matrix.*smap))/(sum(sum(smap))));

%% calculate the distance between each pixel and the expected spatial location
dis_row = (row_matrix - expected_row).^2;
dis_col = (col_matrix - expected_col).^2;
spatial_dist = round(sqrt(dis_row + dis_col));

%% probability function for spatial distance
p_spatial_dist = exp(-(spatial_dist.^2./92^2));

%% uncertainty for the probability
u_spatial_dist = -(p_spatial_dist.*log2(p_spatial_dist) + (1-p_spatial_dist).*log2(1-p_spatial_dist));
% u_spatial_dist = abs(u_spatial_dist);
%}

%% the following calculates the probility function for connectedness

%% we ignore the boundary of the smap
used_smap = smap(2:end-1, 2:end-1);
% [used_row used_col] = size(used_smap);

%% the 8 neighboring pixel set for each image pixel
used_smap_array(:, :, 1) = smap(1:end-2, 1:end-2);
used_smap_array(:, :, 2) = smap(1:end-2, 2:end-1);
used_smap_array(:, :, 3) = smap(1:end-2, 3:end);
used_smap_array(:, :, 4) = smap(2:end-1,1:end-2);
used_smap_array(:, :, 5) = smap(2:end-1, 3:end);
used_smap_array(:, :, 6) = smap(3:end, 1:end-2);
used_smap_array(:, :, 7) = smap(3:end, 2:end-1);
used_smap_array(:, :, 8) = smap(3:end, 3:end);

%% probability function for connectedness
%used_smap_connected_num(:, :) = round(sum(used_smap_array(:, :, :), 3));
used_smap_connected_num(:, :) = sum(used_smap_array(:, :, :), 3);
smap_connected_num = zeros(row, col);
smap_connected_num(2:end-1, 2:end-1) = used_smap_connected_num;
p_connectedness = exp(-(smap_connected_num-8).^2./4^2);

%% uncertainty for the probability
u_connectedness = -(p_connectedness.*log2(p_connectedness) + (1-p_connectedness).*log2(1-p_connectedness));
% u_connectedness = abs(u_connectedness);

% figure('Name', 'u_spatial_dist');imshow(u_spatial_dist);
% figure('Name', 'u_connectedness');imshow(u_connectedness);
%% adaptive weighting map
uncertainty_map = u_connectedness;
% uncertainty_map = u_spatial_dist + u_connectedness;
uncertainty_map = mat2gray(uncertainty_map);
% imshow(weighting_map);

end