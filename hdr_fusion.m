% clc;
% clear;

% smap_dir = 'ethyma_exposure_stack';
% output_dir = 'ethyma_output';
% folder = 'covsal';
% addpath('../fang_uncertainty_weighting');

if ~exist('image_dir', 'var')
    % use image_dir to find exposure saliency maps
    print('image_dir variable not exists')
end

if ~exist('smap_dir', 'var')
    print('smap_dir variable not exists')
end

if ~exist('output_dir', 'var')
    print('output_dir variable not exists')
end

if ~exist('folder', 'var') % model name
    print('folder variable not exists')
end

[~,smap_basename]=fileparts(smap_dir)
smap_dir = fullfile(smap_dir, folder);
% output_dir = fullfile(output_dir, strcat(smap_basename, '_', folder));
% mkdir(output_dir);
img_path_list = dir(smap_dir);
scale = 2;

image_name_list = dir(image_dir);

for i=3:length(image_name_list)
    image_name = image_name_list(i).name;
    [~, image_prefix, ext] = fileparts(image_name);

    smap_name_list = dir(fullfile(smap_dir, strcat(image_prefix,'_*')))
    if isempty(smap_name_list)
        % print('empty smap list')
        continue
    end
    if length(smap_name_list)==1
        % smap = imwrite(smap_name_list)
        smap = imread(fullfile(smap_dir,smap_name_list(1).name));
        imwrite(smap, fullfile(output_dir, strcat(image_prefix,'.jpg')));
        continue
    end

    smap = imresize(imread(fullfile(smap_dir,smap_name_list(1).name)), 1/scale);
    if size(smap,3)==3
        smap = rgb2gray(smap);
    end
    [row, col] = size(smap);
    smap_list = zeros([row col length(smap_name_list)]);
    smap_wm_list = zeros([row col length(smap_name_list)]);
    
    for j=1:length(smap_name_list)

        smap = imresize(imread(fullfile(smap_dir,smap_name_list(j).name)), 1/scale);
        if size(smap,3)==3
            smap = rgb2gray(smap);
        end
        smap_wm = addition_uncertainty_weighting(smap);

        smap_list(:,:,j)=smap;
        smap_wm_list(:,:,j) = smap_wm;
    end

    all_wm = sum(smap_wm_list, 3);

    fusion_map_list = zeros([row col length(smap_name_list)]);
    
    for j=1:length(smap_name_list)
        smap = smap_list(:, :, j);
        other_wm = all_wm - smap_wm_list(:, :, j);
        fusion_map = other_wm.*im2double(smap);
        fusion_map_list(:, :, j) = fusion_map;
    end

    fusion_map = sum(fusion_map_list, 3)./all_wm;
    fusion_map = fusion_map/max(fusion_map(:));
    %figure('Name', prefix);imshow(fusion_map)
    %pause()
    fusion_map = imresize(fusion_map, scale);
    imwrite(fusion_map, fullfile(output_dir, strcat(image_prefix,'.jpg')));
end