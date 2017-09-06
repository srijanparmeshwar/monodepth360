%% Upsample depth map to create a point cloud.
% Input filenames.
rgb_filename = '';
depth_filename = '';
% Output filename.
pc_filename = '';

% Requested height and width.
h = 2048;
w = 4096;

%% Resize RGB and depth map.
rgb = imresize(im2double(imread(rgb_filename)), h, w]);
depth_map = double(read_npy(depth_filename));

% Get output image size.
[h, w, ~] = size(rgb);
    
% Resample depth map and convert to point cloud.
depth_map = imresize(depth_map, [h, w]);
[S, T] = meshgrid(linspace(-pi, pi, w), linspace(-pi / 2 + 1e-6, pi / 2 - 1e-6, h));

% Backproject to point cloud.
x = depth_map .* sin(S) .* cos(T);
y = depth_map .* sin(T);
z = depth_map .* cos(S) .* cos(T);

% Concatenate colours and positions.
p = cat(3, x, y, z, rgb);

crop = 0;
if crop
    p = reshape(p(144:(end - 144), :, :), [], 6);
else
    p = reshape(p, [], 6);
end

% Write point cloud to ASCII file delimited by spaces.
dlmwrite(pc_filename, p, ' ');