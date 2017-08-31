function [Y, dX] = ods_depth(image, depth_map, distance)
%ods_depth Convert RGB and depth map into point cloud, then render ODS. If
% input is distance rather than xz plane distance, set distance to true.
    if nargin < 3
        distance = 0;
    end
    
    % Get output image size.
    [h, w, ~] = size(image);
    
    % Resample depth map and convert to point cloud.
    depth_map = imresize(depth_map, [h, w]);
    [S, T] = meshgrid(linspace(-pi, pi, w), linspace(-pi / 2 + 1e-6, pi / 2 - 1e-6, h));
    
    x = depth_map .* sin(S);
    y = depth_map .* tan(T);
    z = depth_map .* cos(S);
    
    if distance
        x = cos(T) .* x;
        y = cos(T) .* y;
        z = cos(T) .* z;
    end
    
    X = reshape(permute(cat(3, cat(3, x, y, z), image), [2, 1, 3]), [], 6);
    
    if nargout > 1
        [Y, dX] = ods_pc(X, [h, w]);
    else
        Y = ods_pc(X, [h, w]);
    end
    
end