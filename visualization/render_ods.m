function [Y, dX] = render_ods(varargin)
%render_ods Render ODS panorama from either .xyz or separate RGB and depth
% files.
    addpath('npy-matlab');
    
    if length(varargin) == 2
        pc_filename = varargin{1};
        X = load_xyz(pc_filename);
        
        if nargout > 1
            [Y, dX] = ods_pc(X, varargin{2});
        else
            Y = ods_pc(X, varargin{2});
        end
    elseif length(varargin) == 3
        rgb_filename = varargin{1};
        depth_filename = varargin{2};
        
        rgb = imresize(im2double(imread(rgb_filename)), varargin{3});
        depth = double(read_npy(depth_filename));
        
        if nargout > 1
            [Y, dX] = ods_depth(rgb, depth, 1);
        else
            Y = ods_depth(rgb, depth, 1);
        end
    end
end