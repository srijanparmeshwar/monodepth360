function [Y, Z] = equirectangular_to_rectilinear(K, X, output_size)
%equirectangular_to_rectilinear Project front view to rectilinear using
% given intrinsic matrix.
    % Reshape if point cloud. Assumes images are twice as wide as they are
    % tall i.e. W = 2 * H.
    if length(size(X)) < 3
        ih = round(sqrt(size(X, 1) / 2));
        iw = 2 * ih;
        X = permute(reshape(X, [iw, ih, 6]), [2, 1, 3]);
    end

    h = output_size(1);
    w = output_size(2);
    
    fx = K(1);
    fy = K(2);
    cx = K(3);
    cy = K(4);
    
    [ui, vi] = meshgrid(linspace(-1, 1, w), linspace(-1, 1, h)); 
    
    x = - (ui - cx) / fx;
    y = (vi - cy) / fy;
    z = ones(h, w);
    
    S = - atan2(x, z);
    T = atan2(y, sqrt(x .^ 2.0 + z .^ 2.0));
    
    % Normalize to [0, 1].
    u = mod(S / (2.0 * pi) + 0.5, 1.0);
    v = mod(pi / 2 + T / pi, 1.0);
    
    % Scale to width and height.
    U = 1 + (size(X, 2) - 1) * u;
    V = 1 + (size(X, 1) - 1) * v;
    
    % Rectilinear projection of RGB.
    Y = zeros(h, w, 3);
    Y(:, :, 1) = interp2(X(:, :, 4), U, V, 'linear', 0);
    Y(:, :, 2) = interp2(X(:, :, 5), U, V, 'linear', 0);
    Y(:, :, 3) = interp2(X(:, :, 6), U, V, 'linear', 0);
    
    % Rectilinear projection of Z buffer.
    Z = interp2(X(:, :, 3), U, V, 'linear', 0);
end