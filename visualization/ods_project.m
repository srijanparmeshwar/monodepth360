function [S_L, T_L, S_R, T_R] = ods_project(X, r, epsilon)
%ods_project Project point cloud X into latitudes and longitudes given by
% viewing circle with radius r.
    if nargin < 3
        epsilon = 1e-6;
    end

    x = X(:, 1);
    y = X(:, 2);
    z = X(:, 3);
    
    d = sqrt(x .^ 2 + z .^ 2);
    % Clip minimum distance to be outside viewing circle.
    A = max(d, 4 * r);
    alpha = atan2(z , x + epsilon);
    
    % Left eye latitudes and longitudes.
    S_L = mod(asin(r ./ A) - alpha - pi / 2, 2 * pi) - pi;
    T_L = mod(atan2(y, sqrt((x - r .* sin(S_L)) .^ 2 + (z - r .* cos(S_L)) .^ 2) + epsilon) + pi / 2, pi) - pi / 2;
    
    % Right eye latitudes and longitudes.
    S_R= mod(asin(-r ./ A) - alpha - pi / 2, 2 * pi) - pi;
    T_R = mod(atan2(y, sqrt((x - r .* sin(S_R)) .^ 2 + (z - r .* cos(S_R)) .^ 2) + epsilon) + pi / 2, pi) - pi / 2;
end