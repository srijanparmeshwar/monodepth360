function [S_L, T_L, S_R, T_R] = ods_project(X, r, epsilon)
    if nargin < 3
        epsilon = 1e-6;
    end

    x = X(:, 1);
    y = X(:, 2);
    z = X(:, 3);
    
    d = sqrt(x .^ 2 + z .^ 2);
    A = max(d, r);
    alpha = atan2(z , x + epsilon);
    
    S_L = mod(asin(r ./ A) - alpha - pi / 2, 2 * pi) - pi;
    T_L = mod(atan2(y, sqrt((x - r .* sin(S_L)) .^ 2 + (z - r .* cos(S_L)) .^ 2) + epsilon) + pi / 2, pi) - pi / 2;
    
    S_R= mod(asin(-r ./ A) - alpha - pi / 2, 2 * pi) - pi;
    T_R = mod(atan2(y, sqrt((x - r .* sin(S_R)) .^ 2 + (z - r .* cos(S_R)) .^ 2) + epsilon) + pi / 2, pi) - pi / 2;
end