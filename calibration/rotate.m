function [output_image] = rotate(input_image, euler_angles)
%rotate Rotate the equirectangular image with the given rotation in the
% format of Euler angles.

    % Image width and height.
    w = size(input_image, 2);
    h = size(input_image, 1);
    output_image = zeros(h, w, 3);

    % Generate latitude and longitude grid.
    [S, T] = lat_long_grid(w, h);
    
    % Convert to Cartesian coordinates.
    X = cat(3, cos(T) .* sin(S), sin(T), cos(T) .* cos(S));
    X = reshape(permute(X, [3, 1, 2]), 3, []);
    
    % Create rotation matrix using inverse warp.
    euler_angles = - euler_angles;
    R = [
            cos(euler_angles(3)), - sin(euler_angles(3)), 0;
            sin(euler_angles(3)), cos(euler_angles(3)), 0;
            0, 0, 1
        ];
    R = [
            cos(euler_angles(2)), 0, sin(euler_angles(2));
            0, 1, 0;
            - sin(euler_angles(2)), 0, cos(euler_angles(2))
        ] * R;
    R = [
            1, 0, 0;
            0, cos(euler_angles(1)), - sin(euler_angles(1));
            0, sin(euler_angles(1)), cos(euler_angles(1))
        ] * R;
        
    % Rotate and convert back to UV coordinates.
    X = reshape(R * X, [3, h, w]);
    S = atan2(X(1, :, :), X(3, :, :));
    T = atan2(X(2, :, :), sqrt(X(1, :, :) .^ 2 + X(3, :, :) .^ 2));
    U = 1 + (w - 1) * (pi + S) / (2 * pi);
    V = 1 + (h - 1) * (pi / 2 + T) / pi;
    
    % Use bilinear sampling.
    output_image(:, :, 1) = interp2(input_image(:, :, 1), U, V, 'linear', 0);
    output_image(:, :, 2) = interp2(input_image(:, :, 2), U, V, 'linear', 0);
    output_image(:, :, 3) = interp2(input_image(:, :, 3), U, V, 'linear', 0);
end