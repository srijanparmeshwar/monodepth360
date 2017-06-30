% Test points.
N = 1024;
min_length = 2;
test_heading = 2 * pi * rand(1, N);
test_length = min_length + randn(1, N) .^ 12;
x = test_length .* cos(test_heading);
y = randn(1, N);
z = test_length .* sin(test_heading);
X = [x; y; z];
% Centre of camera.
C = [0; 0; 0];
% Interpupillary distance.
r = 0.65;

% Setup canvas.
figure;
hold on;

% Plot test point.
scatter(x, z);
% scatter3(x, y, z);

% Plot camera.
theta = linspace(0, 2 * pi, 60);
plot(C(1) + r * sin(theta), C(2) + r * cos(theta));
% plot3(C(1) + r * sin(theta), zeros(size(theta)), C(2) + r * cos(theta));

% Project test points and calculate vectors to points.
[L, R] = ods_project(X, C, r);
C_L = bsxfun(@plus, C, [r * cos(L(1, :)); zeros(size(L(1, :))); r * sin(L(1, :))]);
C_R = bsxfun(@plus, C, [r * cos(R(1, :)); zeros(size(R(1, :))); r * sin(R(1, :))]);

% Distances.
d_L = zeros(1, N);
d_R = zeros(1, N);

for index = 1:size(X, 2)
    start_x_L = C_L(1, index);
    start_z_L = C_L(3, index);
    
    start_x_R = C_R(1, index);
    start_z_R = C_R(3, index);
    
    end_x = x(index);
    end_z = z(index);
    
    d_L(index) = sqrt((end_x - start_x_L) ^ 2 + (end_z - start_z_L) ^ 2);
    d_R(index) = sqrt((end_x - start_x_R) ^ 2 + (end_z - start_z_R) ^ 2);
    
    plot(linspace(start_x_L, end_x, 2), linspace(start_z_L, end_z, 2));
    plot(linspace(start_x_R, end_x, 2), linspace(start_z_R, end_z, 2));
    
%     plot3(linspace(start_x_L, end_x, 2), linspace(0, y, 2), linspace(start_z_L, end_z, 2));
%     plot3(linspace(start_x_R, end_x, 2), linspace(0, y, 2), linspace(start_z_R, end_z, 2));
%     
%     plot3(linspace(start_x_L, end_x, 2), linspace(0, 0, 2), linspace(start_z_L, end_z, 2));
%     plot3(linspace(start_x_R, end_x, 2), linspace(0, 0, 2), linspace(start_z_R, end_z, 2));
end

f = @(x) x;
g = @(x) 1 ./ x;
figure, scatter(f(d_L), g(L(1, :) - R(1, :) - pi));

r = linspace(0, 1, 256);
g = linspace(0, 1, 256);
[R, G] = meshgrid(r, g);
src_image = cat(3, R, G, zeros(256, 256));
r_flat = reshape(src_image(:, :, 1), 1, []);
g_flat = reshape(src_image(:, :, 2), 1, []);
b_flat = reshape(src_image(:, :, 3), 1, []);
figure, imshow(src_image);

scene = src_image;
x_t = reshape(2 * (2 * scene(:, :, 1) - 1), 1, []);
y_t = reshape(2 * (2 * scene(:, :, 2) - 1), 1, []);
z_t = 5 * ones(1, numel(src_image) / 3);
figure, scatter3(x_t, y_t, z_t, [], cat(1, r_flat, g_flat, b_flat)');

[m, n] = ods_project(cat(1, x_t, y_t, z_t), [0; 0; 0], 0.65);

dst_image_L = zeros(256, 256, 3);
dst_image_R = zeros(256, 256, 3);
for index = 1:(256 * 256)
    u_L = 1 + (255 / (2 * pi)) * mod(m(1, index) - pi, 2 * pi);
    v_L = 1 + (255 / (pi / 2)) * (mod(m(2, index), pi / 2));
    u_R = 1 + (255 / (2 * pi)) * mod(n(1, index), 2 * pi);
    v_R = 1 + (255 / (pi / 2)) * (mod(n(2, index), pi / 2));
    
    u_L = round(u_L);
    v_L = round(v_L);
    dst_image_L(v_L, u_L, 1) = r_flat(index);
    dst_image_L(v_L, u_L, 2) = g_flat(index);
    dst_image_L(v_L, u_L, 3) = b_flat(index);
    
    u_R = round(u_R);
    v_R = round(v_R);
    dst_image_R(v_R, u_R, 1) = r_flat(index);
    dst_image_R(v_R, u_R, 2) = g_flat(index);
    dst_image_R(v_R, u_R, 3) = b_flat(index);
end

figure, imshow(dst_image_L);
figure, imshow(dst_image_R);