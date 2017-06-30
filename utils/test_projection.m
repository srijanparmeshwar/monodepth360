% Test points.
N = 1000;
min_length = 2;
test_heading = 2 * pi * rand(1, N);
test_length = min_length + randn(1, N) .^ 4;
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

figure;
f = @(x) x;
g = @(x) 1 ./ x;
scatter(f(d_L), g(L(1, :) - R(1, :) - pi));

