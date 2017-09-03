close all, clear all;

% Visualise point clouds generated by model in form of rectilinear
% projections and ODS projections.

%% Project front to rectilinear.
X = load_xyz('pc.xyz');
[rectilinear, rectilinear_Z] = equirectangular_to_rectilinear([1.0, 2.0, 0.0, 0.0], X, [256, 512]);
figure, imshow(rectilinear);
figure, imshow(rectilinear_Z / 4);
imwrite(rectilinear, 'rectilinear.jpg');

%% Render ODS from .xyz file.
[pc_Y, pc_dX] = render_ods('pc.xyz', [512, 1024]);
figure, imshow(pc_Y);
figure, imshow(4 * pc_dX);
imwrite(pc_Y, 'pc_ods.jpg');

%% Render ODS from separate RGB and depth map.
[Y, dX] = render_ods('rgb.jpg', 'depth.npy', [512, 1024]);
figure, imshow(Y);
figure, imshow(4 * dX);
imwrite(Y, 'ods.jpg');