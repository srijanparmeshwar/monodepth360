equirectangular_image = imresize(im2double(imread('../monodepth/equirectangular.jpg')), [1024, 2048]);

rx = [0, 0, 0.5];
ry = [-0.5, 0.5, 0];
rz = [0, 0, 0.5];

for index = 1:3
    rotated_image = rotate(equirectangular_image, [rx(index), ry(index), rz(index)]);
    imwrite(rotated_image, sprintf('rotate_test_%d.jpg', index - 1));
end