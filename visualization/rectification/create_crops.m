% Load images.
original = im2double(imread('original.jpg'));
rectified = im2double(imread('rectified.jpg'));

% Select crops and write to files.
y_crop1 = 1401:1600;
x_crop1 = 2001:2400;

y_crop2 = 1001:1200;
x_crop2 = 1601:2000;

original_crop1 = original(y_crop1, x_crop1, :);
rectified_crop1 = rectified(y_crop1, x_crop1, :);

original_crop2 = original(y_crop2, x_crop2, :);
rectified_crop2 = rectified(y_crop2, x_crop2, :);

imwrite(original_crop1, 'oc1.jpg');
imwrite(rectified_crop1, 'rc1.jpg');

imwrite(original_crop2, 'oc2.jpg');
imwrite(rectified_crop2, 'rc2.jpg');