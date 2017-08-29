% addpath('OpticalFlow');
% addpath('OpticalFlow/mex');
% 
% top = im2double(imread('top.jpg'));
% bottom = im2double(imread('bottom.jpg'));
% 
% top_ds = imresize(top, 0.033333);
% bottom_ds = imresize(bottom, 0.033333);
% 
% tic;
% r_opt = optimise(top_ds, bottom_ds);
% toc;
% 
% top_rectified = rotate(top, r_opt);
% imwrite(top_rectified, 'top_rectified.jpg')

% top_display = imresize(top, 0.25);
% bottom_display = imresize(bottom, 0.25);
% top_rectified_display = imresize(top_rectified, 0.25);
% 
% [vx, vy] = flow(top_display, bottom_display);
% [vx_rectified, vy_rectified] = flow(top_rectified_display, bottom_display);

cat1 = cat(1, top_display, bottom_display);
cat2 = cat(1, top_rectified_display, bottom_display);
x1 = 240:20:720;
x2 = zeros(1, length(x1));
x2_r = zeros(1, length(x1));
y1 = repmat(270, 1, length(x1));
y2 = zeros(1, length(y1));
y2_r = zeros(1, length(y1));
for index = 1:length(x1)
    x2(index) = max(min(vx(y1(index), x1(index)) + x1(index), 960), 0);
    y2(index) = max(min(vy(y1(index), x1(index)) + y1(index) + 540, 1080), 0);
    x2_r(index) = max(min(vx_rectified(y1(index), x1(index)) + x1(index), 960), 0);
    y2_r(index) = max(min(vy_rectified(y1(index), x1(index)) + y1(index) + 540, 1080), 0);
end
lines = [x1', y1', x2', y2'];
lines_r = [x1', y1', x2_r', y2_r'];
lines_gt = [x1', y1', x1', y2'];
cat1 = insertShape(cat1, 'Line', lines_gt, 'Color', 'red');
cat2 = insertShape(cat2, 'Line', lines_gt, 'Color', 'red');
cat1 = insertShape(cat1, 'Line', lines, 'Color', 'yellow');
cat2 = insertShape(cat2, 'Line', lines_r, 'Color', 'yellow');
figure, imshow(cat1);
figure, imshow(cat2);
figure, imshow(vx);
figure, imshow(vx_rectified);

% figure, hold on;
% axis off;
% subplot(2, 1, 1);
% imagesc(top_display);
% set(gca, 'xtick', [], 'ytick', []);
% set(gca, 'Position', [0 0.5 1 0.5]);
% subplot(2, 1, 2);
% imagesc(bottom_display);
% set(gca, 'xtick', [], 'ytick', []);
% set(gca, 'Position', [0 0 1 0.5]);
% xs = 240:20:720;
% for x = xs
%     y1 = 135 / 540;
%     y2 = vy(135, x) + y1;
%     line([x, x], 0.5 * [y1, y2]);
% end
% hold off;