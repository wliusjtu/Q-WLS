clear; close all;
addpath('./funs')

name = 'art';
ufactor = 2;

fprintf('\n************************** %s %dx upsampling ***************************\n', name, ufactor);

%% images
depth = double(imread(['./imgs/guided_depth_upsampling/', name, '/depth_', num2str(log2(ufactor)), '_n.png']));
img_guide = double(imread(['./imgs/guided_depth_upsampling/', name, '/', name, '_color.png']));

[m, n, ~] = size(img_guide);
depth =imresize(depth, [m, n]);


%% parameters
lambda = 0.15;  % 0.15/0.5/10/30 for 2x/4x/8x/16x upsampling
sigmaR = 15/255; % 15/255  /  10/255  /  5/255  /  5/255 for 2x/4x/8x/16x upsampling
sigmaR_dual = 5/255; % 5/255  /  5/255  /  3/255  /  3/255 for 2x/4x/8x/16x upsampling
r = 7;
step = 7;
iter = 20;

%% smooth
img_guide_dual = depth;  % initialize the dual guidance weight image

time_start = tic;
res = depth;
for i = 1: iter

    res = QWLS_DualGuide(res, img_guide, img_guide_dual, lambda, sigmaR, sigmaR_dual, r, step);
    img_guide_dual = res;
    res(res > 255) = 255;
    res(res < 0) = 0;

end
time_elapsed = toc(time_start);
fprintf('Elapsed time is %f seconds\n', time_elapsed)

%% compute MAE and show results
depth_gt = double(imread(['./imgs/guided_depth_upsampling/', name, '/', name, '_big.png']));

mae = mean(mean(abs(res - depth_gt)));
fprintf('MEA for the result is %f\n', mae);

%% show the result
figure; imshow(uint8([depth, res]))








