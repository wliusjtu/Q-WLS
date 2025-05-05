clear; close all;
addpath('./funs')

%% images
img = double(imread('./imgs/flash_no_flash_filtering/books_rgb.png'));
img_guide = double(imread('./imgs/flash_no_flash_filtering/books_nir.png'));

%% parameters (EP & SP mode)
lambda = 200;
sigmaR = 4/255;
sigmaR_dual = 4/255;
r = 1;
step = 2;
iter = 10;

%% smooth
img_guide_dual = ones(size(img));  % initialize the dual guidance weight image

time_start = tic;
for i = 1: iter

    res = QWLS_DualGuide(img, img_guide, img_guide_dual, lambda, sigmaR, sigmaR_dual, r, step);
    img_guide_dual = res;
   
end
time_elapsed = toc(time_start);
fprintf('Elapsed time is %f seconds\n', time_elapsed)

%% show the result
figure; imshow(uint8([img, res]))
