clear; close all;
addpath('./funs')

%% images
img = double(imread('./imgs/detail_enhancement/bird1.png'));
img_guide = img;

%% parameters
lambda = 2;
alpha = 1.2;
r = 1;
step = 1;
weightChoice = 1;  % fractional guidance weight in Eq. (6)

%% smooth the image
res = QWLS(img, img_guide, lambda, alpha, r, step, weightChoice);

%% enhance details and show
diff = img - res;
img_enhanced = img + 3 * diff;
figure; imshow(uint8([img, res, img_enhanced]))
