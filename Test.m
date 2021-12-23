clear
clc
close all

lambda = 1;
sigmaR = 1.2;
r = 1;
step = 2;
weightChoice = 0;

img = double(imread('../Bishapur_zan.jpg'));
img = img / max(img(:));
img_guide = img ;
% img_guide = log(img + 0.00001);
% img_guide = img_guide / (max(img_guide(:)) - min(img_guide(:)));

% result = mexQWLS_r1c1_single_v2(img(:, :, 1), img_guide(:, :, 1), lambda, sigmaR, r, step, weightChoice);
% result = mexQWLS_r1c1_double_v2(img(:, :, 1), img_guide(:, :, 1), lambda, sigmaR, r, step, weightChoice);
% result = mexQWLS_r1c3_single_v2(img, img_guide, lambda, sigmaR, r, step, weightChoice);
result = mexQWLS_r1c3_double_v2(img, img_guide, lambda, sigmaR, r, step, weightChoice);
% result2 = mexQWLS_single_v2(img, img_guide, lambda, sigmaR, r, step, weightChoice);
result = mexQWLS_double_v2(img, img_guide, lambda, sigmaR, r, step, weightChoice);

diff = img - result;

figure; imshow(result)
figure; imshow(img + 3 * diff)
% figure; imshow(img)
% imwrite(img + 5 * diff, 'a.png')
% figure; imshow(uint8(result))
% figure; imshow(uint8(img + 5 * diff))


