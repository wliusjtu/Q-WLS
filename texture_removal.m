% clear; close all;
addpath('./funs')

%% load image
img =im2double(imread('./imgs/texture_removal/crossstitch1.png'));

%% parameters 
lambda = 0.00085; 
sigma = 3; % this parameter should be smaller if small structures need to be preserved
sharpness = 0.0001;
iter = 4;
r = 3;
step = 3;

%% smooth and show the result
res = QWLS_RTV(img, lambda, sigma, sharpness, iter, r, step);

figure; imshow([img, res]);