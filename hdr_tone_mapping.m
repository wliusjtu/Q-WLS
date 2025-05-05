% this code is based on the HDR tone mapping framework developed by Zeev Farbman et al. Their code can be downloed at: https://www.cs.huji.ac.il/~danix/epd/

clear; close all
addpath('./funs')

%% load the hdr image
hdr = double(hdrread('./imgs/hdr_tone_mapping/dani_synagogue.hdr'));
I = 0.2989*hdr(:,:,1) + 0.587*hdr(:,:,2) + 0.114*hdr(:,:,3);
logI = log(I+eps);

%% parameters
lambda = 10;
alpha = 1.2;
r = 1;
step = 1;
weightChoice = 1;  % fractional guidance weight in Eq. (6)

%% smooth and process
base = QWLS(logI, logI, lambda, alpha, r, step, weightChoice);

% Compress the base layer and restore detail
compression = 0.25;  % this parameter should be adjusted for different inputs
detail = logI - base;
OUT = base * compression + detail;
OUT = exp(OUT);

% Restore color
OUT = OUT./I;
OUT = hdr .* padarray(OUT, [0 0 2], 'circular' , 'post');

% Finally, shift, scale, and gamma correct the result
gamma = 1.0/2.2;
bias = -min(OUT(:));
gain = 0.45;  % this parameter should be adjusted for different inputs
OUT = (gain * (OUT + bias)).^gamma;

%% show the result
figure; imshow(OUT)
