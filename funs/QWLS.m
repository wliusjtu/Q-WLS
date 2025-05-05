%   The code is created based on the method described in the following paper 
%   "Fast Global Image Smoothing via Quasi Weighted Least Squares", Wei Liu, Pingping Zhang, 
%    Hongxing Qin, Xiaolin Huang, Jie Yang and Michael Ng. International Journal of Computer Vision, 2024
%  
%   The code and the algorithm are for non-comercial use only.


%  ---------------------- Input------------------------
%  img:                          input image to be smoothed, can be gray image or RGB color image
%  img_guide:               guidance image, can be gray image or RGB color image
%  lambda:                    \lambda in Eq.(1)/(4), control smoothing strength
%  alpha:                       the power norm in the guidance weight in Eq. (6) if 
%                                   weightChoice = 0, or equal to 1/mu in the guidance
%                                   weight in Eq. (21) if weightChoice = 1
%  r:                               neighborhood radius
%  step:                         the sliding step between the consecutive extract patches, 
%                                   illustrated in Fig. 5
%  weightChoice:          the exponential guidance weight in Eq. (6) if set as 0,
%                                   the fractional guidance weight in Eq. (21) if set as 1


%  ---------------------- Output------------------------
%  res:                           smoothed output


% ----------------------------------------------------------------
%    functions with the "_single" suffix means the images are processed with
%    float data precision, which can be faster but with precision lost
%    in some cases
%    functions with the "_double" suffix means the images are processed with
%    double float data precision, which can be slower but of high data precision

function res = QWLS(img, img_guide, lambda, alpha, r, step, weightChoice)

%  normalize the guidance image to be in range of [0, 1]
img = double(img);
img_guide = double(img_guide);
img_guide = img_guide / (max(img_guide(:)) - min(img_guide(:)));

% check the channel of the input image and the guidance image
[~, ~, cha] = size(img);
[~, ~, cha_guide] = size(img_guide);

% both the input image and the guidance image are gray images and the
% neighborhood radius is r = 1
if (cha ==1) && (cha_guide ==1) && (r == 1)
    res = mexQWLS_r1c1_single(img, img_guide, lambda, alpha, r, step, weightChoice);
    % res = mexQWLS_r1c1_double(img, img_guide, lambda, alpha, r, step, weightChoice);

% both the input image and the guidance image are 3-channel color images and the
% neighborhood radius is r = 1
elseif (cha ==3) && (cha_guide ==3) && (r == 1)
    res = mexQWLS_r1c3_single(img, img_guide, lambda, alpha, r, step, weightChoice);
    %res = mexQWLS_r1c3_double(img, img_guide, lambda, alpha, r, step, weightChoice);

% other cases
else
    res = mexQWLS_single(img, img_guide, lambda, alpha, r, step, weightChoice);
    %res = mexQWLS_double(img, img_guide, lambda, alpha, r, step, weightChoice);

end

