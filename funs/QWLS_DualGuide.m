%   The code is created based on the method described in the following paper 
%   "Fast Global Image Smoothing via Quasi Weighted Least Squares", Wei Liu, Pingping Zhang, 
%    Hongxing Qin, Xiaolin Huang, Jie Yang and Michael Ng. International Journal of Computer Vision, 2024
%  
%   The code and the algorithm are for non-comercial use only.


%  ---------------------- Input------------------------
%  img:                          input image to be smoothed, can be gray image or RGB color image
%  img_guide:               guidance image, can be gray image or RGB color image
%  img_guide_dual:       dual guidance image, can be gray image or RGB color image
%  lambda:                     \lambda in Eq.(18), control smoothing strength
%  sigmaR:                     equal to 1/mu in the guidance weight in Eq. (21)
%  sigmaR_dual:            equal to 1/nu in the guidance weight in Eq. (20)
%  r:                               neighborhood radius
%  step:                         the sliding step between the consecutive extract patches, 


%  ---------------------- Output------------------------
%  res:                           smoothed output


function res = QWLS_DualGuide(img, img_guide, img_guide_dual, lambda, sigmaR, sigmaR_dual, r, step)

%  normalize the guidance image to be in range of [0, 1]
img = double(img);
img_guide = double(img_guide);
img_guide_dual = double(img_guide_dual);
img_guide = img_guide / (max(img_guide(:)) - min(img_guide(:)));

if (max(img_guide_dual(:)) - min(img_guide_dual(:))) ~= 0
    img_guide_dual = img_guide_dual / (max(img_guide_dual(:)) - min(img_guide_dual(:)));
end

weightChoice = 0; % exponential guidance weight for the dual guidance image filtering

% res = mexQWLS_double_DualGuide(img, img_guide, img_guide_dual, lambda, sigmaR, sigmaR_dual, r, step, weightChoice);
res = mexQWLS_single_DualGuide(img, img_guide, img_guide_dual, lambda, sigmaR, sigmaR_dual, r, step, weightChoice);
