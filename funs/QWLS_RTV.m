%   The code is created based on the method described in the following paper 
%   [1] "Fast Global Image Smoothing via Quasi Weighted Least Squares", Wei Liu, Pingping Zhang, 
%    Hongxing Qin, Xiaolin Huang, Jie Yang and Michael Ng. International Journal of Computer Vision, 2024
%    
%   We adopt the code framwork from the released code of [2] "Structure Extraction from Texture via Relative 
%   Total Variation", Li Xu, Qiong Yan, Yang Xia, Jiaya Jia, ACM  Transactions on Graphics.

%   The code and the algorithm are for non-comercial use only.

%  ---------------------- Input------------------------  
%   I:                    Input UINT8 image, both grayscale and color images are acceptable.
%   lambda:         Parameter controlling the degree of smooth. Range (0, 0.05], 0.01 by default.
%   sigma:           Parameter specifying the maximum size of texture elements. Range (0, 6], 3 by defalut.                       
%   sharpness :   Parameter controlling the sharpness of the final results, which corresponds to 
%                        \epsilon_s in the paper [2]. The smaller the value, the sharper the result.  Range (1e-3, 0.03], 0.02 by defalut.   
%   maxIter:        Number of itearations, 4 by default.
%   r:                   neighborhood radius defined in Eq. (24) of the paper [1]
%   step:              the sliding step between the consecutive extract patches, illustrated in Fig. 5 of the paper [1]


%  ---------------------- Output------------------------
% S:                           smoothed output


function S = QWLS_RTV(I, lambda, sigma, sharpness, maxIter, r, step)

    if (~exist('lambda','var'))
       lambda=0.01;
    end   
    if (~exist('sigma','var'))
       sigma=3.0;
    end 
    if (~exist('sharpness','var'))
        sharpness = 0.02;
    end
    if (~exist('maxIter','var'))
       maxIter=4;
    end    
    I = im2double(I);
    x = I;
    sigma_iter = sigma;
    lambda = lambda/2.0;
    dec=2.0;
    
    for iter = 1:maxIter

        x_s = lpfilter(x, sigma_iter);
        x = mexQWLS_RTV(x, x, x_s, lambda, sharpness, r, step);      
        
        sigma_iter = sigma_iter/dec;
        if sigma_iter < 0.5
            sigma_iter = 0.5;
        end
    end
    S = x;      
end


function ret = conv2_sep(im, sigma)
  ksize = bitor(round(5*sigma),1);
  g = fspecial('gaussian', [1,ksize], sigma); 
  ret = conv2(im,g,'same');
  ret = conv2(ret,g','same');  
end

function FBImg = lpfilter(FImg, sigma)     
    FBImg = FImg;
    for ic = 1:size(FBImg,3)
        FBImg(:,:,ic) = conv2_sep(FImg(:,:,ic), sigma);
    end   
end
