#include <math.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include "mex.h"

#define min(X, Y)  ((X) < (Y) ? (X) : (Y))
#define max(X, Y)  ((X) > (Y) ? (X) : (Y))

// size of input image, guidance image and SG_WLS
int r, rowNum, colNum, rc;

// memory management
double ** memAllocDouble2(int row, int col);
void memFreeDouble2(double **p);

// functions
void expLUT(double *LUT, double sigma, int len);
void fracLUT(double *LUT, double sigma, int len);
void img2vec_col(double **img, double **imgGuide, double *vecImg, double *vecGuide, int step);
void img2vec_row(double **img, double **imgGuide, double *vecImg, double *vecGuide, int step);
void vec2img_col(double **img, double *vec, int step);
void vec2img_row(double **img, double *vec, int step);
void getCount(double *count, int len, int r, int step);
void pointDiv(double **imgInter, double **imgFiltered, double *count, int colDir);
void mean(double **A, double **B, double **result);
void solver(double *vecImg, double *vecGuide, double *vecInter, double *vecFiltered, double *beta, double *rangeLUT, double lambda, int sysLen);

// main function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //////  Input images ////////
    double *img = (double*)mxGetData(prhs[0]), *guidance = (double*)mxGetData(prhs[1]);  // input image to be filtered and the guidance image
    
    if((mxGetDimensions(prhs[0])[0] != mxGetDimensions(prhs[1])[0]) || (mxGetDimensions(prhs[0])[1] != mxGetDimensions(prhs[1])[1]))
        mexErrMsgTxt("The input image and the guidance image should be of the same size.");
    
    // Get row, column and channel number
    rowNum = mxGetDimensions(prhs[0])[0];  // row number
    colNum = mxGetDimensions(prhs[0])[1];  // column number
    rc = rowNum * colNum;
    
    if(!(mxGetNumberOfDimensions(prhs[0]) == 2))  // channel number of image to be filtered
        mexErrMsgTxt("This function only supports single-channel input image.");
    
    if(!(mxGetNumberOfDimensions(prhs[1]) == 2))  // channel number of guidance image
        mexErrMsgTxt("This function only supports single-channel input image.");
    
    //////// SG_WLS parameters  ///////////////
    double lambda = (double)mxGetScalar(prhs[2]); // lambda of the SG_WLS
    double sigmaR = (double)mxGetScalar(prhs[3]); // range sigma for the guidance weight
    r = (int)mxGetScalar(prhs[4]); // raius of the neighborhood
    int step = (int)mxGetScalar(prhs[5]);  // the step size between each SG_WLS
    int weightChoice = (int)mxGetScalar(prhs[6]);  // 0 for exp weight, 1 for frac weight
    
    if(r!=1) mexErrMsgTxt("This function only supports the neighborhood radius of r=1.\n");
    if(step>(2*r + 1)) mexErrMsgTxt("step size should not be larger than 2r+1!\n");
    
    // Transfer the input image and the guidance image into 3-dimention arrays
    double **imgInput = memAllocDouble2(rowNum, colNum);  
    double *ptr_imgInput = &imgInput[0][0];
    for(int i=0; i<rowNum; i++)
        for(int j=0; j<colNum; j++)
            *ptr_imgInput++ = img[j *rowNum + i];
    
    double **imgGuide = memAllocDouble2(rowNum, colNum);
    double *ptr_imgGuide = &imgGuide[0][0];
    for(int i=0; i<rowNum; i++)
        for(int j=0; j<colNum; j++)
            *ptr_imgGuide++ = guidance[j * rowNum + i];
    
    // Intermediate output
    double **imgInter = memAllocDouble2(rowNum, colNum);
    
    // filtered image
    double **imgFiltered = memAllocDouble2(rowNum, colNum); 
    double **imgFiltered_row = memAllocDouble2(rowNum, colNum); 
    double **imgFiltered_col = memAllocDouble2(rowNum, colNum); 
    
    // Output
    plhs[0] = mxDuplicateArray(prhs[0]); // the output has the same size as the input image
    double *imgResult = (double*) mxGetData(plhs[0]);
    
    // Q-WLS related variables
    int patchNum_row = (rowNum - 2 * r - 1) / step + 2;
    int patchNum_col = (colNum - 2 * r - 1) / step + 2;
    int sysLen_row = patchNum_row * colNum * (2 * r + 1);  // the system size of the Q-WLS along the row direction
    int sysLen_col = patchNum_col * rowNum * (2 * r + 1);  // the system size of the Q-WLS along the column direction
    
    int maxSysLen = max(sysLen_row, sysLen_col);
    
    double *beta = (double*) malloc(sizeof(double) * (maxSysLen + 10));
    
    double *vecImg = (double*)malloc(sizeof(double) * (maxSysLen + 10));
    double *vecGuide = (double*)malloc(sizeof(double) * (maxSysLen + 10));
    double *vecInter = (double*)malloc(sizeof(double) * (maxSysLen + 10));  // intermediate variables 'F' in solving P*Y=F, Q*U=F
    double *vecFiltered = (double*)malloc(sizeof(double) * (maxSysLen + 10));  // smoothed output 'U' in solving P*Y=F, Q*U=F
    
    /// accumulate the count of the filtered value at the same location
    double *count_row = (double*) malloc(sizeof(double) * (rowNum + 10));
    double *count_col = (double*) malloc(sizeof(double) * (colNum + 10));
    
    // weight lookup table
    int maxRange = 255 + 10;
    maxRange = maxRange * maxRange;
    double *rangeLUT = (double*)malloc(sizeof(double) * (maxRange + 10));
    
    if(weightChoice == 0) expLUT(rangeLUT, sigmaR, maxRange);
    else if(weightChoice ==1)  fracLUT(rangeLUT, sigmaR, maxRange);
    else mexErrMsgTxt("Weight choice should be 0 (exponential) or 1 (fractional)\n.");
    
    // do filtering
    clock_t tStart = clock(); // time measurement;
    
    //  column direction
    memset(&imgInter[0][0], 0.0, sizeof(double) * rowNum * colNum);
    memset(&count_col[0], 0.0, sizeof(double) * colNum);
    
    img2vec_col(imgInput, imgGuide, vecImg, vecGuide, step);
    solver(vecImg, vecGuide, vecInter, vecFiltered, beta, rangeLUT, lambda, sysLen_col);
    vec2img_col(imgInter, vecFiltered, step);
    
    getCount(count_col, colNum, r, step);
    pointDiv(imgInter, imgFiltered_col, count_col, 1);
    
    // row direction
    memset(&imgInter[0][0], 0.0, sizeof(double) * rowNum * colNum);
    memset(&count_row[0], 0.0, sizeof(double) * rowNum);
    
    img2vec_row(imgInput, imgGuide, vecImg, vecGuide, step);
    solver(vecImg, vecGuide, vecInter, vecFiltered, beta, rangeLUT, lambda, sysLen_row);
    vec2img_row(imgInter, vecFiltered, step);
    
    getCount(count_row, rowNum, r, step);
    pointDiv(imgInter, imgFiltered_row, count_row, 0);
    
    // average the filtered image in row direction and column direction
    mean(imgFiltered_row, imgFiltered_col, imgFiltered);
    
    mexPrintf("Elapsed time is %f seconds.\n", double(clock() - tStart)/CLOCKS_PER_SEC);
    
    // transfer to the output
    double *ptr_imgFiltered = &imgFiltered[0][0];
    for(int i=0; i<rowNum; i++)
        for(int j=0; j<colNum; j++)
            imgResult[j * rowNum + i] = (double)*ptr_imgFiltered++;
    
    // free the memory
    memFreeDouble2(imgInput);
    memFreeDouble2(imgFiltered);
    memFreeDouble2(imgFiltered_row);
    memFreeDouble2(imgFiltered_col);
    memFreeDouble2(imgGuide);
    memFreeDouble2(imgInter);
    free(beta); beta = NULL;
    free(vecImg); vecImg = NULL;
    free(vecGuide); vecGuide = NULL;
    free(vecFiltered); vecFiltered = NULL;
    free(vecInter); vecInter = NULL;
    free(count_row); count_row = NULL;
    free(count_col); count_col = NULL;
    free(rangeLUT); rangeLUT = NULL;
}


//============ Functions =============//
////////////////////////////////
double** memAllocDouble2(int row, int col)
{
    // allocate the memory for a 2-dimension array which can be indexed as p[row][col]
    
	int padding=10;
	double *a, **p;
    
	a=(double*) malloc(sizeof(double) * (row * col + padding));
	if(a==NULL) {mexErrMsgTxt("memAllocDouble: Memory allocate failure.\n"); }
	p=(double**) malloc(sizeof(double*) * row);
    
	for(int i=0; i<row; i++) p[i] = &a[i * col];
    
	return(p);
}


//////////////////////////////////
void memFreeDouble2(double **p)
{
    // free the memory of an allocated 2-demision array
    
	if(p!=NULL)
	{
		free(p[0]);
		free(p);
		p=NULL;
	}
}


////////////////////////////////////////////////////
void expLUT(double *LUT, double sigma, int len)
{
    double sigma_new = 2 * sigma * sigma * 65535.0 * 65535.0;
    double *ptr=&LUT[0];
    for(int i=0; i<len; i++)   *ptr++ = exp(double(-i * i) / sigma_new);
    
}


///////////////////////////////////////////////////
void fracLUT(double *LUT, double sigma, int len)
{
    double *ptr=&LUT[0];
    for(int i=0; i<len; i++)  *ptr++ = 1 / (pow(double(i) / 65536.0, sigma) + 0.00001);
    
}


////////////////////////////////////////////////////
void img2vec_col(double **img, double **imgGuide, double *vecImg, double *vecGuide, int step)
{
    double  *ptr_img, *ptr_guide, *ptr_vecImg, *ptr_vecGuide;
    int row, col, cha, colSlide, left, right, rX2, shift, maxIterColNum;
    bool leftright=true, updown=true, lastpatch=false, shouldbreak=false;
    int colMinusOne = colNum - 1, colPlusOne = colNum + 1;
    
    maxIterColNum = colNum - (colNum - 2 * r - 1)%step - r;
    ptr_vecImg = &vecImg[0];
    ptr_vecGuide = &vecGuide[0];
    ptr_img = &img[0][0];
    ptr_guide = &imgGuide[0][0];
    rX2 = 2 * r;
    
    for(col=r; col<maxIterColNum + step; col+=step)
    {
        if(lastpatch)   // the last patch that centers around column "colNum - r"
        {
            shouldbreak = true;
            col = colNum - r - 1;
            
            if(updown){   
                // we inforce leftright=true at the beginning of each patch 
                ptr_img = &img[0][col - r];
                ptr_guide = &imgGuide[0][col - r];}
            else{
                // we inforce leftright=true at the beginning of each patch
                ptr_img = &img[rowNum - 1][col - r];
                ptr_guide = &imgGuide[rowNum - 1][col - r];} 
        }
        
        if(col==(maxIterColNum - 1))  lastpatch = true;  
        
        left = col - r;
        right = col + r;
        
        if(updown)
        {// column up-bottom extraction sliding direction
            for(row=0; row<rowNum; row++)
            {
                if(leftright)
                {// row left-to-right extraction sliding direction
                    for(colSlide=left; colSlide<=right; colSlide++)
                    {
                        // for the image to be filtered
                        *ptr_vecImg++ = *ptr_img++;
                        
                        // for the guide image
                        *ptr_vecGuide++ = *ptr_guide++;   
                    }  
                    
                    // move to the next row
                    ptr_img += colMinusOne;
                    ptr_guide += colMinusOne;

                    // change sliding direction between left-to-right and right-to-left
                    leftright = !leftright;  
                }
                else
                {// row right-to-left extraction sliding direction
                    for(colSlide=right; colSlide>=left; colSlide--)
                    {
                        // for the image to be filtered
                        *ptr_vecImg++ = *ptr_img--;

                        // for the guidance image
                        *ptr_vecGuide++ = *ptr_guide--;
                    }
                    
                    // move to the next row
                    ptr_img += colPlusOne;
                    ptr_guide += colPlusOne;

                    // change sliding direction between left-to-right and right-to-left
                    leftright = !leftright;   
                }  
            }
            
            if(leftright){
                shift = colNum - step;
                ptr_img -= shift;
                ptr_guide -= shift;}
            else{
                shift = colNum + rX2 - step;
                ptr_img -= shift;
                ptr_guide -= shift;
                leftright = true;}  // we always inforce leftright=true at the beginning of each patch
            
            updown = !updown;
        }
        else // column column bottom-up extraction direction
        {
            for(row=rowNum - 1; row>=0; row--)
            {
                if(leftright)
                {// row left-to-right extraction sliding direction
                    for(colSlide=left; colSlide<=right; colSlide++)
                    {
                        // for the image to be filtered
                        *ptr_vecImg++ = *ptr_img++;
                        
                        // for the guide image
                        *ptr_vecGuide++ = *ptr_guide++;    
                    }  
                    
                    // move to the next row
                    ptr_img -= colPlusOne;
                    ptr_guide -= colPlusOne;

                    // change sliding direction between left-to-right and right-to-left
                    leftright = !leftright;  
                }
                else
                {// row right-to-left extraction sliding direction
                    for(colSlide=right; colSlide>=left; colSlide--)
                    {
                        // for the image to be filtered
                        *ptr_vecImg++ = *ptr_img--;

                        // for the guidance image
                        *ptr_vecGuide++ = *ptr_guide--;
                    }
                    
                    // move to the next row
                    ptr_img -= colMinusOne;
                    ptr_guide -= colMinusOne;

                    // change sliding direction between left-to-right and right-to-left
                    leftright = !leftright;   
                }  
            }
            
            if(leftright){
                shift = colNum + step;
                ptr_img += shift;
                ptr_guide += shift;}
            else{
                shift = colNum - rX2 + step;
                ptr_img += shift;
                ptr_guide += shift;
                leftright = true;}
            
            updown = !updown;
        }
        
        if(shouldbreak) break;  // break the loop after the last patch
    }
}


//////////////////////////////////////////////////
void img2vec_row(double **img, double **imgGuide, double *vecImg, double *vecGuide, int step)
{
    double *ptr_img, *ptr_guide, *ptr_vecImg, *ptr_vecGuide;
    int row, col, cha, rowSlide, maxIterRowNum, up, bottom, shift;
    bool leftright=true, updown=true, lastpatch=false, shouldbreak=false;
    int colPlusOne = colNum + 1, colMinusOne = colNum - 1;
    int stepXCol = step * colNum, diaXCol = 2 * r * colNum;
    
    maxIterRowNum = rowNum - (rowNum - 2 * r - 1)%step - r;
    ptr_vecImg = &vecImg[0];
    ptr_vecGuide = &vecGuide[0];
    ptr_img = &img[0][0];
    ptr_guide = &imgGuide[0][0];
    
    for(row=r; row<maxIterRowNum + step; row+=step)
    {
        if(lastpatch)
        {
            shouldbreak = true;
            row = rowNum - r - 1;
            
            if(leftright){
                // we inforce updown=true at the beginning of each patch
                ptr_img = &img[row - r][0];
                ptr_guide = &imgGuide[row - r][0];}
            
            else{
                // we enforce updown=true at the beginning of each patch
                ptr_img = &img[row - r ][colNum - 1];
                ptr_guide = &imgGuide[row - r][colNum - 1];}   
        }
        
        if(row==(maxIterRowNum - 1))  lastpatch = true;  
        
        up = row - r;
        bottom = row + r;
        
        if(leftright) // row left-to-right extraction direction
        {
            for(col=0; col<colNum; col++)
            {
                if(updown)
                {// up-bottom extraction sliding direction
                    for(rowSlide=up; rowSlide<=bottom; rowSlide++)
                    {
                        // for the image to be filtered
                        *ptr_vecImg++ = *ptr_img;
                        ptr_img += colNum;

                        // for the guidance image
                        *ptr_vecGuide++ = *ptr_guide;
                        ptr_guide += colNum;
                    }

                    // move to the next column
                    ptr_img -=  colMinusOne;
                    ptr_guide -= colMinusOne;

                    // change the extraction sliding direction between bottom-up and up-bottom in each column of the image
                    updown = !updown; 
                }
                else  // bottom-up extraction sliding direction
                {
                    for(rowSlide=bottom; rowSlide>=up; rowSlide--)
                    {
                        // for the image to be filtered
                        *ptr_vecImg++ = *ptr_img;
                        ptr_img -= colNum;        

                        // for the guidance image
                        *ptr_vecGuide++ = *ptr_guide;
                        ptr_guide -= colNum;
                    }

                    // move to the next column
                    ptr_img +=  colPlusOne;
                    ptr_guide += colPlusOne;

                    // change the extraction sliding direction between bottom-up and up-bottom in each column of the image
                    updown = !updown;
                }
            }
            
            if(updown){
                shift = stepXCol - 1;
                ptr_img += shift;
                ptr_guide += shift;}
            else{
                shift = 1 + diaXCol - stepXCol; 
                ptr_img -= shift;
                ptr_guide -= shift;
                updown = true;}  // we always inforce updown=true at the beginning of a patch
            
            leftright = !leftright; 
        }
        else  // row right-to-left extraction direction
        {
            for(col=colNum - 1; col>=0; col--)
            {
                if(updown)
                {// up-bottom extraction sliding direction
                    for(rowSlide=up; rowSlide<=bottom; rowSlide++)
                    {
                        // for the image to be filtered
                        *ptr_vecImg++ = *ptr_img;
                        ptr_img += colNum;

                        // for the guidance image
                        *ptr_vecGuide++ = *ptr_guide;
                        ptr_guide += colNum;
                    }

                    // move to the next column
                    ptr_img -=  colPlusOne;
                    ptr_guide -= colPlusOne;

                    // change the extraction sliding direction between bottom-up and up-bottom in each column of the image
                    updown = !updown; 
                }
                else  // bottom-up extraction sliding direction
                {
                    for(rowSlide=bottom; rowSlide>=up; rowSlide--)
                    {
                        // for the image to be filtered
                        *ptr_vecImg++ = *ptr_img;
                        ptr_img -= colNum;        

                        // for the guidance image
                        *ptr_vecGuide++ = *ptr_guide;
                        ptr_guide -= colNum;
                    }

                    // move to the next column
                    ptr_img +=  colMinusOne;
                    ptr_guide += colMinusOne;

                    // change the extraction sliding direction between bottom-up and up-bottom in each column of the image
                    updown = !updown;
                }
            }
            
            if(updown){
                shift = stepXCol + 1;
                ptr_img += shift;
                ptr_guide +=shift;}
            else{
                shift = 1 - diaXCol + stepXCol;
                ptr_img += shift;
                ptr_guide += shift;
                updown = true;}
            
            leftright = !leftright; 
        } 
        
        if(shouldbreak) break;  // break the loop after the last patch
    }
}


////////////////////////////////////////////////////
void vec2img_col(double **img, double *vec, int step)
{
    double  *ptr_img, *ptr_vec;
    int row, col, cha, colSlide, left, right, rX2, shift, maxIterColNum;
    bool leftright=true, updown=true, lastpatch=false, shouldbreak=false;
    int colMinusOne = colNum - 1, colPlusOne = colNum + 1;
    
    maxIterColNum = colNum - (colNum - 2 * r - 1)%step - r;
    ptr_vec = &vec[0];
    ptr_img = &img[0][0];
    
    rX2 = 2 * r;
    
    for(col=r; col<maxIterColNum + step; col+=step)
    {
        if(lastpatch)   // the last patch that centers around column "colNum - r"
        {
            shouldbreak = true;
            col = colNum - r - 1;
            
            if(updown) ptr_img = &img[0][col - r];
            else  ptr_img = &img[rowNum - 1][col - r];
        }
        
        if(col==(maxIterColNum - 1))  lastpatch = true;  
        
        left = col - r;
        right = col + r;
        
        if(updown)
        {// column up-bottom extraction sliding direction
            for(row=0; row<rowNum; row++)
            {
                if(leftright)
                {// row left-to-right extraction sliding direction
                    for(colSlide=left; colSlide<=right; colSlide++)
                        *ptr_img++ += *ptr_vec++;
                    
                    // move to the next row
                    ptr_img += colMinusOne;

                    // change sliding direction between left-to-right and right-to-left
                    leftright = !leftright;  
                }
                else
                {// row right-to-left extraction sliding direction
                    for(colSlide=right; colSlide>=left; colSlide--)
                        *ptr_img-- += *ptr_vec++;
                    
                    // move to the next row
                    ptr_img += colPlusOne;

                    // change sliding direction between left-to-right and right-to-left
                    leftright = !leftright;   
                }  
            }
            
            if(leftright) ptr_img -= colNum - step;
            else{
                ptr_img -= colNum + rX2 - step;
                leftright = true;}  // we always inforce leftright=true at the beginning of each patch
            
            updown = !updown;
        }
        else // column column bottom-up extraction direction
        {
            for(row=rowNum - 1; row>=0; row--)
            {
                if(leftright)
                {// row left-to-right extraction sliding direction
                    for(colSlide=left; colSlide<=right; colSlide++)
                        *ptr_img++ += *ptr_vec++;
                    
                    // move to the next row
                    ptr_img -= colPlusOne;

                    // change sliding direction between left-to-right and right-to-left
                    leftright = !leftright;  
                }
                else
                {// row right-to-left extraction sliding direction
                    for(colSlide=right; colSlide>=left; colSlide--)
                        *ptr_img-- += *ptr_vec++;
                    
                    // move to the next row
                    ptr_img -= colMinusOne;

                    // change sliding direction between left-to-right and right-to-left
                    leftright = !leftright;   
                }  
            }
            
            if(leftright) ptr_img += colNum + step;
            else{
                ptr_img += colNum - rX2 + step;
                leftright = true;}
            
            updown = !updown;
        }
        
        if(shouldbreak) break;  // break the loop after the last patch
    }
}


//////////////////////////////////////////////////
void vec2img_row(double **img, double *vec, int step)
{
    //////// this function DOES have the "add" function /////////
    double *ptr_img, *ptr_vec;
    int row, col, cha, rowSlide, maxIterRowNum, up, bottom, shift;
    bool leftright=true, updown=true, lastpatch=false, shouldbreak=false;
    int colPlusOne = colNum + 1, colMinusOne = colNum - 1;
    int stepXCol = step * colNum, diaXCol = 2 * r * colNum;
    
    maxIterRowNum = rowNum - (rowNum - 2 * r - 1)%step - r;
    ptr_vec = &vec[0];
    ptr_img = &img[0][0];
    
    for(row=r; row<maxIterRowNum + step; row+=step)
    {
        if(lastpatch)
        {
            shouldbreak = true;
            row = rowNum - r - 1;
            
            if(leftright) ptr_img = &img[row - r][0];
            else ptr_img = &img[row - r ][colNum - 1];
        }
        
        if(row==(maxIterRowNum - 1))  lastpatch = true;  
        
        up = row - r;
        bottom = row + r;
        
        if(leftright) // row left-to-right extraction direction
        {
            for(col=0; col<colNum; col++)
            {
                if(updown)
                {// up-bottom extraction sliding direction
                    for(rowSlide=up; rowSlide<=bottom; rowSlide++)
                    {
                        // for the image to be filtered
                        *ptr_img += *ptr_vec++;
                        ptr_img += colNum;
                    }

                    // move to the next column
                    ptr_img -=  colMinusOne;

                    // change the extraction sliding direction between bottom-up and up-bottom in each column of the image
                    updown = !updown; 
                }
                else  // bottom-up extraction sliding direction
                {
                    for(rowSlide=bottom; rowSlide>=up; rowSlide--)
                    {
                        // for the image to be filtered
                        *ptr_img += *ptr_vec++;
                        ptr_img -= colNum;        
                    }

                    // move to the next column
                    ptr_img +=  colPlusOne;

                    // change the extraction sliding direction between bottom-up and up-bottom in each column of the image
                    updown = !updown;
                }
            }
            
            if(updown) ptr_img += stepXCol - 1;
            else{
                ptr_img -= 1 + diaXCol - stepXCol; 
                updown = true;}  // we always inforce updown=true at the beginning of a patch
            
            leftright = !leftright; 
        }
        else  // row right-to-left extraction direction
        {
            for(col=colNum - 1; col>=0; col--)
            {
                if(updown)
                {// up-bottom extraction sliding direction
                    for(rowSlide=up; rowSlide<=bottom; rowSlide++)
                    {
                        // for the image to be filtered
                        *ptr_img += *ptr_vec++;
                        ptr_img += colNum;
                    }

                    // move to the next column
                    ptr_img -=  colPlusOne;

                    // change the extraction sliding direction between bottom-up and up-bottom in each column of the image
                    updown = !updown; 
                }
                else  // bottom-up extraction sliding direction
                {
                    for(rowSlide=bottom; rowSlide>=up; rowSlide--)
                    {
                        // for the image to be filtered
                        *ptr_img += *ptr_vec++;
                        ptr_img -= colNum;
                    }

                    // move to the next column
                    ptr_img +=  colMinusOne;

                    // change the extraction sliding direction between bottom-up and up-bottom in each column of the image
                    updown = !updown;
                }
            }
            
            if(updown) ptr_img += stepXCol + 1;
            else{
                ptr_img += 1 - diaXCol + stepXCol;
                updown = true;}
            
            leftright = !leftright; 
        } 
        
        if(shouldbreak) break;  // break the loop after the last patch
    }
}


///////////////////////////////////////
void solver(double *vecImg, double *vecGuide, double *vecInter, double *vecFiltered, double *beta, double*rangeLUT, double lambda, int sysLen)
{
    double *ptr_guide_cur, *ptr_guide_nei, *ptr_beta, *ptr_F, *ptr_Y, *ptr_Y_pre, *ptr_X, *ptr_X_pre;
    double diffR, temp, weightR;
    double a_cur, b_cur, b_pre, tempAlpha, tempGamma, tempBeta;
    int i, k, n=sysLen;
    
    ptr_beta = &beta[0];
    ptr_guide_cur = &vecGuide[0];  ptr_guide_nei = &vecGuide[1]; 
    ptr_F = &vecImg[0];  ptr_Y = &vecInter[0];  ptr_Y_pre = ptr_Y;
    
    // i = 0
    {
        //////////  get Laplacian //////////
        // range weight
        temp = *ptr_guide_cur++ - *ptr_guide_nei++;
        diffR = fabs(temp);
        // diffR = temp * temp;
        weightR = rangeLUT[int(diffR * 65535.0)];

        b_cur = -lambda * weightR;
        a_cur = 1 - b_cur;
        b_pre = b_cur;
        
        ///////////////////// decompose and solve L * Y = F ////////////
        tempAlpha = a_cur;
        
        *ptr_Y++ = *ptr_F++ / tempAlpha; 
        
        tempGamma = b_cur;
        tempBeta = tempGamma / tempAlpha;
        *ptr_beta++ = tempBeta;
    }
    
    // i = 1, ..., n - 2
    for(i=1; i<n - 1; i++)
    {
        //////////  get Laplacian //////////
        // range weight
        temp = *ptr_guide_cur++ - *ptr_guide_nei++;
        // diffR = temp * temp;
        diffR = fabs(temp);
        weightR = rangeLUT[int(diffR * 65535.0)];

        b_cur = -lambda * weightR;
        a_cur = 1 - b_cur - b_pre;
        b_pre = b_cur;
        
        ///////////////////// decompose  and solve L * Y = F  ////////////
        tempAlpha = a_cur - tempGamma * tempBeta;
        
        *ptr_Y++ = (*ptr_F++ - tempGamma * (*ptr_Y_pre++)) / tempAlpha;
        
        tempGamma = b_cur;
        tempBeta = tempGamma / tempAlpha;
        *ptr_beta++ = tempBeta;
    }
    
    // i = n - 1
    {
        a_cur = 1 - b_pre;
        tempAlpha = a_cur - tempGamma * tempBeta;
        
       *ptr_Y++ = (*ptr_F++ - tempGamma * (*ptr_Y_pre++)) / tempAlpha;
    }
    
    ///////////  solve U*X = Y ///////////
    ptr_X = &vecFiltered[n-1];  ptr_X_pre = ptr_X;  ptr_Y = &vecInter[n-1];
    ptr_beta = &beta[n - 2];
    
    *ptr_X-- = *ptr_Y--;
    
    for(i=n-2; i>=0; i--)
        *ptr_X-- = (*ptr_Y--) - (*ptr_beta--) * (*ptr_X_pre--);

}


/////////////////////////////////////
void pointDiv(double **imgInter, double **imgFiltered, double *count, int colDir)
{
    double *ptr_inter, *ptr_filtered, *ptr_count, value;
    ptr_inter = &imgInter[0][0];
    ptr_filtered = &imgFiltered[0][0];
    
    if(colDir)
    {
        for(int i=0; i<rowNum; i++){
            ptr_count = &count[0];
            for(int j=0; j<colNum; j++)
                *ptr_filtered++ = *ptr_inter++ / *ptr_count++;}
    }
    else
    {
        ptr_count = &count[0];
        for(int i=0; i<rowNum; i++){
            value = *ptr_count++;
            for(int j=0; j<colNum; j++)
                *ptr_filtered++ = *ptr_inter++ / value;}
    }
} 

void getCount(double *count, int len, int r, int step)
{
    int i, j, step_new, maxIterNum;
    double *ptr;
    
    step_new = 2 * r + 1 - step;
    maxIterNum = r + 1 + ((len - r - r - 1)/step)*step;
    
    // this is for the row direction where we have step as the input value
    ptr = &count[0];
    for(i = r; i < maxIterNum; i += step){
        for(j = -r; j <= r; j++)
            *ptr++ += 1.0;
        ptr -= step_new;}
    
    // the last patch
    ptr = &count[len - 2* r - 1];
    for(j = -r; j <= r; j++)
        *ptr++ += 1.0;
}


////////////////////////////////////////
void mean(double **A, double **B, double **result)
{
    double *ptr_a, *ptr_b, *ptr_result;
    int i, j, k;
    
    ptr_a = &A[0][0];
    ptr_b = &B[0][0];
    ptr_result = &result[0][0];
    
    for(i=0; i<rowNum; i++)
        for(j=0; j<colNum; j++)
            *ptr_result++ = (*ptr_a++ + *ptr_b++) / 2.0; 
}
