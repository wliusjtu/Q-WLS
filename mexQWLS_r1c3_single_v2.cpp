#include <math.h>
#include <time.h>
#include <string.h>
#include "mex.h"

#define min(X, Y)  ((X) < (Y) ? (X) : (Y))
#define max(X, Y)  ((X) > (Y) ? (X) : (Y))

// size of input image, guidance image and SG_WLS
int rowNum, colNum, rc, ccImg, ccGuide, ccImgMinusChaNum, ccImgPlusChaNum, ccGuideMinusChaNum, ccGuidePlusChaNum;
int chaNumImg, chaNumGuide, chaNumImgX2, chaNumGuideX2;
int r, sysLen_row, sysLen_col; 

// memory management
float *** memAllocFloat3(int row, int col, int cha);
float ** memAllocFloat2(int row, int col);
void memFreeFloat3(float ***p);
void memFreeFloat2(float **p);

// functions
void expLUT(float *LUT, float sigma, int chaNum, int len);
void fracLUT(float *LUT, float sigma, int chaNum, int len);
void img2vec_col(float ***img, float ***imgGuide, float **vecImg, float **vecGuide, int step);
void img2vec_row(float ***img, float ***imgGuide, float **vecImg, float **vecGuide, int step);
void vec2img_col(float ***img, float **vec, int step);
void vec2img_row(float ***img, float **vec, int step);
void getCount(float *count, int len, int r, int step);
void pointDiv(float ***imgInter, float ***imgFiltered, float *count, int colDir);
void mean(float ***A, float ***B, float ***result);
void solver(float **vecImg, float **vecGuide, float **vecInter, float **vecFiltered, float *beta, float *rangeLUT, float lambda, int sysLen);

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
    
    if(mxGetNumberOfDimensions(prhs[0]) == 2)  // channel number of image to be filtered
        chaNumImg = 1;  // single channel image
    else
        chaNumImg = mxGetDimensions(prhs[0])[2];;  // rgb image
    
    ccImg = colNum * chaNumImg;
    chaNumImgX2 = 2 * chaNumImg;
    ccImgMinusChaNum = ccImg - chaNumImg;
    ccImgPlusChaNum = ccImg + chaNumImg;
    
    if(mxGetNumberOfDimensions(prhs[1]) == 2)  // channel number of guidance image
        chaNumGuide = 1;  // single channel image
    else 
        chaNumGuide = mxGetDimensions(prhs[1])[2];;  // rgb image
    
    ccGuide = colNum * chaNumGuide;
    chaNumGuideX2 = 2 * chaNumGuide;
    ccGuideMinusChaNum = ccGuide - chaNumGuide;
    ccGuidePlusChaNum = ccGuide + chaNumGuide;
    
    //////// SG_WLS parameters  ///////////////
    float lambda = (float)mxGetScalar(prhs[2]); // lambda of the SG_WLS
    float sigmaR = (float)mxGetScalar(prhs[3]); // range sigma for the guidance weight
    r = (int)mxGetScalar(prhs[4]); // raius of the neighborhood
    int step = (int)mxGetScalar(prhs[5]);  // the step size between each SG_WLS
    int weightChoice = (int)mxGetScalar(prhs[6]);  // 0 for exp weight, 1 for frac weight
    
    if(r!=1) mexErrMsgTxt("This function only supports the neighborhood radius of r=1.");
    
    // Transfer the input image and the guidance image into 3-dimention arrays
    float ***imgInput = memAllocFloat3(rowNum, colNum, chaNumImg);
    float *ptr_imgInput = &imgInput[0][0][0];
    for(int i=0; i<rowNum; i++)
        for(int j=0; j<colNum; j++)
            for(int k=0; k<chaNumImg; k++)
                *ptr_imgInput++ = (float)img[k * rc + j *rowNum + i];
    
    float ***imgGuide = memAllocFloat3(rowNum, colNum, chaNumGuide);
    float *ptr_imgGuide = &imgGuide[0][0][0];
    for(int i=0; i<rowNum; i++)
        for(int j=0; j<colNum; j++)
            for(int k=0; k<chaNumGuide; k++)
                *ptr_imgGuide++ = (float)guidance[k * rc + j * rowNum + i];
    
    // Intermediate output
    float *** imgInter = memAllocFloat3(rowNum, colNum, chaNumImg);
    
    // filtered image
    float ***imgFiltered = memAllocFloat3(rowNum, colNum, chaNumImg);
    float ***imgFiltered_row = memAllocFloat3(rowNum, colNum, chaNumImg);
    float ***imgFiltered_col = memAllocFloat3(rowNum, colNum, chaNumImg);
    
    // Output
    plhs[0] = mxDuplicateArray(prhs[0]); // the output has the same size as the input image
    double *imgResult = (double*) mxGetData(plhs[0]);
    
    // Q-WLS related variables
    int patchNum_row = (rowNum - 2 * r - 1) / step + 2;
    int patchNum_col = (colNum - 2 * r - 1) / step + 2;
    sysLen_row = patchNum_row * colNum * (2 * r + 1);  // the system size of the Q-WLS along the row direction
    sysLen_col = patchNum_col * rowNum * (2 * r + 1);  // the system size of the Q-WLS along the column direction
    
    int maxSysLen = max(sysLen_row, sysLen_col);
    
    float *beta = (float*)malloc(sizeof(float) * (maxSysLen + 10));
    
    float **vecImg = memAllocFloat2(maxSysLen, chaNumImg);
    float **vecGuide = memAllocFloat2(maxSysLen, chaNumGuide); 
    float **vecInter = memAllocFloat2(maxSysLen, chaNumImg);  // intermediate variables 'F' in solving P*Y=F, Q*U=F
    float **vecFiltered = memAllocFloat2(maxSysLen, chaNumImg);  // smoothed output 'U' in solving P*Y=F, Q*U=F
    
    // accumulate the count of the filtered value at the same location
    float *count_row = (float*)malloc(sizeof(float) * (rowNum + 10));
    float *count_col = (float*)malloc(sizeof(float) * (colNum+ 10));
    
    // weight lookup table
    int maxRange = 255 + 10;
    maxRange = chaNumGuide * maxRange * maxRange;
    float *rangeLUT = (float*)malloc(sizeof(float) * (maxRange + 10));
    
    if(weightChoice == 0) expLUT(rangeLUT, sigmaR, chaNumGuide, maxRange);
    else if(weightChoice == 1) fracLUT(rangeLUT, sigmaR, chaNumGuide, maxRange);
    else mexErrMsgTxt("Weight choice should be 0 (exponential) or 1 (fractional)\n.");
    
    // do filtering
    clock_t tStart = clock(); // time measurement;
    
    //  column direction
    memset(&imgInter[0][0][0], 0.0, sizeof(float) * rowNum * colNum * chaNumImg);
    memset(&count_col[0], 0.0, sizeof(float) * colNum);
    
    img2vec_col(imgInput, imgGuide, vecImg, vecGuide, step);
    solver(vecImg, vecGuide, vecInter, vecFiltered, beta, rangeLUT, lambda, sysLen_col);
    vec2img_col(imgInter, vecFiltered, step);
    
    getCount(count_col, colNum, r, step);
    pointDiv(imgInter, imgFiltered_col, count_col, 1);
    
    // row direction
    memset(&imgInter[0][0][0], 0.0, sizeof(float) * rowNum * colNum * chaNumImg);
    memset(&count_row[0], 0.0, sizeof(float) * rowNum);
    
    img2vec_row(imgInput, imgGuide, vecImg, vecGuide, step);
    solver(vecImg, vecGuide, vecInter, vecFiltered, beta, rangeLUT, lambda, sysLen_row);
    vec2img_row(imgInter, vecFiltered, step);
    
    getCount(count_row, rowNum, r, step);
    pointDiv(imgInter, imgFiltered_row, count_row, 0);
    
    // average the filtered image in row direction and column direction
    mean(imgFiltered_row, imgFiltered_col, imgFiltered);
    
    mexPrintf("Elapsed time is %f seconds.\n", float(clock() - tStart)/CLOCKS_PER_SEC);
    
    // transfer to the output
    float *ptr_imgFiltered = &imgFiltered[0][0][0];
    for(int i=0; i<rowNum; i++)
        for(int j=0; j<colNum; j++)
            for(int k=0; k<chaNumImg; k++)
                imgResult[k * rc + j * rowNum + i] = (double)*ptr_imgFiltered++;
    
    // free memory
    memFreeFloat3(imgInput);
    memFreeFloat3(imgFiltered);
    memFreeFloat3(imgFiltered_row);
    memFreeFloat3(imgFiltered_col);
    memFreeFloat3(imgGuide);
    memFreeFloat3(imgInter);
    free(beta); beta = NULL;
    memFreeFloat2(vecImg);
    memFreeFloat2(vecGuide);
    memFreeFloat2(vecFiltered);
    memFreeFloat2(vecInter);
    free(count_row); count_row = NULL;
    free(count_col); count_col = NULL;
    free(rangeLUT); rangeLUT = NULL;

}


//============ Functions =============//

float *** memAllocFloat3(int row, int col, int cha)
{   
    // allocate the memory for a 3-dimension array which can be indexed as pp[row][col][cha]
    
	int padding=10;
	float *a, **p, ***pp;
    
	a=(float*) malloc(sizeof(float) * (row * col * cha + padding));
	if(a==NULL) {mexErrMsgTxt("memAllocFloat: Memory allocate failure.\n"); }
	p=(float**) malloc(sizeof(float*) * row * col);
	pp=(float***) malloc(sizeof(float**) * row);
    
    int cc = col * cha;
	int i, j;
    
	for(i=0; i<row; i++) 
		for(j=0; j<col; j++) 
            p[i * col + j] = &a[i * cc + j * cha];
    
	for(i=0; i<row; i++) 
		pp[i] = &p[i* col];
    
	return(pp);
}


///////////////////////////////
void memFreeFloat3(float ***p)
{
    // free the memory of an allocated 3-dimension array
    
	if(p!=NULL)
	{
		free(p[0][0]);
		free(p[0]);
		free(p);
		p=NULL;
	}
}


////////////////////////////////
float** memAllocFloat2(int row, int col)
{
    // allocate the memory for a 2-dimension array which can be indexed as p[row][col]
    
	int padding=10;
	float *a, **p;
    
	a=(float*) malloc(sizeof(float) * (row * col + padding));
	if(a==NULL) {mexErrMsgTxt("memAllocFloat: Memory allocate failure.\n"); }
	p=(float**) malloc(sizeof(float*) * row);
    
	for(int i=0; i<row; i++) p[i] = &a[i * col];
    
	return(p);
}


//////////////////////////////////
void memFreeFloat2(float **p)
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
void expLUT(float *LUT, float sigma, int chaNum, int len)
{
    float sigma_new = float(chaNum) * 2 * sigma * sigma *65535.0;
    float *ptr=&LUT[0];
    for(int i=0; i<len; i++)   *ptr++ = exp(float(-i) / sigma_new);
    
}


///////////////////////////////////////////////////
void fracLUT(float *LUT, float sigma, int chaNum, int len)
{
    float frac = float(chaNum) * 65535.0;
    float *ptr=&LUT[0];
    for(int i=0; i<len; i++)  *ptr++ = 1 / (pow(sqrt(float(i) / frac), sigma) + 0.00001);
    
}


////////////////////////////////////////////////////
void img2vec_col(float ***img, float ***imgGuide, float **vecImg, float **vecGuide, int step)
{
    float *ptr_img, *ptr_guide, *ptr_vecImg, *ptr_vecGuide;
    int row, col, cha, colSlide, left, right, maxIterColNum;
    bool leftright=true, updown=true, lastpatch=false, shouldbreak=false;
    
    maxIterColNum = colNum - (colNum - 2 * r - 1)%step - r;
    ptr_vecImg = &vecImg[0][0];
    ptr_vecGuide = &vecGuide[0][0];
    ptr_img = &img[0][0][0];
    ptr_guide = &imgGuide[0][0][0];
    
    int diaXChaImg = (2 * r + 1) * chaNumImg;
    int diaXChaGuide = (2 * r + 1) * chaNumGuide;
    int ptrShiftImg = step * chaNumImg;
    int ptrShiftGuide = step * chaNumGuide;
    
    for(col=r; col<maxIterColNum + step; col+=step)
    {
        if(lastpatch)   // the last patch that centers around column "colNum - r"
        {
            shouldbreak = true;
            col = colNum - r - 1;
            
            if(updown){   
                // we inforce leftright=true at the beginning of each patch 
                ptr_img = &img[0][col - r][0];
                ptr_guide = &imgGuide[0][col - r][0];}
            else{
                // we inforce leftright=true at the beginning of each patch
                ptr_img = &img[rowNum - 1][col - r][0];
                ptr_guide = &imgGuide[rowNum - 1][col - r][0];} 
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
                        for(cha=0; cha<chaNumImg; cha++)
                            *ptr_vecImg++ = *ptr_img++;
                        
                        // for the guide image
                        for(cha=0; cha<chaNumGuide; cha++)
                            *ptr_vecGuide++ = *ptr_guide++;   
                    }  
                    
                    // move to the next row
                    ptr_img += ccImgPlusChaNum;
                    ptr_guide += ccGuidePlusChaNum;

                    // change sliding direction between left-to-right and right-to-left
                    leftright = !leftright;  
                }
                else
                {// row right-to-left extraction sliding direction
                    for(colSlide=right; colSlide>=left; colSlide--)
                    {
                        // for the image to be filtered
                        ptr_img -= chaNumImgX2;
                        for(cha=0; cha<chaNumImg; cha++)
                            *ptr_vecImg++ = *ptr_img++;

                        // for the guidance image
                        ptr_guide -= chaNumGuideX2;
                        for(cha=0; cha<chaNumGuide; cha++)
                            *ptr_vecGuide++ = *ptr_guide++;
                    }
                    
                    // move to the next row
                    ptr_img += ccImgMinusChaNum;
                    ptr_guide += ccGuideMinusChaNum;

                    // change sliding direction between left-to-right and right-to-left
                    leftright = !leftright;   
                }  
            }
            
            if(leftright){
                ptr_img -= ccImg - ptrShiftImg;
                ptr_guide -= ccGuide - ptrShiftGuide;}
            else{
                ptr_img -= ccImgPlusChaNum + diaXChaImg - ptrShiftImg;
                ptr_guide -= ccGuidePlusChaNum + diaXChaGuide - ptrShiftGuide;
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
                        for(cha=0; cha<chaNumImg; cha++)
                            *ptr_vecImg++ = *ptr_img++;
                        
                        // for the guide image
                        for(cha=0; cha<chaNumGuide; cha++)
                            *ptr_vecGuide++ = *ptr_guide++;    
                    }  
                    
                    // move to the next row
                    ptr_img -= ccImgMinusChaNum;
                    ptr_guide -= ccGuideMinusChaNum;

                    // change sliding direction between left-to-right and right-to-left
                    leftright = !leftright;  
                }
                else
                {// row right-to-left extraction sliding direction
                    for(colSlide=right; colSlide>=left; colSlide--)
                    {
                        // for the image to be filtered
                        ptr_img -= chaNumImgX2;
                        for(cha=0; cha<chaNumImg; cha++)
                            *ptr_vecImg++ = *ptr_img++;

                        // for the guidance image
                        ptr_guide -= chaNumGuideX2;
                        for(cha=0; cha<chaNumGuide; cha++)
                            *ptr_vecGuide++ = *ptr_guide++;
                    }
                    
                    // move to the next row
                    ptr_img -= ccImgPlusChaNum;
                    ptr_guide -= ccGuidePlusChaNum;

                    // change sliding direction between left-to-right and right-to-left
                    leftright = !leftright;   
                }  
            }
            
            if(leftright){
                ptr_img += ccImg + ptrShiftImg;
                ptr_guide += ccGuide + ptrShiftGuide;}
            else{
                ptr_img += ccImgMinusChaNum - diaXChaImg + ptrShiftImg;
                ptr_guide += ccGuideMinusChaNum - diaXChaGuide + ptrShiftGuide;
                leftright = true;}
            
            updown = !updown;
        }
        
        if(shouldbreak) break;  // break the loop after the last patch
    }
}


//////////////////////////////////////////////////
void img2vec_row(float ***img, float ***imgGuide, float **vecImg, float **vecGuide, int step)
{
    float *ptr_img, *ptr_guide, *ptr_vecImg, *ptr_vecGuide;
    int row, col, cha, rowSlide, maxIterRowNum, up, bottom;
    bool leftright=true, updown=true, lastpatch=false, shouldbreak=false;
    
    maxIterRowNum = rowNum - (rowNum - 2 * r - 1)%step - r;
    ptr_vecImg = &vecImg[0][0];
    ptr_vecGuide = &vecGuide[0][0];
    ptr_img = &img[0][0][0];
    ptr_guide = &imgGuide[0][0][0];
    
    int ptrShiftImg = step * ccImg;
    int ptrShiftGuide = step * ccGuide;
    int diaXCCImg = 2 * r * ccImg;
    int diaXCCGuide = 2 * r * ccGuide;
    
    for(row=r; row<maxIterRowNum + step; row+=step)
    {
        if(lastpatch)
        {
            shouldbreak = true;
            row = rowNum - r - 1;
            
            if(leftright){
                // we inforce updown=true at the beginning of each patch
                ptr_img = &img[row - r][0][0];
                ptr_guide = &imgGuide[row - r][0][0];}
            
            else{
                // we inforce updown=true at the beginning of each patch
                ptr_img = &img[row - r ][colNum - 1][0];
                ptr_guide = &imgGuide[row - r][colNum - 1][0];}   
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
                        for(cha=0; cha<chaNumImg; cha++)
                            *ptr_vecImg++ = *ptr_img++;
                        ptr_img += ccImgMinusChaNum;

                        // for the guidance image
                        for(cha=0; cha<chaNumGuide; cha++)
                            *ptr_vecGuide++ = *ptr_guide++;
                        ptr_guide += ccGuideMinusChaNum;
                    }

                    // move to the next column
                    ptr_img -=  ccImgMinusChaNum;
                    ptr_guide -= ccGuideMinusChaNum;

                    // change the extraction sliding direction between bottom-up and up-bottom in each column of the image
                    updown = !updown; 
                }
                else  // bottom-up extraction sliding direction
                {
                    for(rowSlide=bottom; rowSlide>=up; rowSlide--)
                    {
                        // for the image to be filtered
                        for(cha=0; cha<chaNumImg; cha++) 
                            *ptr_vecImg++ = *ptr_img++;
                        ptr_img -= ccImgPlusChaNum;        

                        // for the guidance image
                        for(cha=0; cha<chaNumGuide; cha++)
                            *ptr_vecGuide++ = *ptr_guide++;
                        ptr_guide -= ccGuidePlusChaNum;
                    }

                    // move to the next column
                    ptr_img +=  ccImgPlusChaNum;
                    ptr_guide += ccGuidePlusChaNum;

                    // change the extraction sliding direction between bottom-up and up-bottom in each column of the image
                    updown = !updown;
                }
            }
            
            if(updown){
                ptr_img += ptrShiftImg - chaNumImg;
                ptr_guide += ptrShiftGuide - chaNumGuide;}
            else{
                ptr_img -= chaNumImg + diaXCCImg - ptrShiftImg;
                ptr_guide -= chaNumGuide + diaXCCGuide - ptrShiftGuide;
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
                        for(cha=0; cha<chaNumImg; cha++)
                            *ptr_vecImg++ = *ptr_img++;
                        ptr_img += ccImgMinusChaNum;

                        // for the guidance image
                        for(cha=0; cha<chaNumGuide; cha++)
                            *ptr_vecGuide++ = *ptr_guide++;
                        ptr_guide += ccGuideMinusChaNum;
                    }

                    // move to the next column
                    ptr_img -=  ccImgPlusChaNum;
                    ptr_guide -= ccGuidePlusChaNum;

                    // change the extraction sliding direction between bottom-up and up-bottom in each column of the image
                    updown = !updown; 
                }
                else  // bottom-up extraction sliding direction
                {
                    for(rowSlide=bottom; rowSlide>=up; rowSlide--)
                    {
                        // for the image to be filtered
                        for(cha=0; cha<chaNumImg; cha++) 
                            *ptr_vecImg++ = *ptr_img++;
                        ptr_img -= ccImgPlusChaNum;        

                        // for the guidance image
                        for(cha=0; cha<chaNumGuide; cha++)
                            *ptr_vecGuide++ = *ptr_guide++;
                        ptr_guide -= ccGuidePlusChaNum;
                    }

                    // move to the next column
                    ptr_img +=  ccImgMinusChaNum;
                    ptr_guide += ccGuideMinusChaNum;

                    // change the extraction sliding direction between bottom-up and up-bottom in each column of the image
                    updown = !updown;
                }
            }
            
            if(updown){
                ptr_img += ptrShiftImg + chaNumImg;
                ptr_guide += ptrShiftGuide + chaNumGuide;}
            else{
                ptr_img += chaNumImg - diaXCCImg + ptrShiftImg;
                ptr_guide += chaNumGuide - diaXCCGuide + ptrShiftGuide;
                updown = true;}
            
            leftright = !leftright; 
        } 
        
        if(shouldbreak) break;  // break the loop after the last patch
    }
}


////////////////////////////////////////////////////
void vec2img_col(float ***img, float **vec, int step)
{
    float *ptr_img, *ptr_vec;
    int row, col, cha, colSlide, left, right, maxIterColNum;
    bool leftright=true, updown=true, lastpatch=false, shouldbreak=false;
    
    maxIterColNum = colNum - (colNum - 2 * r - 1)%step - r;
    ptr_vec = &vec[0][0];
    ptr_img = &img[0][0][0];
    
    int diaXChaImg = (2 * r + 1) * chaNumImg;
    int ptrShiftImg = step * chaNumImg;
    
    for(col=r; col<maxIterColNum + step; col+=step)
    {
        if(lastpatch)   // the last patch that centers around column "colNum - r"
        {
            shouldbreak=true;
            col = colNum - r - 1;
            
            if(updown)   
                // we inforce leftright=true at the beginning of each patch 
                ptr_img = &img[0][col - r][0];
            else  
                // we inforce leftright=true at the beginning of each patch 
                ptr_img = &img[rowNum - 1][col - r][0];
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
                        for(cha=0; cha<chaNumImg; cha++)
                            *ptr_img++ += *ptr_vec++;
                    }  
                    
                    // move to the next row
                    ptr_img += ccImgPlusChaNum;
                    
                    // change sliding direction between left-to-right and right-to-left
                    leftright = !leftright;  
                }
                else
                {// row right-to-left extraction sliding direction
                    for(colSlide=right; colSlide>=left; colSlide--)
                    {
                        // for the image to be filtered
                        ptr_img -= chaNumImgX2;
                        for(cha=0; cha<chaNumImg; cha++)
                            *ptr_img++ += *ptr_vec++;
                    }
                    
                    // move to the next row
                    ptr_img += ccImgMinusChaNum;

                    // change sliding direction between left-to-right and right-to-left
                    leftright = !leftright;      
                }  
            }
            
            if(leftright)
                ptr_img -= ccImg - ptrShiftImg;
            else{
                ptr_img -= ccImgPlusChaNum + diaXChaImg - ptrShiftImg;
                leftright = true;}
            
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
                        for(cha=0; cha<chaNumImg; cha++)
                            *ptr_img++ += *ptr_vec++;
                    }  
                    
                    // move to the next row
                    ptr_img -= ccImgMinusChaNum;

                    // change sliding direction between left-to-right and right-to-left
                    leftright = !leftright;  
                }
                else
                {// row right-to-left extraction sliding direction
                    for(colSlide=right; colSlide>=left; colSlide--)
                    {
                        // for the image to be filtered
                        ptr_img -= chaNumImgX2;
                        for(cha=0; cha<chaNumImg; cha++)
                            *ptr_img++ += *ptr_vec++;
                    }
                    
                    // move to the next row
                    ptr_img -= ccImgPlusChaNum;
                    
                    // change sliding direction between left-to-right and right-to-left
                    leftright = !leftright;      
                }  
            }
            
            if(leftright)
                ptr_img += ccImg + ptrShiftImg;
            else{
                ptr_img += ccImgMinusChaNum - diaXChaImg + ptrShiftImg;
                leftright = true;}
            
            updown = !updown;
        }
        
        if(shouldbreak) break;
    }
}

//////////////////////////////////////////////////
void vec2img_row(float ***img, float **vec, int step)
{
    //////// this function DOES have the "add" function /////////
    float *ptr_img, *ptr_vec;
    int row, col, cha, rowSlide, maxIterRowNum, up, bottom;
    bool leftright=true, updown=true, lastpatch=false, shouldbreak=false;
    
    maxIterRowNum = rowNum - (rowNum - 2 * r - 1)%step - r;
    ptr_vec = &vec[0][0];
    ptr_img = &img[0][0][0];
    
    int ptrShiftImg = step * ccImg;
    int diaXCCImg = 2 * r * ccImg;
   
    for(row=r; row<rowNum; row+=step)
    {
        if(lastpatch)
        {
            shouldbreak = true;
            row = rowNum - r - 1;
            
            if(leftright)
                // we inforce updown=true at the beginning of each patch
                ptr_img = &img[row - r][0][0];
            else
                // we inforce updown=true at the beginning of each patch
                ptr_img = &img[row - r ][colNum - 1][0];
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
                        for(cha=0; cha<chaNumImg; cha++)
                            *ptr_img++ += *ptr_vec++;
                        ptr_img += ccImgMinusChaNum;
                    }

                    // move to the next column
                    ptr_img -=  ccImgMinusChaNum;

                    // change the extraction sliding direction between bottom-up and up-bottom in each column of the image
                    updown = !updown; 
                }
                else  // bottom-up extraction sliding direction
                {
                    for(rowSlide=bottom; rowSlide>=up; rowSlide--)
                    {
                        // for the image to be filtered
                        for(cha=0; cha<chaNumImg; cha++) 
                            *ptr_img++ += *ptr_vec++;
                        ptr_img -= ccImgPlusChaNum;
                    }

                    // move to the next column
                    ptr_img +=  ccImgPlusChaNum;

                    // change the extraction sliding direction between bottom-up and up-bottom in each column of the image
                    updown = !updown;
                }
            }
            
            if(updown)
                ptr_img += ptrShiftImg - chaNumImg;
            else{
                ptr_img -= chaNumImg + diaXCCImg - ptrShiftImg;
                updown = true;}
            
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
                        for(cha=0; cha<chaNumImg; cha++)
                            *ptr_img++ += *ptr_vec++;
                        ptr_img += ccImgMinusChaNum;
                    }

                    // move to the next column
                    ptr_img -=  ccImgPlusChaNum;
                    
                    // change the extraction sliding direction between bottom-up and up-bottom in each column of the image
                    updown = !updown; 
                }
                else  // bottom-up extraction sliding direction
                {
                    for(rowSlide=bottom; rowSlide>=up; rowSlide--)
                    {
                        // for the image to be filtered
                        for(cha=0; cha<chaNumImg; cha++) 
                            *ptr_img++ += *ptr_vec++;
                        ptr_img -= ccImgPlusChaNum; 
                    }

                    // move to the next column
                    ptr_img +=  ccImgMinusChaNum;

                    // change the extraction sliding direction between bottom-up and up-bottom in each column of the image
                    updown = !updown;
                }
            }
            
            if(updown)
                ptr_img += ptrShiftImg + chaNumImg;
            else{
                ptr_img += chaNumImg - diaXCCImg + ptrShiftImg;;
                updown = true;}
            
            leftright = !leftright; 
        } 
        
        if(shouldbreak) break;
    }
}


///////////////////////////////////////
void solver(float **vecImg, float **vecGuide, float **vecInter, float **vecFiltered, float *beta, float *rangeLUT, float lambda, int sysLen)
{
    float *ptr_guide_cur, *ptr_guide_nei, *ptr_beta, *ptr_F, *ptr_Y, *ptr_Y_pre, *ptr_X, *ptr_X_pre;
    float diffR, temp;
    float weightR, a_cur, b_cur, b_pre, tempAlpha, tempGamma, tempBeta;
    int i, k, n=sysLen, colNumGuideVector = chaNumGuide + 2;
    
    ptr_beta = &beta[0];
    ptr_guide_cur = &vecGuide[0][0];  ptr_guide_nei = &vecGuide[1][0]; 
    ptr_F = &vecImg[0][0];  ptr_Y = &vecInter[0][0];  ptr_Y_pre = ptr_Y;
    
    // i = 0
    {
        //////////  get Laplacian //////////
        // range weight
        diffR = 0;
        for(k=0; k<chaNumGuide; k++){
            temp = *ptr_guide_cur++ - *ptr_guide_nei++;
            diffR += temp * temp;}
        weightR = rangeLUT[int(diffR * 65535.0)];

        b_cur = -lambda * weightR;
        a_cur = 1 - b_cur;
        b_pre = b_cur;
        
        ///////////////////// decompose and solve L * Y = F ////////////
        tempAlpha = a_cur;
        
        for(k=0; k<chaNumImg; k++)
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
        diffR = 0;
        for(k=0; k<chaNumGuide; k++){
            temp = *ptr_guide_cur++ - *ptr_guide_nei++;
            diffR += temp * temp;}
        weightR = rangeLUT[int(diffR * 65535.0)];
        
        b_cur = -lambda * weightR;
        a_cur = 1 - b_cur - b_pre;
        b_pre = b_cur;
        
        ///////////////////// decompose  and solve L * Y = F  ////////////
        tempAlpha = a_cur - tempGamma * tempBeta;
        
        for(k=0; k<chaNumImg; k++)
            *ptr_Y++ = (*ptr_F++ - tempGamma * (*ptr_Y_pre++)) / tempAlpha;
        
        tempGamma = b_cur;
        tempBeta = tempGamma / tempAlpha;
        *ptr_beta++ = tempBeta;
    }
    
    // i = n - 1
    {
        a_cur = 1 - b_pre;
        tempAlpha = a_cur - tempGamma * tempBeta;
        
        for(k=0; k<chaNumImg; k++)
           *ptr_Y++ = (*ptr_F++ - tempGamma * (*ptr_Y_pre++)) / tempAlpha;
    }
    
    ///////////  solve U*X = Y ///////////
    ptr_X = &vecFiltered[n-1][chaNumImg - 1];  ptr_X_pre = ptr_X;  ptr_Y = &vecInter[n-1][chaNumImg - 1];
    ptr_beta = &beta[n - 2];
    
    for(k=0; k<chaNumImg; k++)
        *ptr_X-- = *ptr_Y--;
    
    for(i=n-2; i>=0; i--){
        tempBeta = *ptr_beta--;
        for(k=0; k<chaNumImg; k++)
            *ptr_X-- = *ptr_Y-- - tempBeta * (*ptr_X_pre--);}
}


/////////////////////////////////////
void pointDiv(float ***imgInter, float ***imgFiltered, float *count, int colDir)
{
    float *ptr_inter, *ptr_filtered, *ptr_count, value;
    ptr_inter = &imgInter[0][0][0];
    ptr_filtered = &imgFiltered[0][0][0];
    
    if(colDir)
    {
        for(int i=0; i<rowNum; i++){
            ptr_count = &count[0];
            for(int j=0; j<colNum; j++){  
                value = *ptr_count++;
                for(int k=0; k<chaNumImg; k++)
                    *ptr_filtered++ = *ptr_inter++ / value;}}
        
    }
    else
    {
        ptr_count = &count[0];
        for(int i=0; i<rowNum; i++){
            value = *ptr_count++;
            for(int j=0; j<colNum; j++){     
                for(int k=0; k<chaNumImg; k++)
                    *ptr_filtered++ = *ptr_inter++ / value;}}
    }
} 

void getCount(float *count, int len, int r, int step)
{
    int i, j, step_new, maxIterNum;
    float *ptr;
    
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

void mean(float ***A, float ***B, float ***result)
{
    float *ptr_a, *ptr_b, *ptr_result;
    int i, j, k;
    
    ptr_a = &A[0][0][0];
    ptr_b = &B[0][0][0];
    ptr_result = &result[0][0][0];
    
    for(i=0; i<rowNum; i++)
        for(j=0; j<colNum; j++)
            for(k=0; k<chaNumImg; k++)
                *ptr_result++ = (*ptr_a++ + *ptr_b++) / 2.0; 
}
