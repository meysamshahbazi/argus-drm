
#include "cudaDeinterlace.h"


// orginal 
__global__ void gpuDeinterlace(unsigned char* input_cur, unsigned char* filed_buf,unsigned char* output,
 unsigned char *bottom_field, size_t width, size_t height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	int pitch = 3072; //1536;// this is very important and not necesserly is equal to 2*width !!

	__shared__ unsigned char field_flag;
	// field_flag = input_cur[721*2+2];//
	field_flag = input_cur[y*pitch + 720*4 + 0];//
	// if(x==0 && y ==0 ) {
	// 	for (int i = 0; i < 12; i++ )
	// 		printf("%d\t", input_cur[y*pitch + 720*4+i]);
	// 	printf("\n\n");
	// }

	if(x >= 4*760 || y >= 288)
		return;

	// this work 
	if(field_flag == 0){
		filed_buf[y*pitch+x] = input_cur[y*pitch+x]; 
		if(x==0 && y ==0 ) *bottom_field = 0;
	}
	else if(field_flag != 0) {
		output[(2*y+1)*pitch + x] =  input_cur[y*pitch+x];
		if(x==0 && y ==0 ) *bottom_field = 1;
	}

	// __syncthreads();
	if(field_flag != 0){
		output[2*y*pitch + x] = filed_buf[y*pitch+x]; 

	}
}

// __global__ void gpuDeinterlaceTest(unsigned char* input_cur, unsigned char* filed_buf,unsigned char* output,
//  unsigned char *bottom_field, size_t width, size_t height)
// {
// 	const int x = blockIdx.x * blockDim.x + threadIdx.x;
// 	const int y = blockIdx.y * blockDim.y + threadIdx.y;

// 	int pitch = 3072; //1536;// this is very important and not necesserly is equal to 2*width !!

// 	__shared__ unsigned char field_flag;
// 	// field_flag = input_cur[721*2+2];//
// 	field_flag = input_cur[y*pitch + 720*4 + 0];//
// 	// if(x==0 && y ==0 ) {
// 	// 	for (int i = 0; i < 12; i++ )
// 	// 		printf("%d\t", input_cur[y*pitch + 720*4+i]);
// 	// 	printf("\n\n");
// 	// }

// 	if(x >= 4*760 || y >= 288)
// 		return;

// 	// this work 
// 	if(field_flag == 0){
// 		filed_buf[y*pitch+x] = input_cur[y*pitch+x]; 
// 		if(x==0 && y ==0 ) *bottom_field = 0;
// 	}
// 	else if(field_flag != 0) {
// 		output[y*pitch + x] =  input_cur[y*pitch+x];
// 		if(x==0 && y ==0 ) *bottom_field = 1;
// 	}

// 	if(field_flag != 0){
// 		output[(288 + y)*pitch + x] = filed_buf[y*pitch+x]; 

// 	}
// }

cudaError_t cudaDeinterlace( void* input_cur,void* filed_buf, void* output,
unsigned char *bottom_field, size_t width, size_t height, cudaStream_t stream)
{	
	const dim3 blockDim(32, 8,1);
	const dim3 gridDim(iDivUp(4*760,blockDim.x), iDivUp(288,blockDim.y),1); // 736 * 288

	// const dim3 blockDim(1, 8,1);
	// const dim3 gridDim(iDivUp(1,blockDim.x), iDivUp(288,blockDim.y),1); // 736 * 288
	unsigned char *d_bottom_field;
	cudaMalloc(&d_bottom_field,1);
	gpuDeinterlace<<<gridDim, blockDim, 0, stream>>>(
						(unsigned char *) input_cur, 
						(unsigned char *) filed_buf,
						(unsigned char *) output, d_bottom_field, width, height);// TODO: add pitch

	cudaMemcpy(bottom_field,d_bottom_field,1,cudaMemcpyDeviceToHost);
	cudaFree(d_bottom_field);
	return cudaGetLastError();
}

