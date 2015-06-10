#include <cstdint>
#include <cuda_runtime.h>

#ifdef __INTELLISENSE__
#define __launch_bounds__(blocksize)
#endif
// Implementations of clmemset and memcopy
__device__ void *clmemset(void *s, uint8_t c, size_t n)
{
	uint8_t *p = (uint8_t*)s;
	while(n--)
	{
		*p++ = c;
	}
	return s;
}

__device__ void clmemcpy(uint64_t *const __restrict__ dest, const uint64_t *const __restrict__ src, size_t num)
{
	for(int i = 0; i < num / 8; i++)
	{
		dest[i] = src[i];
	}
}

#if defined(_MSC_VER)
#define ALIGN(x) __declspec(align(x))
#else
#define ALIGN(x) __attribute__((aligned(x)))
#endif

enum blake2b_constant
{
	BLAKE2B_BLOCKBYTES = 128,
	BLAKE2B_OUTBYTES = 64,
	BLAKE2B_KEYBYTES = 64,
	BLAKE2B_SALTBYTES = 16,
	BLAKE2B_PERSONALBYTES = 16
};
/*
#pragma pack(push, 1)
ALIGN(64) typedef struct __blake2b_state
{
uint64_t h[8];
uint64_t t[2];
uint64_t f[2];
uint8_t  buf[2 * BLAKE2B_BLOCKBYTES];
size_t   buflen;
uint8_t  last_node;
} blake2b_state;
#pragma pack(pop)

// Streaming API
int blake2b_update(blake2b_state *S, const uint8_t *in, uint64_t inlen);
int blake2b_final(blake2b_state *S, uint8_t *out);
*/
#if __CUDA_ARCH__ >= 320
__device__ __forceinline__
uint64_t rotr64(const uint64_t value, const int offset)
{
	uint2 result;
	if(offset < 32)
	{
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
	}
	else
	{
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
	}
	return __double_as_longlong(__hiloint2double(result.y, result.x));
}
#else
__device__ __forceinline__
uint64_t rotr64(const uint64_t x, const int offset)
{
	uint64_t result;
	asm("{\n\t"
		".reg .b64 lhs;\n\t"
		".reg .u32 roff;\n\t"
		"shr.b64 lhs, %1, %2;\n\t"
		"sub.u32 roff, 64, %2;\n\t"
		"shl.b64 %0, %1, roff;\n\t"
		"add.u64 %0, %0, lhs;\n\t"
		"}\n"
		: "=l"(result) : "l"(x), "r"(offset));
	return result;
}
#endif

__constant__ uint64_t blake2b_IV[8] =
{
	0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
	0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
	0x510e527fade682d1, 0x9b05688c2b3e6c1f,
	0x1f83d9abfb41bd6b, 0x5be0cd19137e2179
};

__constant__ uint8_t blake2b_sigma[12][16] =
{
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 }
};

__device__ static void blake2b_compress(uint64_t *const __restrict__ h, const uint64_t *const __restrict__ block)
{
	uint64_t m[16];
	uint64_t v[16];
	int i;

	for(i = 0; i < 16; ++i)
		m[i] = block[i];

	for(i = 0; i < 8; ++i)
		v[i] = h[i];

	v[8] = 0x6a09e667f3bcc908;
	v[9] = 0xbb67ae8584caa73b;
	v[10] = 0x3c6ef372fe94f82b;
	v[11] = 0xa54ff53a5f1d36f1;
	v[12] = 80 ^ 0x510e527fade682d1;
	v[13] = 0x9b05688c2b3e6c1f;
	v[14] = ~0x1f83d9abfb41bd6b;
	v[15] = 0x5be0cd19137e2179;

#define G(r,i,a,b,c,d) \
	do { \
		a = a + b + m[blake2b_sigma[r][2*i+0]]; \
		d = rotr64(d ^ a, 32); \
		c = c + d; \
		b = rotr64(b ^ c, 24); \
		a = a + b + m[blake2b_sigma[r][2*i+1]]; \
		d = rotr64(d ^ a, 16); \
		c = c + d; \
		b = rotr64(b ^ c, 63); \
			} while(0)

#define ROUND(r,v)	\
	do { \
		G(r,0,v[ 0],v[ 4],v[ 8],v[12]); \
		G(r,1,v[ 1],v[ 5],v[ 9],v[13]); \
		G(r,2,v[ 2],v[ 6],v[10],v[14]); \
		G(r,3,v[ 3],v[ 7],v[11],v[15]); \
		G(r,4,v[ 0],v[ 5],v[10],v[15]); \
		G(r,5,v[ 1],v[ 6],v[11],v[12]); \
		G(r,6,v[ 2],v[ 7],v[ 8],v[13]); \
		G(r,7,v[ 3],v[ 4],v[ 9],v[14]); \
			} while(0)

	ROUND(0, v);
	ROUND(1, v);
	ROUND(2, v);
	ROUND(3, v);
	ROUND(4, v);
	ROUND(5, v);
	ROUND(6, v);
	ROUND(7, v);
	ROUND(8, v);
	ROUND(9, v);
	ROUND(10, v);
	ROUND(11, v);

	for(i = 0; i < 8; ++i)
		h[i] = h[i] ^ v[i] ^ v[i + 8];

#undef G
#undef ROUND
}

#define blocksize 512

__global__ void __launch_bounds__(blocksize) nonceGrind(uint8_t *const __restrict__ headerIn, uint8_t *const __restrict__ hashOut, const uint8_t *const __restrict__ targ, uint8_t *const __restrict__ nonceOut)
{
	uint8_t headerHash[32];
	int i, z;

	// Set nonce
	const uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
	*((uint32_t*)(headerIn+32)) = id;

	uint64_t *out = (uint64_t*)headerHash;
	uint64_t *in = (uint64_t*)headerIn;
	uint64_t buf[32] = { 0 };
	uint64_t h[8] =
	{
		0x6A09E667F2BDC928, 0xbb67ae8584caa73b,
		0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
		0x510e527fade682d1, 0x9b05688c2b3e6c1f,
		0x1f83d9abfb41bd6b, 0x5be0cd19137e2179
	};

#pragma unroll
	for(i = 0; i < 10; i++)
		buf[i] = in[i];

#pragma unroll
	for(i = 10; i < 32; i++)
		buf[i] = 0;

	blake2b_compress(h, buf);

#pragma unroll
	for(i = 0; i < 4; i++)
		out[i] = h[i];

	// Compare header to target
	z = 0;
	while(targ[z] == headerHash[z])
	{
		z++;
	}
	if(headerHash[z] < targ[z])
	{
		// Transfer the output to global space.
		for(i = 0; i < 8; i++)
		{
			nonceOut[i] = headerIn[i + 32];
		}
		for(i = 0; i < 32; i++)
		{
			hashOut[i] = headerHash[i];
		}
		return;
	}
}

void nonceGrindcuda(cudaStream_t cudastream, int threads, char *blockHeader, char *headerHash, char *targ, char *nonceOut)
{
	nonceGrind << <threads / blocksize, blocksize, 0, cudastream >> >((uint8_t*)blockHeader, (uint8_t*)headerHash, (uint8_t*)targ, (uint8_t*)nonceOut);
}


