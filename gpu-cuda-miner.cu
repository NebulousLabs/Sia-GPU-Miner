#include <cstdint>
#include <cuda_runtime.h>

__device__ void blake2b(uint8_t *out, const uint8_t *in);

// The kernel that grinds nonces until it finds a hash below the target
__global__ void nonceGrind(uint8_t *headerIn, uint8_t *hashOut, uint8_t *targ, uint8_t *nonceOut)
{
	uint8_t header[80];
	uint8_t headerHash[32];
	uint8_t target[32];

	headerHash[0] = 255;

	int i, z;
	for(i = 0; i < 32; i++)
	{
		target[i] = targ[i];
		header[i] = headerIn[i];
	}
	for(i = 32; i < 80; i++)
	{
		header[i] = headerIn[i];
	}

	// Set nonce
	const int id = blockDim.x * blockIdx.x + threadIdx.x;
	header[32] = id / (256 * 256 * 256);
	header[33] = id / (256 * 256);
	header[34] = id / 256;
	header[35] = id % 256;

	// Hash the header
	blake2b(headerHash, header);

	// Compare header to target
	z = 0;
	while(target[z] == headerHash[z])
	{
		z++;
	}
	if(headerHash[z] < target[z])
	{
		// Transfer the output to global space.
		for(i = 0; i < 8; i++)
		{
			nonceOut[i] = header[i + 32];
		}
		for(i = 0; i < 32; i++)
		{
			hashOut[i] = headerHash[i];
		}
		return;
	}
}

void nonceGrindcuda(cudaStream_t cudastream, int gridsize, int blocksize, char *blockHeader, char *headerHash, char *targ, char *nonceOut)
{
	nonceGrind << <gridsize, blocksize, 0, cudastream >> >((uint8_t*)blockHeader, (uint8_t*)headerHash, (uint8_t*)targ, (uint8_t*)nonceOut);
}


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

__device__ void clmemcpy(void *dest, const void *src, size_t num)
{
	char *dest8 = (char*)dest;
	char *src8 = (char*)src;
	for(int i = 0; i < num; i++)
	{
		dest8[i] = src8[i];
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

__device__ static inline uint64_t load64(const void *src)
{
	return *(uint64_t *)(src);
}

__device__ static inline void store64(void *dst, uint64_t w)
{
	*(uint64_t *)(dst) = w;
}

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

__device__ static void blake2b_compress(blake2b_state *S, const uint8_t block[BLAKE2B_BLOCKBYTES])
{
	uint64_t m[16];
	uint64_t v[16];
	int i;

	for(i = 0; i < 16; ++i)
		m[i] = load64(block + i * sizeof(m[i]));

	for(i = 0; i < 8; ++i)
		v[i] = S->h[i];

	v[8] = blake2b_IV[0];
	v[9] = blake2b_IV[1];
	v[10] = blake2b_IV[2];
	v[11] = blake2b_IV[3];
	v[12] = S->t[0] ^ blake2b_IV[4];
	v[13] = S->t[1] ^ blake2b_IV[5];
	v[14] = S->f[0] ^ blake2b_IV[6];
	v[15] = S->f[1] ^ blake2b_IV[7];

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
		S->h[i] = S->h[i] ^ v[i] ^ v[i + 8];

#undef G
#undef ROUND
}

__device__ void blake2b(uint8_t *out, const uint8_t *in)
{
	blake2b_state S[1];

	clmemset(S, 0, sizeof(blake2b_state));
	for(int i = 0; i < 8; ++i) S->h[i] = blake2b_IV[i];
	S->h[0] ^= 0x0000000001010020UL;

	uint64_t inlen = 80;
	size_t left = S->buflen;
	size_t fill = 2 * BLAKE2B_BLOCKBYTES - left;

	if(inlen > fill)
	{
		clmemcpy(S->buf + left, in, fill); // Fill buffer
		S->buflen += fill;
		blake2b_compress(S, S->buf); // Compress
		clmemcpy(S->buf, S->buf + BLAKE2B_BLOCKBYTES, BLAKE2B_BLOCKBYTES); // Shift buffer left
		S->buflen -= BLAKE2B_BLOCKBYTES;
	}
	else // inlen <= fill
	{
		clmemcpy(S->buf + left, in, inlen);
		S->buflen += inlen; // Be lazy, do not compress
	}


	S->t[0] += S->buflen;
	S->f[0] = ~((uint64_t)0);
	clmemset(S->buf + S->buflen, 0, 2 * BLAKE2B_BLOCKBYTES - S->buflen); // Padding
	blake2b_compress(S, S->buf);

	uint8_t buffer[BLAKE2B_OUTBYTES];
	for(int i = 0; i < 8; ++i) // Output full hash to temp buffer
		store64(buffer + sizeof(S->h[i]) * i, S->h[i]);

	clmemcpy(out, buffer, 32);
}
