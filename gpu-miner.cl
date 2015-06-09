void clmemcpy( __private void *dest, __private const void *src, __private size_t num) {
	int i = 0 ;
	char *dest8 = (char*)dest;
	char *src8 = (char*)src;
#pragma unroll
	for (int i = 0; i < num; i++) {
		dest8[i] = src8[i];
	}
}

#if defined(_MSC_VER)
#define ALIGN(x) __declspec(align(x))
#else
#define ALIGN(x) __attribute__((aligned(x)))
#endif


static inline ulong load64( __private const void *src )
{
  return *( ulong * )( src );
}

static inline void store64( __private void *dst, __private ulong w )
{
  *( ulong * )( dst ) = w;
}

static inline ulong rotr64( __private const ulong w, __private const unsigned c )
{
  return ( w >> c ) | ( w << ( 64 - c ) );
}

// blake2b-ref.c
__constant ulong blake2b_IV[8] =
{
	0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
	0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
	0x510e527fade682d1, 0x9b05688c2b3e6c1f,
	0x1f83d9abfb41bd6b, 0x5be0cd19137e2179
};

__constant uchar blake2b_sigma[12][16] =
{
	{	0,	1,	2,	3,	4,	5,	6,	7,	8,	9, 10, 11, 12, 13, 14, 15 } ,
	{ 14, 10,	4,	8,	9, 15, 13,	6,	1, 12,	0,	2, 11,	7,	5,	3 } ,
	{ 11,	8, 12,	0,	5,	2, 15, 13, 10, 14,	3,	6,	7,	1,	9,	4 } ,
	{	7,	9,	3,	1, 13, 12, 11, 14,	2,	6,	5, 10,	4,	0, 15,	8 } ,
	{	9,	0,	5,	7,	2,	4, 10, 15, 14,	1, 11, 12,	6,	8,	3, 13 } ,
	{	2, 12,	6, 10,	0, 11,	8,	3,	4, 13,	7,	5, 15, 14,	1,	9 } ,
	{ 12,	5,	1, 15, 14, 13,	4, 10,	0,	7,	6,	3,	9,	2,	8, 11 } ,
	{ 13, 11,	7, 14, 12,	1,	3,	9,	5,	0, 15,	4,	8,	6,	2, 10 } ,
	{	6, 15, 14,	9, 11,	3,	0,	8, 12,	2, 13,	7,	1,	4, 10,	5 } ,
	{ 10,	2,	8,	4,	7,	6,	1,	5, 15, 11,	9, 14,	3, 12, 13 , 0 } ,
	{	0,	1,	2,	3,	4,	5,	6,	7,	8,	9, 10, 11, 12, 13, 14, 15 } ,
	{ 14, 10,	4,	8,	9, 15, 13,	6,	1, 12,	0,	2, 11,	7,	5,	3 }
};

// The kernel that grinds nonces until it finds a hash below the target
__kernel void nonceGrind(__global uchar *headerIn, __global uchar *hashOut, __global uchar *targetIn, __global uchar *nonceOut) {
	private uchar header[256] = {0};
	private uchar headerHash[64];
	private uchar target[32];
	headerHash[0] = 255;

	int i;
#pragma unroll
	for (i = 0; i < 32; i++) {
		target[i] = targetIn[i];
		header[i] = headerIn[i];
	}
#pragma unroll
	for (i = 32; i < 80; i++) {
		header[i] = headerIn[i];
	}

	// Set nonce
	private int id = get_global_id(0);
	// Support global work sizes of up to 256^4 - 1
	header[32] = id / (256 * 256 * 256);
	header[33] = id / (256 * 256);
	header[34] = id / 256;
	header[35] = id % 256;

	// BLAKE2B START
	ulong h[8] = {0};
	ulong t[2] = {80, 0};
	ulong f[2] = {0};
	for( int i = 0; i < 8; ++i ) h[i] = blake2b_IV[i];
	h[0] ^= 0x0000000001010020UL;
	f[0] = ~((ulong)0);

	ulong m[16];
	ulong v[16];

	for( i = 0; i < 16; ++i )
		m[i] = load64( header + i * sizeof( m[i] ) );

	for( i = 0; i < 8; ++i )
		v[i] = h[i];

	v[ 8] = blake2b_IV[0];
	v[ 9] = blake2b_IV[1];
	v[10] = blake2b_IV[2];
	v[11] = blake2b_IV[3];
	v[12] = t[0] ^ blake2b_IV[4];
	v[13] = t[1] ^ blake2b_IV[5];
	v[14] = f[0] ^ blake2b_IV[6];
	v[15] = f[1] ^ blake2b_IV[7];
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
#define ROUND(r)	\
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
	ROUND( 0 );
	ROUND( 1 );
	ROUND( 2 );
	ROUND( 3 );
	ROUND( 4 );
	ROUND( 5 );
	ROUND( 6 );
	ROUND( 7 );
	ROUND( 8 );
	ROUND( 9 );
	ROUND( 10 );
	ROUND( 11 );

	for( i = 0; i < 8; ++i )
		h[i] = h[i] ^ v[i] ^ v[i + 8];

#undef G
#undef ROUND

	for( int i = 0; i < 8; ++i ) // Output full hash to temp buffer
		store64( headerHash + sizeof( h[i] ) * i, h[i] );
	// BLAKE2B END

	// Compare header to target
	int z = 0;
	while (target[z] == headerHash[z]) {
		z++;
	}
	if (headerHash[z] < target[z]) {
		// Transfer the output to global space.
#pragma unroll
		for (i = 0; i < 8; i++) {
			nonceOut[i] = header[i + 32];
		}
#pragma unroll
		for (i = 0; i < 32; i++) {
			hashOut[i] = headerHash[i];
		}
		return;
	}
}
