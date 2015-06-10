static inline ulong rotr64( __const ulong w, __const unsigned c )
{
  return ( w >> c ) | ( w << ( 64 - c ) );
}

// The kernel that grinds nonces until it finds a hash below the target
__kernel void nonceGrind(__global uint *headerIn, __global uchar *hashOut, __global uint *targetIn, __global uchar *nonceOut) {
	uchar header[256] = {0};
	uchar target[16];

	// Transfer inputs from global memory
	int i;
	for (i = 0; i < 4; i++) {
		*(uint*)(target + i * 4) = targetIn[i];
	}
	for (i = 0; i < 20; i++) {
		*(uint*)(header + i * 4) = headerIn[i];
	}

	// Set nonce
	*(uint*)(header + 32) = get_global_id(0);

	uchar blake2b_sigma[12][16] =
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
#define G(r,i,a,b,c,d) \
	a = a + b + m[blake2b_sigma[r][2*i+0]]; \
	d = rotr64(d ^ a, 32); \
	c = c + d; \
	b = rotr64(b ^ c, 24); \
	a = a + b + m[blake2b_sigma[r][2*i+1]]; \
	d = rotr64(d ^ a, 16); \
	c = c + d; \
	b = rotr64(b ^ c, 63);
#define ROUND(r)	\
	G(r,0,v[ 0],v[ 4],v[ 8],v[12]); \
	G(r,1,v[ 1],v[ 5],v[ 9],v[13]); \
	G(r,2,v[ 2],v[ 6],v[10],v[14]); \
	G(r,3,v[ 3],v[ 7],v[11],v[15]); \
	G(r,4,v[ 0],v[ 5],v[10],v[15]); \
	G(r,5,v[ 1],v[ 6],v[11],v[12]); \
	G(r,6,v[ 2],v[ 7],v[ 8],v[13]); \
	G(r,7,v[ 3],v[ 4],v[ 9],v[14]);
	// BLAKE2B START

	ulong iv[8] = { 0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1, 0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179 };
	ulong v[16];
	ulong t[2] = {80, 0};
	ulong f[2] = {~0, 0};
	ulong m[16];
	for( i = 0; i < 16; ++i )
		m[i] = *(ulong*)( header + i * 8 );
	v[ 8] = iv[0];
	v[ 9] = iv[1];
	v[10] = iv[2];
	v[11] = iv[3];
	v[12] = t[0] ^ iv[4];
	v[13] = iv[5];
	v[14] = f[0] ^ iv[6];
	v[15] = iv[7];
	iv[0] ^= 0x0000000001010020UL;
	for( i = 0; i < 8; ++i )
		v[i] = iv[i];
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
	uchar headerHash[64];
	for( i = 0; i < 8; ++i )
		iv[i] = iv[i] ^ v[i] ^ v[i + 8];
	for( int i = 0; i < 8; ++i ) // Output full hash to temp buffer
		*(ulong*)(headerHash + 8 * i) = iv[i];

	// BLAKE2B END
#undef G
#undef ROUND

	// Compare hash to target
	i = 0;
	while (target[i] == headerHash[i]) {
		i++;
	}
	if (headerHash[i] < target[i]) {
		// Transfer the output to global space.
		for (i = 0; i < 8; i++) {
			nonceOut[i] = header[i + 32];
		}
		for (i = 0; i < 16; i++) {
			hashOut[i] = headerHash[i];
		}
		return;
	}
}
