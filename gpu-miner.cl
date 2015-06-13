static inline ulong rotr64( __const ulong w, __const unsigned c ) { return ( w >> c ) | ( w << ( 64 - c ) ); }

// The kernel that grinds nonces until it finds a hash below the target
__kernel void nonceGrind(__global ulong *headerIn, __global ulong *hashOut, __global uchar *targetIn, __global ulong *nonceOut) {
	ulong m[16] = {0};
#pragma unroll
	for(int i = 0; i < 10; i++ )
		m[i] = headerIn[i];

	uchar nonce[8] = {0};
	*(ulong*)(nonce) = headerIn[4];
	*(uint*)(nonce) = get_global_id(0);
	m[4] = *(ulong*)(nonce);             // Removing this breaks the program.
	*(uint*)(m + 32) = get_global_id(0); // Removing this does not. (Why?)

	ulong v[16] = { 0x6a09e667f2bdc928, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
                    0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
					0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
					0x510e527fade68281, 0x9b05688c2b3e6c1f, 0xe07c265404be4294, 0x5be0cd19137e2179 };
	uchar blake2b_sigma[12][16] = {
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
	a = a + b + m[blake2b_sigma[r][2*i]]; \
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
#undef G
#undef ROUND

	// Surely there is a way to merge iv and headerHash, but I haven't figured it out.
	int i = 0;
	ulong iv[2] = { 0x6a09e667f2bdc928, 0xbb67ae8584caa73b };
	iv[0] = iv[0] ^ v[0] ^ v[8];
	iv[1] = iv[1] ^ v[1] ^ v[9];
	uchar headerHash[64];
	*(ulong*)(headerHash) = iv[0];
	*(ulong*)(headerHash + 8) = iv[1];
	while (targetIn[i] == headerHash[i]) {
		i++;
	}
	if (headerHash[i] < targetIn[i]) {
		// Transfer the output to global space.
		nonceOut[0] = *(ulong*)(nonce);
		hashOut[0] = iv[0];
		hashOut[1] = iv[1];
		return;
	}
}
