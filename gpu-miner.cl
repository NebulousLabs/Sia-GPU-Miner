int blake2b( uchar *out, uchar *in );

// The kernel that grinds nonces until it finds a hash below the target
__kernel void nonceGrind(__global uchar *headerIn, __global uchar *hashOut, __global uchar *targ, __global uchar *nonceOut, __global bool *nonceOutLock, __global uint *numItersIn) {
	private uchar blockHeader[80];
	private uchar headerHash[32];
	private uchar target[32];
	private uint numOuterIter = *numItersIn / 256;
	headerHash[0] = 255;

	// Copy header to private memory
	private int i, j, z;
	for (i = 0; i < 80; i++) {
		blockHeader[i] = headerIn[i];
	}

	// Set nonce
	private int id = get_global_id(0);
	blockHeader[32] = id / (256 * 256 * 256);
	blockHeader[33] = id / (256 * 256);
	blockHeader[34] = id / 256;
	blockHeader[35] = id % 256;

	for (i = 0; i < 32; i++) {
		target[i] = targ[i];
	}

	// Grind nonce values
	for (i = 0; i < numOuterIter; i++) {
		// inc nonce
		blockHeader[38] = i;
		for (j = 0; j < 256; j++) {
			blockHeader[39] = j;

			// Hash the header
			blake2b(headerHash, blockHeader);

			// Compare header to target
			z = 0;
			while (target[z] == headerHash[z]) {
				z++;
			}
			if (headerHash[z] < target[z]) {
				// Transfer the output to global space.
				if (!(*nonceOutLock)) {
					*nonceOutLock = true;
					for (i = 0; i < 8; i++) {
						nonceOut[i] = blockHeader[i + 32];
					}
					for (i = 0; i < 32; i++) {
						hashOut[i] = headerHash[i];
					}
					// No reason to unlock (for now)
				}
				return;
			}
		}
		// Check if a disserent thread found the hash
		if (*nonceOutLock) {
			return;
		}
	}
}

// Implementations of clmemset and memcopy
void *clmemset( __private void *s, __private int c, __private size_t n) {
	uchar *p = s;
	while(n--) {
		*p++ = (uchar)c;
	}
	return s;
}

void clmemcpy( __private void *dest, __private const void *src, __private size_t num) {
	int i = 0 ;
	char *dest8 = (char*)dest;
	char *src8 = (char*)src;
	for (int i = 0; i < num; i++) {
		dest8[i] = src8[i];
	}
}

// the code taken from offical Blake2b C reference:
#ifndef __BLAKE2B__
#define __BLAKE2B__

// blake2.h

#if defined(_MSC_VER)
#define ALIGN(x) __declspec(align(x))
#else
#define ALIGN(x) __attribute__((aligned(x)))
#endif

#if defined(__cplusplus)
extern "C" {
#endif

  enum blake2b_constant
  {
	BLAKE2B_BLOCKBYTES = 128,
	BLAKE2B_OUTBYTES   = 64,
	BLAKE2B_KEYBYTES   = 64,
	BLAKE2B_SALTBYTES  = 16,
	BLAKE2B_PERSONALBYTES = 16
  };

#pragma pack(push, 1)
  typedef struct __blake2b_param
  {
	uchar  digest_length; // 1
	uchar  key_length;	// 2
	uchar  fanout;		// 3
	uchar  depth;		 // 4
	uint leaf_length;   // 8
	ulong node_offset;   // 16
	uchar  node_depth;	// 17
	uchar  inner_length;  // 18
	uchar  reserved[14];  // 32
	uchar  salt[BLAKE2B_SALTBYTES]; // 48
	uchar  personal[BLAKE2B_PERSONALBYTES];  // 64
  } blake2b_param;

  ALIGN( 64 ) typedef struct __blake2b_state
  {
	ulong h[8];
	ulong t[2];
	ulong f[2];
	uchar  buf[2 * BLAKE2B_BLOCKBYTES];
	size_t   buflen;
	uchar  last_node;
  } blake2b_state;

#pragma pack(pop)

  // Streaming API
  int blake2b_init( __private blake2b_state *S );
  int blake2b_init_key( __private blake2b_state *S, __private const uchar outlen, __private const void *key, __private const uchar keylen );
  int blake2b_init_param( __private blake2b_state *S, __private const blake2b_param *P );
  int blake2b_update( __private blake2b_state *S, __private const uchar *in, __private ulong inlen );
  int blake2b_final( __private blake2b_state *S, __private uchar *out );

#if defined(__cplusplus)
}
#endif

// blake2-impl.c

static inline uint load32( __private const void *src )
{
#if defined(NATIVE_LITTLE_ENDIAN)
  return *( uint * )( src );
#else
  const uchar *p = ( uchar * )src;
  uint w = *p++;
  w |= ( uint )( *p++ ) <<  8;
  w |= ( uint )( *p++ ) << 16;
  w |= ( uint )( *p++ ) << 24;
  return w;
#endif
}

static inline ulong load64( __private const void *src )
{
#if defined(NATIVE_LITTLE_ENDIAN)
  return *( ulong * )( src );
#else
  const uchar *p = ( uchar * )src;
  ulong w = *p++;
  w |= ( ulong )( *p++ ) <<  8;
  w |= ( ulong )( *p++ ) << 16;
  w |= ( ulong )( *p++ ) << 24;
  w |= ( ulong )( *p++ ) << 32;
  w |= ( ulong )( *p++ ) << 40;
  w |= ( ulong )( *p++ ) << 48;
  w |= ( ulong )( *p++ ) << 56;
  return w;
#endif
}

static inline void store32( __private void *dst, __private uint w )
{
#if defined(NATIVE_LITTLE_ENDIAN)
  *( uint * )( dst ) = w;
#else
  uchar *p = ( uchar * )dst;
  *p++ = ( uchar )w; w >>= 8;
  *p++ = ( uchar )w; w >>= 8;
  *p++ = ( uchar )w; w >>= 8;
  *p++ = ( uchar )w;
#endif
}

static inline void store64( __private void *dst, __private ulong w )
{
#if defined(NATIVE_LITTLE_ENDIAN)
  *( ulong * )( dst ) = w;
#else
  uchar *p = ( uchar * )dst;
  *p++ = ( uchar )w; w >>= 8;
  *p++ = ( uchar )w; w >>= 8;
  *p++ = ( uchar )w; w >>= 8;
  *p++ = ( uchar )w; w >>= 8;
  *p++ = ( uchar )w; w >>= 8;
  *p++ = ( uchar )w; w >>= 8;
  *p++ = ( uchar )w; w >>= 8;
  *p++ = ( uchar )w;
#endif
}

static inline ulong load48( __private const void *src )
{
  const uchar *p = ( const uchar * )src;
  ulong w = *p++;
  w |= ( ulong )( *p++ ) <<  8;
  w |= ( ulong )( *p++ ) << 16;
  w |= ( ulong )( *p++ ) << 24;
  w |= ( ulong )( *p++ ) << 32;
  w |= ( ulong )( *p++ ) << 40;
  return w;
}

static inline void store48( __private void *dst, __private ulong w )
{
  uchar *p = ( uchar * )dst;
  *p++ = ( uchar )w; w >>= 8;
  *p++ = ( uchar )w; w >>= 8;
  *p++ = ( uchar )w; w >>= 8;
  *p++ = ( uchar )w; w >>= 8;
  *p++ = ( uchar )w; w >>= 8;
  *p++ = ( uchar )w;
}

static inline uint rotl32( __private const uint w, __private const unsigned c )
{
  return ( w << c ) | ( w >> ( 32 - c ) );
}

static inline ulong rotl64( __private const ulong w, __private const unsigned c )
{
  return ( w << c ) | ( w >> ( 64 - c ) );
}

static inline uint rotr32( __private const uint w, __private const unsigned c )
{
  return ( w >> c ) | ( w << ( 32 - c ) );
}

static inline ulong rotr64( __private const ulong w, __private const unsigned c )
{
  return ( w >> c ) | ( w << ( 64 - c ) );
}

// prevents compiler optimizing out clmemset()
static inline void secure_zero_memory( __private void *v, __private size_t n )
{
  volatile uchar *p = ( volatile uchar * )v;

  while( n-- ) *p++ = 0;
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

static inline int blake2b_set_lastnode( __private blake2b_state *S )
{
	S->f[1] = ~((ulong)0);
	return 0;
}

static inline int blake2b_clear_lastnode( __private blake2b_state *S )
{
	S->f[1] = ((ulong)0);
	return 0;
}

// Some helper functions, not necessarily useful
static inline int blake2b_set_lastblock( __private blake2b_state *S )
{
	if( S->last_node ) blake2b_set_lastnode( S );

	S->f[0] = ~((ulong)0);
	return 0;
}

static inline int blake2b_clear_lastblock( __private blake2b_state *S )
{
	if( S->last_node ) blake2b_clear_lastnode( S );

	S->f[0] = ((ulong)0);
	return 0;
}

static inline int blake2b_increment_counter( __private blake2b_state *S, __private const ulong inc )
{
	S->t[0] += inc;
	S->t[1] += ( S->t[0] < inc );
	return 0;
}

static inline int blake2b_init0( __private blake2b_state *S )
{
	clmemset( S, 0, sizeof( blake2b_state ) );

	for( int i = 0; i < 8; ++i ) S->h[i] = blake2b_IV[i];

	return 0;
}

// init xors IV with input parameter block
int blake2b_init_param( __private blake2b_state *S, __private const blake2b_param *P )
{
	blake2b_init0( S );
	uchar *p = ( uchar * )( P );

	// IV XOR ParamBlock
	for( size_t i = 0; i < 8; ++i )
		S->h[i] ^= load64( p + sizeof( S->h[i] ) * i );

	return 0;
}


int blake2b_init( __private blake2b_state *S )
{
	blake2b_param P[1];

	P->digest_length = 32;
	P->key_length = 0;
	P->fanout = 1;
	P->depth = 1;
	store32( &P->leaf_length, 0 );
	store64( &P->node_offset, 0 );
	P->node_depth = 0;
	P->inner_length = 0;
	clmemset( P->reserved, 0, sizeof( P->reserved ) );
	clmemset( P->salt,		 0, sizeof( P->salt ) );
	clmemset( P->personal, 0, sizeof( P->personal ) );
	return blake2b_init_param( S, P );
}


int blake2b_init_key( __private blake2b_state *S, __private const uchar outlen, __private const void *key, __private const uchar keylen )
{
	blake2b_param P[1];

	if ( ( !outlen ) || ( outlen > BLAKE2B_OUTBYTES ) ) return -1;

	if ( !key || !keylen || keylen > BLAKE2B_KEYBYTES ) return -1;

	P->digest_length = outlen;
	P->key_length		= keylen;
	P->fanout				= 1;
	P->depth				 = 1;
	store32( &P->leaf_length, 0 );
	store64( &P->node_offset, 0 );
	P->node_depth		= 0;
	P->inner_length	= 0;
	clmemset( P->reserved, 0, sizeof( P->reserved ) );
	clmemset( P->salt,		 0, sizeof( P->salt ) );
	clmemset( P->personal, 0, sizeof( P->personal ) );

	if( blake2b_init_param( S, P ) < 0 ) return -1;

	{
		uchar block[BLAKE2B_BLOCKBYTES];
		clmemset( block, 0, BLAKE2B_BLOCKBYTES );
		clmemcpy( block, key, keylen );
		blake2b_update( S, block, BLAKE2B_BLOCKBYTES );
		secure_zero_memory( block, BLAKE2B_BLOCKBYTES ); // Burn the key from stack
	}
	return 0;
}

static int blake2b_compress( __private blake2b_state *S, __private const uchar block[BLAKE2B_BLOCKBYTES] )
{
	ulong m[16];
	ulong v[16];
	int i;

	for( i = 0; i < 16; ++i )
		m[i] = load64( block + i * sizeof( m[i] ) );

	for( i = 0; i < 8; ++i )
		v[i] = S->h[i];

	v[ 8] = blake2b_IV[0];
	v[ 9] = blake2b_IV[1];
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
		S->h[i] = S->h[i] ^ v[i] ^ v[i + 8];

#undef G
#undef ROUND
	return 0;
}

// inlen now in bytes
int blake2b_update( __private blake2b_state *S, __private const uchar *in, __private ulong inlen )
{
	while( inlen > 0 )
	{
		size_t left = S->buflen;
		size_t fill = 2 * BLAKE2B_BLOCKBYTES - left;

		if( inlen > fill )
		{
			clmemcpy( S->buf + left, in, fill ); // Fill buffer
			S->buflen += fill;
			blake2b_increment_counter( S, BLAKE2B_BLOCKBYTES );
			blake2b_compress( S, S->buf ); // Compress
			clmemcpy( S->buf, S->buf + BLAKE2B_BLOCKBYTES, BLAKE2B_BLOCKBYTES ); // Shift buffer left
			S->buflen -= BLAKE2B_BLOCKBYTES;
			in += fill;
			inlen -= fill;
		}
		else // inlen <= fill
		{
			clmemcpy( S->buf + left, in, inlen );
			S->buflen += inlen; // Be lazy, do not compress
			in += inlen;
			inlen -= inlen;
		}
	}

	return 0;
}

// Is this correct?
int blake2b_final( __private blake2b_state *S, __private uchar *out )
{
	uchar buffer[BLAKE2B_OUTBYTES];

	if( S->buflen > BLAKE2B_BLOCKBYTES )
	{
		blake2b_increment_counter( S, BLAKE2B_BLOCKBYTES );
		blake2b_compress( S, S->buf );
		S->buflen -= BLAKE2B_BLOCKBYTES;
		clmemcpy( S->buf, S->buf + BLAKE2B_BLOCKBYTES, S->buflen );
	}

	blake2b_increment_counter( S, S->buflen );
	blake2b_set_lastblock( S );
	clmemset( S->buf + S->buflen, 0, 2 * BLAKE2B_BLOCKBYTES - S->buflen ); // Padding
	blake2b_compress( S, S->buf );

	for( int i = 0; i < 8; ++i ) // Output full hash to temp buffer
		store64( buffer + sizeof( S->h[i] ) * i, S->h[i] );

	clmemcpy( out, buffer, 32 );
	return 0;
}

// inlen, at least, should be ulong. Others can be size_t.
int blake2b( __private uchar *out, __private uchar *in )
{
	private blake2b_state S[1];

	blake2b_init( S );

	blake2b_update( S, in, 80 );
	blake2b_final( S, out );
	return 0;
}

#endif
