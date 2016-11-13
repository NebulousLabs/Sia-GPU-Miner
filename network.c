#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#else
#ifdef __WINDOWS__
#include <windows.h>
#define sleep(seconds) Sleep(seconds*1000)
#endif
#endif

#include <curl/curl.h>

#include "network.h"

// Buffer for receiving data through curl
struct inBuffer {
	uint8_t *bytes;
	size_t len; // Number of bytes read in/allocated
};

// URL strings for receiving and submitting blocks
#define MAX_HOST_LEN (2048)
char bfw_url[MAX_HOST_LEN];
char submit_url[MAX_HOST_LEN];

// CURL object to connect to siad
CURL *curl;

// check_http_response
int check_http_response(CURL *curl) {
	long http_code = 0;
	curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
	if (http_code == 400) {
		fprintf(stderr, "HTTP error %lu - check that the wallet is unlocked\n", http_code);
		return 1;
	} else if (http_code < 200 || http_code > 299) {
		fprintf(stderr, "HTTP error %lu\n", http_code);
		return 1;
	}
	return 0;
}

// set_host establishes the hostname and port that siad is on.
void set_host(char *host, char *port) {
	size_t host_len = 21 + strlen(host) + strlen(port);
	if (host_len >= MAX_HOST_LEN) {
		fprintf(stderr, "Error: host is over of size, host_len=%zu > MAX_HOST_LEN=%d\n", host_len, MAX_HOST_LEN);
		exit(1);
	}
	
	sprintf(bfw_url, "%s%s/miner/header", host, port);
	sprintf(submit_url, "%s%s/miner/header", host, port);
}

static void printMem(const uint8_t *mem, size_t size, const char *format, int num_in_line) {
	if (format == NULL) {
		format = "%02x ";
	}
	
	if (num_in_line == 0) {
		num_in_line = 16;
	}
	
	for (int i=0; i<size; i++) {
		printf(format, mem[i]);
		if ((i+1)%num_in_line == 0) {
			printf("\n");
		}
	}
	printf("\n");
}

//static void printMemHex(const uint8_t *mem, size_t size){
//    printMem(mem, size, "%02x ", 16);
//}
//
//static void printMemChar(const uint8_t *mem, size_t size){
//    printMem(mem, size, "%c", 16);
//}
//
//static void printMemDec(const uint8_t *mem, size_t size){
//    printMem(mem, size, "%d ", 16);
//}

// Write network data to a buffer (inBuf)
size_t writefunc(void *ptr, size_t size, size_t num_elems, struct inBuffer *inBuf) {
	if (inBuf == NULL) {
		return size*num_elems;
	}
	size_t new_len = size*num_elems;
	inBuf->bytes = (uint8_t*)malloc(new_len);
	if (inBuf->bytes == NULL) {
		fprintf(stderr, "malloc() failed\n");
		exit(EXIT_FAILURE);
	}
	memcpy(inBuf->bytes, ptr, size*num_elems);
	inBuf->len = new_len;

	return size*num_elems;
}

// init_network initializes curl networking.
void init_network() {
	curl =  curl_easy_init();
	if (!curl) {
		fprintf(stderr, "Error on curl_easy_init().\n");
		exit(1);
	}
}

// get_header_for_work fetches a block header from siad. This block header is
// ready for nonce grinding.
int get_header_for_work(uint8_t *target, uint8_t *header) {

	CURLcode res;
	struct inBuffer inBuf;

	// Get data from siad
	curl_easy_reset(curl);
	curl_easy_setopt(curl, CURLOPT_USERAGENT, "Sia-Agent");
	curl_easy_setopt(curl, CURLOPT_URL, bfw_url);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writefunc);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &inBuf);

	res = curl_easy_perform(curl);
	if(res != CURLE_OK) {
		fprintf(stderr, "Failed to get header from %s, curl_easy_perform() failed: %s\n", bfw_url, curl_easy_strerror(res));
		fprintf(stderr, "Are you sure that siad is running?\n");
		// Pause in order to prevent spamming the console
		sleep(3); // 3 seconds
	}

	if (check_http_response(curl)) {
		return 1;
	}
	if (inBuf.len != 112) {
		fprintf(stderr, "curl did not receive correct bytes (got %lu, expected 112)\n", inBuf.len);
		return 1;
	}

	// Copy data to return
	memcpy(target, inBuf.bytes,     32);
	memcpy(header, inBuf.bytes+32,  80);

	free(inBuf.bytes);

	return 0;
}

// submit_header submits a block header to siad.
int submit_header(uint8_t *header) {
	CURLcode res;
	struct inBuffer inBuf;

	curl_easy_reset(curl);
	curl_easy_setopt(curl, CURLOPT_USERAGENT, "Sia-Agent");
	curl_easy_setopt(curl, CURLOPT_URL, submit_url);
	curl_easy_setopt(curl, CURLOPT_POST, 1);
	curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, 80);
	curl_easy_setopt(curl, CURLOPT_POSTFIELDS, header);
	// Prevent printing to stdout
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writefunc);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &inBuf);

	res = curl_easy_perform(curl);
	if (res != CURLE_OK) {
		fprintf(stderr, "Failed to submit block, curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
		return 1;
	}
	
	if (inBuf.bytes) {
		printMem(inBuf.bytes, inBuf.len, "%c", INT_MAX);
		
		free(inBuf.bytes);
	}
	
	return check_http_response(curl);
}

// free_network closes the network connection.
void free_network() {
	curl_easy_cleanup(curl);
}
