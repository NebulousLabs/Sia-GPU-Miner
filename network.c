#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <curl/curl.h>

#include "network.h"

// Buffer for receiving data through curl
struct inBuffer {
	uint8_t *bytes;
	size_t len; // Number of bytes read in/allocated
};

// URL strings for receiving and submitting blocks
char *bfw_url, *submit_url;

// CURL object to connect to siad
CURL *curl;

// check_http_response
int check_http_response(CURL *curl) {
	long http_code = 0;
	curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
	if (http_code != 200) {
		fprintf(stderr, "HTTP error %lu", http_code);
		return 1;
	}
	return 0;
}

// set_port establishes the port that siad is on.
void set_port(char *port) {
	bfw_url = malloc(29 + strlen(port));
	submit_url = malloc(28 + strlen(port));
	sprintf(bfw_url, "localhost:%s/miner/headerforwork", port);
	sprintf(submit_url, "localhost:%s/miner/submitheader", port);
}

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
	curl_easy_setopt(curl, CURLOPT_URL, bfw_url);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writefunc);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &inBuf);

	res = curl_easy_perform(curl);
	if(res != CURLE_OK) {
		fprintf(stderr, "Failed to get block for work, curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
		fprintf(stderr, "Are you sure that siad is running?\n");
		// Pause in order to prevent spamming the console
		printf("Would you like to retry connecting? (y/n)");
		do {
			char ans = getchar();
			if (ans == 'n' || ans == 'N') {
				exit(1);
			}
			if (ans == 'y' || ans == 'Y') {
				return 1;
			}
		} while (1);
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

	curl_easy_reset(curl);
	curl_easy_setopt(curl, CURLOPT_URL, submit_url);
	curl_easy_setopt(curl, CURLOPT_POST, 1);
	curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, 80);
	curl_easy_setopt(curl, CURLOPT_POSTFIELDS, header);
	// Prevent printing to stdout
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writefunc);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, NULL);

	res = curl_easy_perform(curl);
	if (res != CURLE_OK) {
		fprintf(stderr, "Failed to submit block, curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
		return 1;
	}
	return check_http_response(curl);
}

// free_network closes the network connection.
void free_network() {
	curl_easy_cleanup(curl);
}
