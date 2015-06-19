#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <curl/curl.h>

#include "network.h"

// TODO: document what this does. It should also probably be renamed.
struct inData {
	uint8_t *bytes;
	size_t len;
};

// TODO: Is it safe to have these as global variables?
char *bfw_url, *submit_url;
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

// Write network data to an array of bytes
//
// TODO: nmemb is not a good name. 'in' is also probably not a good name.
size_t writefunc(void *ptr, size_t size, size_t nmemb, struct inData *in) {
	if (in == NULL)
		return size*nmemb;
	size_t new_len = size*nmemb;
	in->bytes = (uint8_t*)malloc(new_len);
	if (in->bytes == NULL) {
		fprintf(stderr, "malloc() failed\n");
		exit(EXIT_FAILURE);
	}
	memcpy(in->bytes, ptr, size*nmemb);
	in->len = new_len;

	return size*nmemb;
}

// init_network initializes curl networking.
void init_network() {
	curl =  curl_easy_init();
	if (!curl) {
		fprintf(stderr, "Error on curl_easy_init().\n");
	}
}

// get_header_for_work fetches a block header from siad. This block header is
// ready for nonce grinding.
int get_header_for_work(uint8_t *target, uint8_t *header) {
	if (!curl) {
		fprintf(stderr, "Invalid curl object passed to get_block_for_work()\n");
		exit(1);
	}

	CURLcode res;
	struct inData in;

	// Get data from siad
	curl_easy_reset(curl);
	curl_easy_setopt(curl, CURLOPT_URL, bfw_url);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writefunc);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &in);

	res = curl_easy_perform(curl);
	if(res != CURLE_OK) {
		fprintf(stderr, "Failed to get block for work, curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
		fprintf(stderr, "Are you sure that siad is running?\n");
		exit(1);
	}
	if (check_http_response(curl)) {
		return 1;
	}
	if (in.len != 112) {
		fprintf(stderr, "curl did not receive correct bytes (got %lu, expected 112)\n", in.len);
		return 1;
	}

	// Copy data to return
	memcpy(target, in.bytes,     32);
	memcpy(header, in.bytes+32,  80);

	free(in.bytes);

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
		exit(1);
	}
	return check_http_response(curl);
}

// free_network closes the network connection.
void free_network() {
	curl_easy_cleanup(curl);
}
