#include <cstdlib>
#include <cstdio>
#include <cstring>
using namespace std;
#include "network.h"

struct inData {
	uint8_t *bytes;
	size_t len;
};

char *bfw_url, *submit_url;

void set_port(char *port) {
	bfw_url = malloc(29 + strlen(port));
	submit_url = malloc(28 + strlen(port));
	sprintf(bfw_url, "localhost:%s/miner/blockforwork", port);
	sprintf(submit_url, "localhost:%s/miner/submitblock", port);
}

// Write network data to an array of bytes
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

int get_block_for_work(CURL *curl, uint8_t *target, uint8_t *header, uint8_t **block, size_t *blocklen) {
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
	if (in.len < 174) {
		fprintf(stderr, "curl did not receive enough bytes (got %zu, expected at least 174)\n", in.len);
		return 1;
	}

	// Copy data to return
	*blocklen = in.len - 112;
	*block = (uint8_t*)malloc(*blocklen);
	memcpy(target, in.bytes,     32);
	memcpy(header, in.bytes+32,  80);
	memcpy(*block, in.bytes+112, in.len-112);

	return 0;
}

void submit_block(CURL *curl, uint8_t *block, size_t blocklen) {
	if (curl) {
		CURLcode res;
		curl_off_t numBytes = blocklen;

		curl_easy_reset(curl);
		curl_easy_setopt(curl, CURLOPT_URL, submit_url);
		curl_easy_setopt(curl, CURLOPT_POST, 1);
		curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, numBytes);
		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, block);
		// Prevent printing to stdout
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writefunc);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, NULL);

		res = curl_easy_perform(curl);
		if (res != CURLE_OK) {
			fprintf(stderr, "Failed to submit block, curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
			exit(1);
		}
	} else {
		printf("Invalid curl object passed to submit_block()\n");
		exit(1);
	}
}
