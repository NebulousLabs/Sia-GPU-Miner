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

int check_http_response(CURL *curl)
{
	long http_code = 0;
	CURLcode err = curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
	if(err == CURLE_OK)
	{
		if(http_code != 200)
		{
			fprintf(stderr, "HTTP error %lu\n", http_code);
			return 1;
		}
	}
	return 0;
}

void set_port(char *port) {
	bfw_url = (char*)malloc(29 + strlen(port));
	submit_url = (char*)malloc(28 + strlen(port));
	if(bfw_url == NULL || submit_url == NULL)
	{
		printf("malloc error\n");
		exit(EXIT_FAILURE);
	}
	sprintf(bfw_url, "localhost:%s/miner/headerforwork", port);
	sprintf(submit_url, "localhost:%s/miner/submitheader", port);
}

// Write network data to an array of bytes
size_t writefunc(void *ptr, size_t size, size_t nmemb, struct inData *in) {
	size_t new_len = size*nmemb;
	if(in == NULL || new_len == 0)
		return new_len;
	in->bytes = (uint8_t*)malloc(new_len);
	if (in->bytes == NULL) {
		fprintf(stderr, "malloc() failed\n");
		exit(EXIT_FAILURE);
	}
	memcpy(in->bytes, ptr, size*nmemb);
	in->len = new_len;

	return size*nmemb;
}

int get_header_for_work(CURL *curl, uint8_t *target, uint8_t *header) {
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
	if(check_http_response(curl))
	{
		return 1;
	}
	if(in.len != 112)
	{
		fprintf(stderr, "\ncurl did not receive correct bytes (got %d, expected 112)\n", in.len);
		return 1;
	}

	// Copy data to return
	memcpy(target, in.bytes,     32);
	memcpy(header, in.bytes+32,  80);
	free(in.bytes);
	return 0;
}

void submit_header(CURL *curl, uint8_t *header) {
	if (curl) {
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
		check_http_response(curl);
	}
	else
	{
		printf("Invalid curl object passed to submit_block()\n");
		exit(1);
	}
}
