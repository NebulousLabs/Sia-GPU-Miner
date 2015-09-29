#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
using namespace std;
#include "network.h"

extern double target_to_diff(const uint32_t *const target);

static char bfw_url[255], submit_url[255];
static CURL *curl;
static char curlerrorbuffer[CURL_ERROR_SIZE];
struct inData in;

// Write network data to an array of bytes
size_t writefunc(void *ptr, size_t size, size_t nmemb, struct inData *in)
{
	size_t new_len = size*nmemb;
	if(in == NULL || new_len == 0)
		return 0;

	in->bytes = (uint8_t*)realloc(in->bytes, in->len + new_len);
	if(in->bytes == NULL)
	{
		fprintf(stderr, "malloc() failed\n");
		exit(EXIT_FAILURE);
	}
	memcpy(in->bytes + in->len, ptr, new_len);
	in->len += new_len;

	return new_len;
}

void network_init(const char *domain, const char *port, const char *useragent)
{
	CURLcode res;

	curl = curl_easy_init();
	if(curl == NULL)
	{
		printf("\nError: can't init curl\n");
		exit(EXIT_FAILURE);
	}
	/* when we want to debug stuff
	res = curl_easy_setopt(curl, CURLOPT_VERBOSE, 1);
	if(res != CURLE_OK)
	{
		printf("verbose res=%d\n", res);
		fprintf(stderr, "%s\n", curlerrorbuffer);
		curl_easy_cleanup(curl);
		exit(1);
	}
	*/
	res = curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, curlerrorbuffer);
	if(res != CURLE_OK)
	{
		fprintf(stderr, "%s\n", curl_easy_strerror(res));
		curl_easy_cleanup(curl);
		exit(1);
	}

	if(bfw_url == NULL || submit_url == NULL)
	{
		printf("\nmalloc error\n");
		exit(EXIT_FAILURE);
	}
	sprintf(bfw_url, "http://%s:%s/miner/headerforwork", domain, port);
	sprintf(submit_url, "http://%s:%s/miner/submitheader", domain, port);

	res = curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writefunc);
	if(res != CURLE_OK)
	{
		fprintf(stderr, "%s\n", curlerrorbuffer);
		curl_easy_cleanup(curl);
		exit(1);
	}

	res = curl_easy_setopt(curl, CURLOPT_WRITEDATA, &in);
	if(res != CURLE_OK)
	{
		fprintf(stderr, "%s\n", curlerrorbuffer);
		curl_easy_cleanup(curl);
		exit(1);
	}

	res = curl_easy_setopt(curl, CURLOPT_USERAGENT, useragent);
	if(res != CURLE_OK)
	{
		fprintf(stderr, "%s\n", curlerrorbuffer);
		curl_easy_cleanup(curl);
		exit(1);
	}
}

void network_cleanup(void)
{
	curl_easy_cleanup(curl);
}

int check_http_response(CURL *curl)
{
	long http_code = 0;
	CURLcode err = curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
	if(err == CURLE_OK)
	{
		if(http_code != 200 && http_code != 0)
		{
			fprintf(stderr, "\nHTTP error %lu", http_code);
			if(in.len > 0)
			{
				in.bytes = (uint8_t*)realloc(in.bytes, in.len + 1);
				*(in.bytes + in.len) = 0;
				printf(": %s", in.bytes);
			}
			printf("\n");
			return 1;
		}
	}
	return 0;
}

int get_header_for_work(uint8_t *target, uint8_t *header)
{
	static double diff = 0.0;
	double tmp;

	CURLcode res;
	in.bytes = NULL;
	in.len = 0;

	// Get data from siad
	curl_easy_setopt(curl, CURLOPT_POST, 0);
	res = curl_easy_setopt(curl, CURLOPT_URL, bfw_url);
	if(res != CURLE_OK)
	{
		fprintf(stderr, "%s\n", curlerrorbuffer);
		curl_easy_cleanup(curl);
		exit(1);
	}

	res = curl_easy_perform(curl);
	if(res != CURLE_OK)
	{
		fprintf(stderr, "%s\n", curlerrorbuffer);
		curl_easy_cleanup(curl);
		exit(1);
	}
	if(check_http_response(curl))
	{
		return 1;
	}
	if(in.len != 112)
	{
		fprintf(stderr, "\ncurl did not receive correct bytes (got %Iu, expected 112)\n", in.len);
		free(in.bytes);
		return 1;
	}

	// Copy data to return
	memcpy(target, in.bytes, 32);
	memcpy(header, in.bytes + 32, 80);
	tmp = target_to_diff((uint32_t*)target);
	if(tmp != diff)
	{
		printf("\nnew difficulty = %lu\n", lround(tmp));
		diff = tmp;
	}

	free(in.bytes);
	return 0;
}

bool submit_header(uint8_t *header)
{
	CURLcode res;
	in.len = 0;
	in.bytes = NULL;
		
	res = curl_easy_setopt(curl, CURLOPT_URL, submit_url);
	if(res != CURLE_OK)
	{
		fprintf(stderr, "%s\n", curlerrorbuffer);
		curl_easy_cleanup(curl);
		exit(1);
	}
	curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_0);
	curl_easy_setopt(curl, CURLOPT_POST, 1);
	curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, 80);
	res = curl_easy_setopt(curl, CURLOPT_POSTFIELDS, header);
	if(res != CURLE_OK)
	{
		fprintf(stderr, "%s\n", curlerrorbuffer);
		curl_easy_cleanup(curl);
		exit(1);
	}

	res = curl_easy_perform(curl);
	if(res != CURLE_OK)
	{
		fprintf(stderr, "%s\n", curlerrorbuffer);
		fprintf(stderr, "failed to submit header\n");
		curl_easy_cleanup(curl);
		exit(1);
	}
	if(check_http_response(curl) != 0)
	{
		free(in.bytes);
		return false;
	}
	else
	{
		free(in.bytes);
		return true;
	}
}

