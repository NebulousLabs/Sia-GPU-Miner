#include <stdint.h>
#include <curl/curl.h>

int get_block_for_work(CURL *curl, uint8_t *target, uint8_t *header, uint8_t **block, size_t *blocklen);
void submit_block(CURL *curl, uint8_t *block, size_t blocklen);
