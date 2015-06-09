#include <cstdint>
#include <curl/curl.h>

void set_port(char *port);
int get_header_for_work(CURL *curl, uint8_t *target, uint8_t *header);
void submit_header(CURL *curl, uint8_t *header);
