#include <cstdint>
#include <curl/curl.h>

#ifndef WIN32
#include <unistd.h>
#define Sleep(duration) usleep((duration)*1000)
#endif

struct inData
{
	uint8_t *bytes;
	size_t len;
};

void set_port(char *port);
int get_header_for_work(uint8_t *target, uint8_t *header);
bool submit_header(uint8_t *header);
void network_init(const char *domain, const char *port, const char *useragent);
void network_cleanup(void);
