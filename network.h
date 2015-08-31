#include <stdint.h>

void set_host(char *host, char *port);
void init_network();
int get_header_for_work(uint8_t *target, uint8_t *header);
int submit_header(uint8_t *header);
void free_network();
