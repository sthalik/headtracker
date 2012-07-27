#include "config.h"

typedef enum {
	cfg_type_float = 0,
	cfg_type_int   = 1,
	cfg_type_bool  = 2
} ht_cfg_type_t;

typedef union {
	float f;
	int i;
} ht_cfg_value_t;

typedef struct {
	const char* name;
	int offset;
	ht_cfg_type_t type;
	ht_cfg_value_t default_value;
	ht_cfg_value_t min;
	ht_cfg_value_t max;
	const char* docstring;
} ht_reflection_t;

ht_config_t ht_load_config(FILE* stream);
ht_config_t ht_load_config(const char* filename);
headtracker_t* ht_make_context(int camera_idx, const ht_config_t* config = NULL);
void ht_free_context(headtracker_t* ctx);