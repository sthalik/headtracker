#pragma once
struct ht_context;
typedef struct ht_context headtracker_t;

typedef struct ht_config {
	float focal_length;
	float classification_delay;
	float feature_quality_level;
	int   pyrlk_pyramids;
	int   pyrlk_win_size_w;
	int   pyrlk_win_size_h;
	int   max_tracked_features;
	float ransac_min_consensus;
	int   ransac_iter;
	int   ransac_min_features;
	float ransac_max_consensus_error;
	bool  use_harris;
	float max_detect_features;
	int   min_track_start_features;
	int   max_init_retries;
	float features_detect_threshold;
	float filter_lumps_feature_count_threshold;
	int   feature_max_failed_ransac;
	int   ransac_posit_iter;
	float ransac_posit_eps;
	int   depth_avg_frames;
	float min_feature_distance;
	float detect_feature_distance;
	float filter_lumps_distance_threshold;
	float ransac_best_error_importance;
	float ransac_max_error;
} ht_config_t;

typedef struct {
	float rotx, roty, rotz;
	float tx, ty, tz;
	bool filled;
	float confidence;
	float feature_ratio;
} ht_result_t;

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

typedef struct {
	int width;
	int height;
	char const* data;
} ht_frame_t;

HT_API(headtracker_t*) ht_make_context(int camera_idx, const ht_config_t* config = NULL);
HT_API(ht_config_t) ht_load_config(FILE* stream);
HT_API(ht_config_t) ht_load_config(const char* filename);
HT_API(void) ht_free_context(headtracker_t* ctx);
HT_API(ht_frame_t) ht_get_bgr_frame(headtracker_t* ctx);
HT_API(ht_config_t) ht_make_config();
HT_API(bool) ht_cycle(headtracker_t* ctx, ht_result_t* euler);

HT_API(void) ht_store_config(const headtracker_t* ctx, FILE* stream);
HT_API(void) ht_store_config(const headtracker_t* ctx, const char* filename);
HT_API(void) ht_store_config(const ht_config_t& config, const char* filename);
HT_API(void) ht_store_config(const ht_config_t& config, FILE* stream);