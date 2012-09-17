#pragma once
#include "api.h"
#include <opencv2/opencv.hpp>
struct ht_context;
typedef struct ht_context headtracker_t;

using namespace std;
using namespace cv;

typedef struct ht_config {
	float field_of_view;
	float classification_delay;
	int   feature_quality_level;
	int   pyrlk_pyramids;
	int   pyrlk_win_size_w;
	int   pyrlk_win_size_h;
	int   max_tracked_features;
	int   max_init_retries;
	float features_detect_threshold;
	float filter_lumps_feature_count_threshold;
	int   feature_max_failed_ransac;
	double ransac_posit_eps;
	int   depth_avg_frames;
	float min_feature_distance;
	float filter_lumps_distance_threshold;
	float ransac_max_error;
	int   max_keypoints;
	int   keypoint_quality;
	float keypoint_distance;
	float feature_detect_ratio;
	int   force_width;
	int   force_height;
	int   force_fps;
	int   camera_index;
	int   keypoint_max_failed_ransac;
	bool  debug;
	float ransac_smaller_error_preference;
    int   ransac_posit_iter;
    int   ransac_num_iters;
    int   ransac_min_features;
    double pyrlk_min_eigenval;
} ht_config_t;

typedef struct {
	float rotx, roty, rotz;
	float tx, ty, tz;
	bool filled;
	float confidence;
} ht_result_t;

typedef enum {
	cfg_type_float = 0,
	cfg_type_int   = 1,
	cfg_type_bool  = 2,
    cfg_type_double = 3
} ht_cfg_type_t;

typedef union
{
	double d;
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
    Mat data;
} ht_frame_t;

HT_API(headtracker_t*) ht_make_context(const ht_config_t* config, const char* filename);
HT_API(ht_config_t) ht_load_config(FILE* stream);
HT_API(ht_config_t) ht_load_config_from_file(const char* filename);
HT_API(void) ht_free_context(headtracker_t* ctx);
HT_API(ht_frame_t) ht_get_bgr_frame(headtracker_t* ctx);
HT_API(ht_config_t) ht_make_config();
HT_API(bool) ht_cycle(headtracker_t* ctx, ht_result_t* euler);

HT_API(void) ht_store_config(const headtracker_t* ctx, FILE* stream);
HT_API(void) ht_store_config_in_file(const headtracker_t* ctx, const char* filename);
