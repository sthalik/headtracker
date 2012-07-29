#include "stdafx.h"

using namespace std;
using namespace cv;

#define DEFINE_UNION_INITIALIZER(t, m) \
	static __inline ht_cfg_value_t ht_cfg_##t##_to_union(t datum) { \
		ht_cfg_value_t ret; \
		ret.m = datum; \
		return ret; \
	}

DEFINE_UNION_INITIALIZER(float, f);
DEFINE_UNION_INITIALIZER(int, i);
DEFINE_UNION_INITIALIZER(bool, i);

#define F(v, t, d, min_, max_, ...) { #v, offsetof(ht_config_t, v), cfg_type_##t, ht_cfg_##t##_to_union(d), ht_cfg_##t##_to_union(min_), ht_cfg_##t##_to_union(max_), __VA_ARGS__ }
#define FIELD_END { NULL, -1, cfg_type_int, 0 }

static const ht_reflection_t ht_reflection_info[] = {
	F(classification_delay, float, 334.0f, 200.0f, 2000.0f,
		"Delay between two Haar classifications at the very start of tracking, until enough features are tracked."),
	F(features_detect_threshold, float, 0.98f, 0.5f, 1.0f,
		"Only detect new features when the amount of features presently tracked falls below this ratio."),
	F(feature_max_failed_ransac, int, 2, 0, 6,
		"Maximum consecutive amount of RANSAC evaluations of a feature that fail for it to no longer be tracked."),
	F(feature_quality_level, int, 30, HT_FEATURE_MIN_QUALITY_LEVEL, HT_FEATURE_MAX_QUALITY_LEVEL,
		"Maximum feature quality level compared to the strongest feature during their detection. See cvGoodFeaturesToTrack."),
	F(filter_lumps_feature_count_threshold, float, 0.51f, 0.5f, 1.0f,
		"Only filter features too close to each other if amount of features falls above this threshold."),
	F(focal_length, float, 675.0f, 300.0f, 2000.0f,
		"Camera field of view in OpenCV-implementation-dependent values."),
	F(max_init_retries, int, 100, 2, 1000,
		"Maximum retries before restarting the initialization process."),
	F(max_tracked_features, int, 60, 40, 200,
		"Maximum features to be tracked at once."),
	F(min_track_start_features, int, 20, 10, 100,
		"Minimum features to track before initialization ends and main phase starts."),
	F(pyrlk_pyramids, int, 4, 3, 50,
		"Maximum mipmaps for optical flow tracking. See cvCalcOpticalFlowPyrLK"),
	F(pyrlk_win_size_h, int, 15, 3, 45,
		"Window size for optical flow tracking, horizontal."),
	F(pyrlk_win_size_w, int, 15, 4, 60,
		"Window size for optical flow tracking, vertical"),
	F(ransac_iter, int, 96, 48, 288,
		"RANSAC iterations per frame."),
	F(ransac_max_consensus_error, float, 10.0f, 5.0f, 30.0f,
		"Maximum total RANSAC consensus error (see ransac_max_error)."),
	F(ransac_min_consensus, float, 0.4f, 0.2f, 0.6f,
		"Minimum RANSAC consensus size, with regards to the amount of presently tracked features"),
	F(ransac_posit_eps, float, 2.0f, 0.0001f, 10.0f,
		"cvPOSIT epsilon for the purpose of estimating a feature set in RANSAC."),
	F(ransac_posit_iter, int, 50, 10, 500,
		"cvPOSIT max iteration count for estimating a RANSAC feature set."),
	F(depth_avg_frames, int, 30, 1, 120,
		"Amount of frames for arithmetic averaging of depth info. "
		"Depth info is used for turning pixel-based indicators into absolute measures, independent of closeness to the camera."),
	F(ransac_best_error_importance, float, 0.1f, 0.0f, 10.0f,
		"How much smaller reprojection error is favored against more features in a RANSAC iteration that's about to be turned into a consensus."),
	F(ransac_max_error, float, 0.975f, 0.9f, 1.001f,
		"Maximum error of one RANSAC iteration compared to the previous one."),
	F(filter_lumps_distance_threshold, float, 0.7f, 0.5f, 1.0f,
		"How much too close to each other features have to be to filter them. Features that failed the last POSIT iteration are removed first."),
	F(min_feature_distance, float, 4.9f, 3.00001f, 15.00001f,
		"Distance between two features at the time of their detection, including already detected ones."),
	F(max_keypoints, int, 18, 8, 48,
		"Maximum keypoints to track"),
	F(keypoint_quality, int, 40, 2, 60,
		"Starting keypoint quality"),
	F(keypoint_distance, float, 10.20001f, 10.0f, 50.0f,
		"Minimum Euclidean distance between keypoints"),
	F(ransac_min_features, int, 6, 4, 24,
		"Points to start each RANSAC iteration"),
	F(feature_detect_ratio, float, 0.7f, 0.1f, 2.0f,
		"Features to detect at one time to consider given quality level as satisfactory."),
	F(force_width, int, 0, 0, 10000,
		"Force capture width of a webcam."),
	F(force_height, int, 0, 0, 10000,
		"Force capture height of a webcam."),
	F(force_fps, int, 0, 0, 10000,
		"Force capture frames per second of a webcam."),
	F(camera_index, int, 0, -1, 100000,
		"Choose a different camera by its platform-specific index."),
	FIELD_END
};

ht_reflection_t ht_find_config_entry(const char* name) {
	for (int i = 0; ht_reflection_info[i].name; i++)
		if (!strcmp(name, ht_reflection_info[i].name))
			return ht_reflection_info[i];

	throw exception();
}

HT_API(ht_config_t) ht_make_config() {
	ht_config_t ret;

	for (int i = 0; ht_reflection_info[i].name; i++) {
		const ht_reflection_t& field = ht_reflection_info[i];
		void* ptr = ((char*) &ret) + field.offset;
		switch (field.type) {
		case cfg_type_bool:
			*(bool*) ptr = field.default_value.i ? 1 : 0;
			break;
		case cfg_type_float:
			*(float*) ptr = field.default_value.f;
			break;
		case cfg_type_int:
			*(int*) ptr = field.default_value.i;
			break;
		default:
			throw exception();
		}
	}
	return ret;
}

HT_API(ht_config_t) ht_load_config(FILE* stream) {
	char buf[256];
	ht_config_t cfg = ht_make_config();

	buf[255] = '\0';

	while (fgets(buf, 255, stream)) {
			int len = strlen(buf);
		char* value = strchr(buf, ' ');
		if (value == NULL)
			throw new exception();
		*value++ = '\0';
		if (strlen(value) == 0)
			throw exception();
		ht_reflection_t info = ht_find_config_entry(buf);
		void* ptr = ((char*) &cfg) + info.offset;
		switch (info.type) {
		case cfg_type_bool:
			*(bool*) ptr = strtol(value, NULL, 10) ? 1 : 0;
			break;
		case cfg_type_float:
			*(float*) ptr = (float) strtod(value, NULL);
			if (*(float*) ptr > info.max.f || *(float*) ptr < info.min.f)
				throw exception();
			break;
		case cfg_type_int:
			*(int*) ptr = (int) strtol(value, NULL, 10);
			if (*(int*) ptr > info.max.i || *(int*) ptr < info.min.i)
				throw exception();
			break;
		default:
			throw exception();
		}
	}

	return cfg;
}

HT_API(ht_config_t) ht_load_config(const char* filename) {
	FILE* stream = fopen(filename, "r");

	if (stream == NULL)
		throw exception();

	ht_config_t cfg;

	try {
		cfg = ht_load_config(stream);
	} catch (exception e) {
		fclose(stream);
		throw e;
	}

	fclose(stream);

	return cfg;
}

HT_API(void) ht_store_config(const ht_config_t& config, FILE* stream) {
	for (int i = 0; ht_reflection_info[i].name; i++) {
		const ht_reflection_t& info = ht_reflection_info[i];
		void* ptr = ((char*) &config) + info.offset;
		fprintf(stream, "%s ", info.name);
		switch (info.type) {
		case cfg_type_bool:
			fprintf(stream, "%d", (int) *(bool*)ptr);
			break;
		case cfg_type_int:
			fprintf(stream, "%d", *(int*)ptr);
			break;
		case cfg_type_float:
			fprintf(stream, "%f", *(float*)ptr);
			break;
		default:
			throw exception();
		}
		fprintf(stream, "\n");
	}
}

HT_API(void) ht_store_config(const ht_config_t& config, const char* filename) {
	FILE* stream = fopen(filename, "w");

	if (stream == NULL)
		throw exception();

	try {
		ht_store_config(config, stream);
	} catch (exception e) {
		fclose(stream);
		throw e;
	}
	fclose(stream);
}

HT_API(void) ht_store_config(const headtracker_t* ctx, const char* filename) {
	ht_store_config(ctx->config, filename);
}

HT_API(void) ht_store_config(const headtracker_t* ctx, FILE* stream) {
	ht_store_config(ctx->config, stream);
}