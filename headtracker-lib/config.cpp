#include "api.h"
#include "stdafx.h"

using namespace std;
using namespace cv;

#define DEFINE_UNION_INITIALIZER(t, m) \
    static __inline ht_cfg_value_t ht_cfg_##t##_to_union(t datum) { \
    ht_cfg_value_t ret; \
    ret.m = datum; \
    return ret; \
    }

DEFINE_UNION_INITIALIZER(float, f)
DEFINE_UNION_INITIALIZER(double, d)
DEFINE_UNION_INITIALIZER(int, i)
DEFINE_UNION_INITIALIZER(bool, i)

#define F(v, t, d, min_, max_, ...) { #v, offsetof(ht_config_t, v), cfg_type_##t, ht_cfg_##t##_to_union(d), ht_cfg_##t##_to_union(min_), ht_cfg_##t##_to_union(max_), __VA_ARGS__ }
#define FIELD_END { NULL, -1, cfg_type_int, ht_cfg_int_to_union(0), ht_cfg_int_to_union(0), ht_cfg_int_to_union(0), NULL }

static const ht_reflection_t ht_reflection_info[] = {
    F(classification_delay, float, 300.0f, 200.0f, 2000.0f,
    "Delay between two Haar classifications at the very start of tracking, until enough features are tracked."),
    F(keypoint_max_failed_ransac, int, 0, 0, 6,
    "Maximum consecutive amount of RANSAC evaluations of a keypoint that fail for it to no longer be tracked."),
    F(field_of_view, float, 69.0f, 40.0f, 180.0f,
    "Camera field of view in degrees."),
    F(max_init_retries, int, 150, 2, 1000,
    "Maximum retries before restarting the initialization process."),
    F(pyrlk_pyramids, int, 4, 0, 10,
    "Maximum mipmaps for optical flow tracking. See cvCalcOpticalFlowPyrLK"),
    F(pyrlk_win_size_w, int, 35, 4, 60,
    "Window size for optical flow tracking, vertical"),
    F(pyrlk_win_size_h, int, 35, 3, 45,
    "Window size for optical flow tracking, horizontal."),
    F(ransac_posit_eps, float, 1e-4, 1.0e-10, 1.0e5,
    "cvPOSIT epsilon for the purpose of estimating a feature set in RANSAC."),
    F(ransac_posit_iter, int, 5, 10, 200,
    "cvPOSIT iteration count for the purpose of estimating a feature set in RANSAC."),
    F(max_keypoints, int, 200, 30, 150,
    "Maximum keypoints to track"),
    F(keypoint_quality, int, 20, 2, 60,
    "Starting keypoint quality"),
    F(keypoint_distance, float, 2.6f, 1.5f, 50.0f,
    "Minimum Euclidean distance between keypoints"),
    F(keypoint_3distance, float, 12.0f, 1.5f, 50.0f,
    "Minimum Euclidean distance between keypoints"),
    F(keypoint_10distance, float, 18.0f, 1.5f, 50.0f,
    "Minimum Euclidean distance between keypoints"),
    F(force_width, int, 640, 0, 10000,
    "Force capture width of a webcam."),
    F(force_height, int, 480, 0, 10000,
    "Force capture height of a webcam."),
    F(force_fps, int, 0, 0, 10000,
    "Force capture frames per second of a webcam."),
    F(camera_index, int, -1, -1, 100000,
    "Choose a different camera by its platform-specific index."),
    F(ransac_num_iters, int, 12, 5, 100,
    "RANSAC iterations per frame"),
    F(ransac_max_error, float, 10, 2.0, 50.0,
    ""),
    F(ransac_avg_error, float, 0.9, 0.0, 1.0,
    ""),
    F(pyrlk_min_eigenval, float, 1.0e-4, 1.0e-10, 1.0e-1,
    "Min eigenval for Lukas-Kanade"),
    F(max_best_error, float, 8, 4, 1000,
    "Max RANSAC error"),
    F(debug, bool, true, 0, 1),
    F(ransac_max_threads, int, 8, 1, 256,
    "Max threads for RANSAC"),
    F(ransac_min_features, float, 0.6, 0.1, 1.0,
    "Min features for RANSAC"),
    F(feature_good_nframes, int, 15, 1, 600),
    FIELD_END
};

ht_reflection_t ht_find_config_entry(const char* name) {
    for (int i = 0; ht_reflection_info[i].name; i++)
        if (!strcmp(name, ht_reflection_info[i].name))
            return ht_reflection_info[i];

    throw exception();
}

HT_API(void) ht_make_config(ht_config_t* ret) {
    for (int i = 0; ht_reflection_info[i].name; i++) {
        const ht_reflection_t& field = ht_reflection_info[i];
        void* ptr = ((char*) ret) + field.offset;
        switch (field.type) {
        case cfg_type_bool:
            *(bool*) ptr = field.default_value.i ? 1 : 0;
            printf("%s = %d\n", field.name, field.default_value.i);
            break;
        case cfg_type_float:
            *(float*) ptr = field.default_value.f;
            printf("%s = %f\n", field.name, field.default_value.f);
            break;
        case cfg_type_double:
            *(double*) ptr = field.default_value.d;
            printf("%s = %f\n", field.name, field.default_value.d);
            break;
        case cfg_type_int:
            *(int*) ptr = field.default_value.i;
            printf("%s = %d\n", field.name, field.default_value.i);
            break;
        default:
            fprintf(stderr, "bad config type for field %s\n", field.name);
            continue;
        }
    }
}

HT_API(void) ht_load_config(FILE* stream, ht_config_t* cfg) {
    char buf[256];
    ht_make_config(cfg);

    buf[255] = '\0';

    while (fgets(buf, 255, stream)) {
        char* value = strchr(buf, ' ');
        if (value == NULL)
            continue;
        *value++ = '\0';
        if (strlen(value) == 0)
            continue;
        ht_reflection_t info = ht_find_config_entry(buf);
        void* ptr = ((char*) cfg) + info.offset;
        switch (info.type) {
        case cfg_type_bool:
            *(bool*) ptr = strtol(value, NULL, 10) ? 1 : 0;
            break;
        case cfg_type_float:
            *(float*) ptr = (float) strtod(value, NULL);
            if (*(float*) ptr > info.max.f || *(float*) ptr < info.min.f)
                continue;
            break;
        case cfg_type_double:
            *(double*) ptr = strtod(value, NULL);
            if (*(double*) ptr > info.max.d || *(double*) ptr < info.min.d)
                continue;
            break;
        case cfg_type_int:
            *(int*) ptr = (int) strtol(value, NULL, 10);
            if (*(int*) ptr > info.max.i || *(int*) ptr < info.min.i)
                continue;
            break;
        default:
            continue;
        }
    }
}

static void ht_store_config_internal(const ht_config_t& config, FILE* stream) {
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
        case cfg_type_double:
            fprintf(stream, "%f", *(double*)ptr);
        default:
            throw exception();
        }
        fprintf(stream, "\n");
    }
}

static void ht_store_config_internal(const ht_config_t& config, const char* filename) {
    FILE* stream = fopen(filename, "w");

    if (stream == NULL)
        throw exception();

    try {
        ht_store_config_internal(config, stream);
    } catch (exception e) {
        fclose(stream);
        throw e;
    }
    fclose(stream);
}

HT_API(void) ht_store_config_in_file(const headtracker_t* ctx, const char* filename) {
    ht_store_config_internal(ctx->config, filename);
}

HT_API(void) ht_store_config(const headtracker_t* ctx, FILE* stream) {
    ht_store_config_internal(ctx->config, stream);
}
