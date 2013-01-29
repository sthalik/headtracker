#include "ht-api.h"
#include "ht-internal.h"
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
    F(classification_delay, float, 4000, 200, 10000,
    "Delay between two Haar classifications at the very start of tracking, until enough features are tracked."),
    F(field_of_view, float, 69, 40, 180,
    "Camera field of view in degrees."),
    F(pyrlk_pyramids, int, 3, 0, 10,
    "Maximum mipmaps for optical flow tracking. See cvCalcOpticalFlowPyrLK"),
    F(pyrlk_win_size_w, int, 21, 4, 60,
    "Window size for optical flow tracking, horizontal"),
    F(pyrlk_win_size_h, int, 21, 3, 45,
    "Window size for optical flow tracking, vertical."),
    F(max_keypoints, int, 250, 30, 400,
    "Maximum keypoints to track"),
    F(keypoint_quality, int, 10, 10, 60,
    "Starting keypoint quality"),
    F(keypoint_distance, float, 2.4, 1.5, 50,
    "Minimum Euclidean distance between keypoints"),
    F(keypoint_3distance, float, 6, 1.5, 50,
    "Minimum Euclidean distance between keypoints"),
    F(force_width, int, 640, 0, 10000,
    "Force capture width of a webcam."),
    F(force_height, int, 480, 0, 10000,
    "Force capture height of a webcam."),
    F(force_fps, int, 0, 0, 10000,
    "Force capture frames per second of a webcam."),
    F(camera_index, int, -1, -1, 100000,
    "Choose a different camera by its platform-specific index."),
    F(ransac_num_iters, int, 100, 50, 300,
    "RANSAC iterations per frame"),
    F(ransac_max_reprojection_error, float, 4, 2, 50,
    ""),
    F(ransac_max_inlier_error, float, 4, 2, 50,
    ""),
    F(ransac_max_mean_error, float, 2.9, 2, 50,
    ""),
    F(ransac_abs_max_mean_error, float, 4, 2, 50,
    ""),
    F(debug, bool, 0, 0, 1,
    ""),
    F(ransac_min_features, float, 0.75, 0.1, 1,
    "Min features for RANSAC"),
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
            break;
        case cfg_type_float:
            *(float*) ptr = field.default_value.f;
            break;
        case cfg_type_double:
            *(double*) ptr = field.default_value.d;
            break;
        case cfg_type_int:
            *(int*) ptr = field.default_value.i;
            break;
        default:
            //fprintf(stderr, "bad config type for field %s\n", field.name);
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
