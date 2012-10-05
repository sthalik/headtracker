#include "stdafx.h"

using namespace std;
using namespace cv;

void ht_draw_features(headtracker_t& ctx) {
    for (int i = 0; i < ctx.config.max_keypoints; i++) {
        if (ctx.keypoints[i].idx != -1) {
            Scalar color;
            if (ctx.keypoints[i].frames < ctx.config.feature_good_nframes)
                color = Scalar(0, 255, 255);
            else
                color = Scalar(255, 255, 0);
            circle(ctx.color, cvPoint(ctx.keypoints[i].position.x, ctx.keypoints[i].position.y), 1, color, -1);
        }
    }
}

static void ht_remove_lumps(headtracker_t& ctx) {
    float mindist = 1;
    float min3dist = ctx.config.keypoint_3distance * 0.6 * ctx.zoom_ratio;
    min3dist *= min3dist;
    float min10dist = ctx.config.keypoint_10distance * 0.7 * ctx.zoom_ratio;
    min10dist *= min10dist;
    int max = ctx.config.max_keypoints;
    for (int i = 0; i < max; i++) {
        bool foundp = false;
        if (ctx.keypoints[i].idx == -1)
            continue;
        int threes = 0;
        int tens = 0;
        for (int j = 0; j < i; j++) {
            if (ctx.keypoints[j].idx == -1 )
                continue;
            float x = ctx.keypoints[j].position.x - ctx.keypoints[i].position.x;
            float y = ctx.keypoints[j].position.y - ctx.keypoints[i].position.y;
            float d = x * x + y * y;
            if (d < min3dist)
                threes++;
            if (d < min10dist)
                tens++;
            if ((d < mindist || tens >= 10 || threes >= 3) && ctx.keypoints[i].frames < ctx.keypoints[j].frames)
            {
                foundp = true;
                break;
            }
        }
        if (foundp) {
            ctx.keypoints[i].idx = -1;
            ctx.keypoint_count--;
        }
    }
}

void ht_remove_outliers(headtracker_t& ctx) {
    for (int i = 0; i < ctx.config.max_keypoints; i++) {
        if (ctx.keypoints[i].idx != -1 && !ht_triangle_exists(ctx.keypoints[i].position, ctx.model)) {
            ctx.keypoints[i].idx = -1;
            ctx.keypoint_count--;
        }
    }
}

void ht_track_features(headtracker_t& ctx) {
    if (ctx.restarted) {
        buildOpticalFlowPyramid(ctx.grayscale,
                                *ctx.pyr_a,
                                cvSize(ctx.config.pyrlk_win_size_w, ctx.config.pyrlk_win_size_h),
                                ctx.config.pyrlk_pyramids);
        return;
    }

    buildOpticalFlowPyramid(ctx.grayscale,
                            *ctx.pyr_b,
                            cvSize(ctx.config.pyrlk_win_size_w, ctx.config.pyrlk_win_size_h),
                            ctx.config.pyrlk_pyramids);
    int cnt = ctx.keypoint_count;
    if (cnt > 0) {
        int k = 0;

        vector<Point2f> new_features = vector<Point2f>(cnt);
        Mat features_found = Mat(cnt, 1, CV_8U);
        vector<Point2f> old_features = vector<Point2f>(cnt);

        for (int i = 0; i < ctx.config.max_keypoints; i++) {
            if (ctx.keypoints[i].idx == -1)
                continue;
            old_features[k++] = ctx.keypoints[i].position;
        }

        if (k > 0) {
            calcOpticalFlowPyrLK(*ctx.pyr_a,
                                 *ctx.pyr_b,
                                 old_features,
                                 new_features,
                                 features_found,
                                 noArray(),
                                 cvSize(ctx.config.pyrlk_win_size_w, ctx.config.pyrlk_win_size_h),
                                 ctx.config.pyrlk_pyramids,
                                 TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.01),
                                 OPTFLOW_LK_GET_MIN_EIGENVALS,
                                 ctx.config.pyrlk_min_eigenval);
            for (int i = 0, j = 0; i < k; i++, j++) {
                for (; j < ctx.config.max_keypoints && ctx.keypoints[j].idx == -1; j++)
                    ;;
                if (j == ctx.config.max_keypoints)
                    break;
                if (!features_found.at<char>(i)) {
                    ctx.keypoints[j].idx = -1;
                        ctx.keypoint_count--;
                } else {
                    ctx.keypoints[j].position = new_features[i];
                }
            }
        }
    }
    std::swap(ctx.pyr_a, ctx.pyr_b);
}

static bool ht_feature_quality_level(const KeyPoint x, const KeyPoint y) {
    return x.response < y.response;
}

void ht_get_features(headtracker_t& ctx, model_t& model) {
    ht_remove_lumps(ctx);

    if (!model.projection)
        return;

    CvRect roi = ht_get_roi(ctx, ctx.model);

    float max_dist = max(1.5f, ctx.config.keypoint_distance * ctx.zoom_ratio);
    float max_3dist = max(2.0f, ctx.config.keypoint_3distance * ctx.zoom_ratio);
    float max_10dist = ctx.config.keypoint_10distance * ctx.zoom_ratio;
    max_dist *= max_dist;
    max_3dist *= max_3dist;
    max_10dist *= max_10dist;
    int good = 0;
    int cnt = ctx.keypoint_count;
    if (cnt < ctx.config.max_keypoints) {
        vector<KeyPoint> corners;
        ORB detector = ORB(ctx.config.max_keypoints * 8,
                           1.2f,
                           8,
                           ctx.config.keypoint_quality,
                           0,
                           2,
                           ORB::HARRIS_SCORE,
                           ctx.config.keypoint_quality);
        Mat img = ctx.grayscale(roi);
        detector(img, noArray(), corners);
        sort(corners.begin(), corners.end(), ht_feature_quality_level);
        int cnt = corners.size();
        vector<Point2f> keypoints_to_add(corners.size());

        for (int i = 0; i < cnt; i++) {
            corners[i].pt.x += roi.x;
            corners[i].pt.y += roi.y;

            CvPoint2D32f kp = corners[i].pt;

            bool overlap = false;
            int threes = 0;
            int tens = 0;

            for (int j = 0; j < i; j++) {
                float d = ht_distance2d_squared(kp, corners[j].pt);

                if (d < max_3dist)
                    threes++;

                if (d < max_10dist)
                    tens++;

                if (d < max_dist || threes >= 3 || tens >= 10) {
                    overlap = true;
                    break;
                }
            }

            if (overlap)
                continue;

            if (!ht_triangle_exists(kp, model))
                continue;

            keypoints_to_add[good++] = corners[i].pt;
        }

        if (good < ctx.config.max_keypoints/2) {
            if (ctx.config.keypoint_quality > HT_FEATURE_MIN_QUALITY_LEVEL)
                ctx.config.keypoint_quality--;
        } else {
            if (ctx.config.keypoint_quality < HT_FEATURE_MAX_QUALITY_LEVEL)
                ctx.config.keypoint_quality++;
        }

        int kpidx = 0;
        for (int i = 0; i < good && ctx.keypoint_count < ctx.config.max_keypoints; i++) {
            CvPoint2D32f kp = keypoints_to_add[i];
            bool overlap = false;
            int threes = 0;
            int tens = 0;

            for (int j = 0; j < ctx.config.max_keypoints; j++) {
                float dist = ht_distance2d_squared(kp, ctx.keypoints[j].position);
                if (ctx.keypoints[j].idx != -1) {
                    if (dist < max_3dist)
                        ++threes;
                    if (dist < max_10dist)
                        ++tens;
                    if (dist < max_dist || threes >= 3 || tens >= 10) {
                        overlap = true;
                        break;
                    }
                }
            }

            if (overlap)
                continue;

            triangle_t t;
            int idx;
            CvPoint2D32f uv;

            if (!ht_triangle_at(kp, &t, &idx, model, uv))
                continue;

            for (; kpidx < ctx.config.max_keypoints; kpidx++) {
                if (ctx.keypoints[kpidx].idx == -1) {
                    ctx.keypoints[kpidx].idx = idx;
                    ctx.keypoints[kpidx].position = kp;
                    ctx.keypoint_uv[kpidx] = ht_get_triangle_pos(uv, t);
                    ctx.keypoints[kpidx].frames = 0;
                    ctx.keypoint_count++;
                    break;
                }
            }
        }
    }
}
