#include "stdafx.h"

using namespace std;
using namespace cv;

void ht_draw_features(headtracker_t& ctx) {
    for (int i = 0; i < ctx.config.max_keypoints; i++) {
        if (ctx.keypoints[i].idx != -1)
            circle(ctx.color, cvPoint(ctx.keypoints[i].position.x, ctx.keypoints[i].position.y), 1, CV_RGB(255, 0, 255), -1);
    }
}

static void ht_remove_lumps(headtracker_t& ctx) {
    float mindist = ctx.config.keypoint_distance / ctx.zoom_ratio;
    mindist /= 1.5;
    mindist *= mindist;
    for (int i = 0; i < ctx.config.max_keypoints && ctx.config.max_keypoints * 5 / 6 < ctx.keypoint_count; i++) {
        bool foundp = false;
        if (ctx.keypoints[i].idx == -1)
            continue;
        for (int j = 0; j < i; j++) {
            if (ctx.keypoints[j].idx == -1)
                continue;
            float x = ctx.keypoints[j].position.x - ctx.keypoints[i].position.x;
            float y = ctx.keypoints[j].position.y - ctx.keypoints[i].position.y;
            if (x * x + y * y < mindist) {
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

void ht_track_features(headtracker_t& ctx) {
    if (ctx.restarted)
        buildOpticalFlowPyramid(ctx.grayscale,
                                *ctx.pyr_a,
                                cvSize(ctx.config.pyrlk_win_size_w, ctx.config.pyrlk_win_size_h),
                                ctx.config.pyrlk_pyramids);

    buildOpticalFlowPyramid(ctx.grayscale,
                            *ctx.pyr_b,
                            cvSize(ctx.config.pyrlk_win_size_w, ctx.config.pyrlk_win_size_h),
                            ctx.config.pyrlk_pyramids);

    if (ctx.keypoint_count > 0) {
        int sz = ctx.keypoint_count;

        int k = 0;

        vector<Point2f> new_features = vector<Point2f>(sz);
        Mat features_found = Mat(sz, 1, CV_8U);
        vector<Point2f> old_features = vector<Point2f>(sz);

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
                                 TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 50, 0.009),
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

    if (ctx.keypoint_count >= ctx.config.max_keypoints)
          return;

    float min_x = (float) ctx.grayscale.cols, max_x = 0.0f;
    float min_y = (float) ctx.grayscale.rows, max_y = 0.0f;

    for (int i = 0; i < ctx.model.count; i++) {
        float minx = min(model.projection[i].p1.x, min(model.projection[i].p2.x, model.projection[i].p3.x));
        float maxx = max(model.projection[i].p1.x, max(model.projection[i].p2.x, model.projection[i].p3.x));
        float miny = min(model.projection[i].p1.y, min(model.projection[i].p2.y, model.projection[i].p3.y));
        float maxy = max(model.projection[i].p1.y, max(model.projection[i].p2.y, model.projection[i].p3.y));
        if (maxx > max_x)
            max_x = maxx;
        if (minx < min_x)
            min_x = minx;
        if (maxy > max_y)
            max_y = maxy;
        if (miny < min_y)
            min_y = miny;
    }

    if (!model.projection)
        return;

    CvRect roi = cvRect(min_x, min_y, max_x - min_x, max_y - min_y);

    roi.x = max(0, min(roi.x, ctx.grayscale.cols - 1));
    roi.y = max(0, min(roi.y, ctx.grayscale.rows - 1));
    roi.width = max(0, min(ctx.grayscale.cols - roi.x, roi.width));
    roi.height = max(0, min(ctx.grayscale.rows - roi.y, roi.height));

    if (roi.width == 0 || roi.height == 0)
        return;

    vector<KeyPoint> corners;

    Mat mat = ctx.grayscale(roi);

    float max_dist = max(1.5f, ctx.config.keypoint_distance / ctx.zoom_ratio);
start_keypoints:
    int good = 0;
    if (ctx.keypoint_count < ctx.config.max_keypoints) {
        max_dist *= max_dist;
        ORB detector = ORB(ctx.config.max_keypoints * 12, 1.2f, 8, ctx.config.keypoint_quality, 0, 2, 0, ctx.config.keypoint_quality);
        detector(mat, noArray(), corners);
        sort(corners.begin(), corners.end(), ht_feature_quality_level);
        int cnt = corners.size();
        vector<Point2f> keypoints_to_add(corners.size());

        for (int i = 0; i < cnt; i++) {
            corners[i].pt.x += roi.x;
            corners[i].pt.y += roi.y;

            CvPoint2D32f kp = corners[i].pt;

            bool overlap = false;

            for (int j = 0; j < i; j++)
                if (ht_distance2d_squared(kp, corners[j].pt) < max_dist) {
                    overlap = true;
                    break;
                }

            if (overlap)
                continue;

            if (!ht_triangle_exists(kp, model))
                continue;

            keypoints_to_add[good++] = corners[i].pt;
        }

        if (good + ctx.keypoint_count < ctx.config.max_keypoints) {
            if (ctx.config.keypoint_quality > HT_FEATURE_MIN_QUALITY_LEVEL) {
                ctx.config.keypoint_quality--;
                if (ctx.state == HT_STATE_INITIALIZING) {
                    corners.clear();
                    goto start_keypoints;
                }
            }
        } else if (good > ctx.config.max_keypoints) {
            if (ctx.config.keypoint_quality < HT_FEATURE_MAX_QUALITY_LEVEL)
                ctx.config.keypoint_quality++;
        }

        if (good > 0) {
            int kpidx = 0;
            for (int i = 0; i < good && ctx.keypoint_count < ctx.config.max_keypoints; i++) {
                CvPoint2D32f kp = keypoints_to_add[i];
                bool overlap = false;

                for (int j = 0; j < ctx.config.max_keypoints; j++) {
                    if (ctx.keypoints[j].idx != -1 && ht_distance2d_squared(kp, ctx.keypoints[j].position) < max_dist) {
                        overlap = true;
                        break;
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
                        ctx.keypoint_count++;
                        break;
                    }
                }
            }
        }
    }
}
