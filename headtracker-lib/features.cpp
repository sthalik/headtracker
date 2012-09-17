#include "stdafx.h"

using namespace std;
using namespace cv;

void ht_remove_lumps(headtracker_t& ctx) {
    float max = ctx.config.min_feature_distance * ctx.config.filter_lumps_distance_threshold * ctx.zoom_ratio;
    float threshold = ctx.config.max_tracked_features * ctx.config.filter_lumps_feature_count_threshold;
    if (ctx.feature_count > threshold) {
        for (int i = 0; i < ctx.model.count && ctx.feature_count > threshold; i++) {
            if (ctx.features[i].x == -1)
                continue;
            for (int j = 0; j < i; j++) {
                if (ctx.features[j].x == -1)
                    continue;
                float dist = sqrt(ht_distance2d_squared(ctx.features[i], ctx.features[j]));
                if (dist < max) {
                    ctx.features[i].x = -1;
                    ctx.feature_count--;
                }
            }
        }
    }
}

void ht_draw_features(headtracker_t& ctx) {
    for (int i = 0; i < ctx.model.count; i++) {
        if (ctx.features[i].x == -1)
            continue;

        circle(ctx.color, cvPoint(ctx.features[i].x, ctx.features[i].y), 2, Scalar(255, 255, 0), -1);
    }

    for (int i = 0; i < ctx.config.max_keypoints; i++) {
        if (ctx.keypoints[i].idx != -1)
            circle(ctx.color, cvPoint(ctx.keypoints[i].position.x, ctx.keypoints[i].position.y), 2, CV_RGB(255, 0, 255), -1);
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

    int sz = 0, maxTriangles = ctx.model.count;

    vector<Point2f> new_features = vector<Point2f>(sz);
    Mat features_found = Mat(sz, 1, CV_8U);
    vector<Point2f> old_features = vector<Point2f>();

    for (int i = 0; i < maxTriangles; i++) {
        if (ctx.features[i].x != -1) {
            old_features.push_back(ctx.features[i]);
            sz++;
        }
    }

    if (sz > 0) {
        calcOpticalFlowPyrLK(*ctx.pyr_a,
                             *ctx.pyr_b,
                             old_features,
                             new_features,
                             features_found,
                             noArray(),
                             cvSize(ctx.config.pyrlk_win_size_w, ctx.config.pyrlk_win_size_h),
                             ctx.config.pyrlk_pyramids,
                             TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 40, 0.009),
                             0,
                             ctx.config.pyrlk_min_eigenval);

        for (int i = 0, j = 0; i < sz; i++, j++) {
            for (; j < ctx.model.count; j++)
                if (ctx.features[j].x != -1)
                    break;

            if (j == ctx.model.count)
                break;

            if (!features_found.at<char>(i)) {
                ctx.features[j].x = -1;
                ctx.feature_count--;
            } else
                ctx.features[j] = new_features[i];
        }
    }

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
                                 TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 40, 0.009),
                                 0,
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
    if (!(ctx.keypoint_count < ctx.config.max_keypoints ||
          ctx.feature_count < ctx.config.max_tracked_features * ctx.config.features_detect_threshold))
          return;

    float min_x = (float) ctx.grayscale.cols, max_x = 0.0f;
    float min_y = (float) ctx.grayscale.rows, max_y = 0.0f;

    int sz = model.count;

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

    float max_dist = ctx.config.keypoint_distance * ctx.zoom_ratio;
start_keypoints:
    int good = 0;
    if (ctx.keypoint_count < ctx.config.max_keypoints) {
        max_dist *= max_dist;
        ORB detector = ORB(ctx.config.max_keypoints * 20, 1.1f, 12, ctx.config.keypoint_quality, 0, 2, 0, ctx.config.keypoint_quality);
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

        if (good < ctx.config.max_keypoints) {
            if (ctx.config.keypoint_quality > HT_FEATURE_MIN_QUALITY_LEVEL) {
                ctx.config.keypoint_quality--;
                if (ctx.state == HT_STATE_INITIALIZING) {
                    corners.clear();
                    goto start_keypoints;
                }
            }
        } else {
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

    corners.clear();

    if (ctx.feature_count >= ctx.config.max_tracked_features * ctx.config.features_detect_threshold)
        return;

    int max = ctx.config.max_tracked_features;

redetect:
    good = 0;
    PyramidAdaptedFeatureDetector pyrfast(new FastFeatureDetector(ctx.config.feature_quality_level), 3);
    pyrfast.detect(mat, corners);

    sort(corners.begin(), corners.end(), ht_feature_quality_level);

    int count = corners.size(), k = 0;

    if (count == 0)
        return;

    vector<Point2f> features_to_add(corners.size());

    float max_distance = ctx.config.min_feature_distance * ctx.zoom_ratio;
    max_distance *= max_distance;

    for (int i = 0; i < count; i++) {
        corners[i].pt.x += roi.x;
        corners[i].pt.y += roi.y;

        for (int j = 0; j < i; j++)
            if (i != j && ht_distance2d_squared(corners[i].pt, corners[j].pt) < max_distance)
                goto end;

        if (!ht_triangle_exists(corners[i].pt, model))
            continue;

        good++;

        features_to_add[k++] = corners[i].pt;
end:
        ;
    }

    if (good > max * ctx.config.feature_detect_ratio) {
        if (ctx.config.feature_quality_level < HT_FEATURE_MAX_QUALITY_LEVEL)
            ctx.config.feature_quality_level++;
    } else {
        if (ctx.config.feature_quality_level > HT_FEATURE_MIN_QUALITY_LEVEL) {
            ctx.config.feature_quality_level--;
            if (ctx.state == HT_STATE_INITIALIZING) {
                corners.clear();
                goto redetect;
            }
        }
    }

    if (k > 0) {
        for (int i = 0; i < k && ctx.feature_count < ctx.config.max_tracked_features; i++) {
            triangle_t t;
            int idx;
            CvPoint2D32f uv;

            for (int j = 0; j < model.count; j++)
                if (ctx.features[j].x != -1 && ht_distance2d_squared(features_to_add[i], ctx.features[j]) < max_distance)
                    goto end2;

            if (!(ht_triangle_at(features_to_add[i], &t, &idx, model, uv)))
                continue;

            if (ctx.features[idx].x != -1)
                continue;

            ctx.features[idx] = features_to_add[i];
            ctx.feature_uv[idx] = ht_get_triangle_pos(uv, t);
            ctx.feature_count++;
    end2:
            ;
        }
    }
}