#include "ht-api.h"
#include "ht-internal.h"
using namespace std;
using namespace cv;

void ht_draw_features(headtracker_t& ctx) {
    int j = 0;
	float mult = ctx.color.cols / (float)ctx.grayscale.cols;
    for (int i = 0; i < ctx.config.max_keypoints; i++) {
        if (ctx.keypoints[i].idx != -1) {
            circle(ctx.color,
                   Point(ctx.keypoints[i].position.x * mult, ctx.keypoints[i].position.y * mult),
                   2,
                   Scalar(255, 255, 0),
                   -1);
            j++;
        }
    }
    if (ctx.config.debug)
        fprintf(stderr, "%d features\n", j);
}

void ht_track_features(headtracker_t& ctx) {

    if (ctx.restarted) {
        buildOpticalFlowPyramid(ctx.grayscale,
                                *ctx.pyr_a,
								Size(ctx.config.pyrlk_win_size_w, ctx.config.pyrlk_win_size_h),
                                ctx.config.pyrlk_pyramids);
    }

    buildOpticalFlowPyramid(ctx.grayscale,
                            *ctx.pyr_b,
                            Size(ctx.config.pyrlk_win_size_w, ctx.config.pyrlk_win_size_h),
                            ctx.config.pyrlk_pyramids);
    int k = 0;

    vector<Point2f> old_features;

    for (int i = 0; i < ctx.config.max_keypoints; i++) {
        if (ctx.keypoints[i].idx == -1)
            continue;
        old_features.push_back(ctx.keypoints[i].position);
        k++;
    }

    vector<Point2f> new_features = vector<Point2f>(k);
    Mat features_found(1, k, CV_8UC1);

    if (k > 0) {
        calcOpticalFlowPyrLK(*ctx.pyr_a,
		                     *ctx.pyr_b,
                             old_features,
                             new_features,
                             features_found,
                             noArray(),
                             Size(ctx.config.pyrlk_win_size_w, ctx.config.pyrlk_win_size_h),
                             ctx.config.pyrlk_pyramids,
                             TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.01));
        for (int i = 0, j = 0; i < k; i++, j++) {
            for (; j < ctx.config.max_keypoints && ctx.keypoints[j].idx == -1; j++)
                ;;
            if (j == ctx.config.max_keypoints)
                break;
            if (!features_found.at<char>(i)) {
                ctx.keypoints[j].idx = -1;
            } else {
                ctx.keypoints[j].position = new_features[i];
            }
        }
    }
    std::swap(ctx.pyr_a, ctx.pyr_b);
}

void ht_get_features(headtracker_t& ctx, model_t& model) {
    if (!model.projection)
        return;

    Rect roi = ht_get_bounds(ctx, model);

	if (!(roi.width > 20 && roi.height > 40))
		return;
    
    float max_dist = ctx.config.keypoint_distance * ctx.zoom_ratio;
    max_dist *= max_dist;
    vector<KeyPoint> corners;
	Mat img = ctx.grayscale(roi);
    //ORB foo(2000, 1.2, 8, 2, 0, 2, ORB::HARRIS_SCORE, 2);
    //foo.detect(img, corners);
	//Ptr<FeatureDetector> fast = FeatureDetector::create("FAST");
    //fast->detect(img, corners);
    //GridAdaptedFeatureDetector detector(fast, ctx.config.max_keypoints, 4, 2);
	//detector.detect(img, corners);
start:
    FASTX(img, corners, ctx.fast_state, true, FastFeatureDetector::TYPE_9_16);
    if (corners.size() < ctx.config.max_keypoints*1.5 && ctx.fast_state > 5)
    {
        corners.clear();
        ctx.fast_state--;
        goto start;
    }
    if (corners.size() > ctx.config.max_keypoints*3.0 && ctx.fast_state < 50)
    {
        corners.clear();
        ctx.fast_state++;
        goto start;
    }
    //ctx.detector->detect(img, corners);
    if (ctx.config.debug)
        fprintf(stderr, "new keypoints: %d\n", corners.size());
    int cnt = corners.size();
    int no_triangle = 0, overlapped = 0;

    int kpidx = 0;

    for (int i = 0; i < cnt; i++) {
        Point2f kp = corners[i].pt;
        kp.x += roi.x;
        kp.y += roi.y;
        bool overlap = false;

        for (int j = 0; j < ctx.config.max_keypoints; j++) {
            if (ctx.keypoints[j].idx != -1) {
				float dist = ht_distance2d_squared(kp, ctx.keypoints[j].position);
                if (dist < max_dist) {
					overlap = true;
					break;
                }
            }
        }

        if (overlap)
        {
            overlapped++;
            continue;
        }

        triangle_t t;
        int idx;
        Point2f uv;

        if (!ht_triangle_at(kp, &t, &idx, model, uv))
        {
            no_triangle++;
            continue;
        }

        for (; kpidx < ctx.config.max_keypoints; kpidx++) {
            if (ctx.keypoints[kpidx].idx == -1) {
                ctx.keypoints[kpidx].idx = idx;
                ctx.keypoints[kpidx].position = kp;
                ctx.keypoint_uv[kpidx] = ht_get_triangle_pos(uv, t);
                break;
            }
        }
        if (kpidx == ctx.config.max_keypoints)
            break;
    }
    if (ctx.config.debug)
        fprintf(stderr, "no-triangle=%d, overlapped=%d\n", no_triangle, overlapped);
}
