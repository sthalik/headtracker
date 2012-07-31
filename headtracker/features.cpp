#include "stdafx.h"

using namespace std;
using namespace cv;

void ht_remove_lumps(headtracker_t& ctx) {
	float max = ctx.config.min_feature_distance * ctx.config.filter_lumps_distance_threshold * ctx.zoom_ratio;
	float threshold = ctx.config.max_tracked_features * ctx.config.filter_lumps_feature_count_threshold;
	if (ctx.feature_count > threshold) {
		for (int i = 0; i < ctx.model.count; i++) {
			if (ctx.features[i].x == -1 || ctx.feature_failed_iters == 0)
				continue;
			for (int j = 0; j < i; j++) {
				if (ctx.features[j].x == -1)
					continue;
				float dist = sqrt(ht_distance2d_squared(ctx.features[i], ctx.features[j]));
				if (dist < max) {
					int idx = ctx.feature_failed_iters[i] >= ctx.feature_failed_iters[j]
									? i
									: j;
					ctx.features[idx].x = -1;
					ctx.feature_failed_iters[idx] = 0;
					ctx.feature_count--;
				}
			}
		}
	}
	if (ctx.feature_count > threshold) {
		for (int i = 0; i < ctx.model.count; i++) {
			if (ctx.features[i].x == -1)
				continue;
			for (int j = 0; j < i; j++) {
				if (ctx.features[j].x == -1)
					continue;
				float dist = sqrt(ht_distance2d_squared(ctx.features[i], ctx.features[j]));
				if (dist < max) {
					ctx.features[i].x = -1;
					ctx.feature_failed_iters[i] = 0;
					ctx.feature_count--;
				}
			}
		}
	}
}

void ht_draw_features(headtracker_t& ctx) {
	if (!ctx.features)
		return;

	for (int i = 0; i < ctx.model.count; i++) {
		if (ctx.features[i].x == -1)
			continue;

		CvScalar color;
		int size;

		if (ctx.feature_failed_iters[i] == 0) {
			color = CV_RGB(0, 255, 255);
			size = 1;
		} else {
			color = CV_RGB(255, 0, 0);
			size = 2;
		}

		cvCircle(ctx.color, cvPoint(ctx.features[i].x, ctx.features[i].y), size, color, -1);
	}

	for (int i = 0; i < ctx.config.max_keypoints; i++) {
		if (ctx.keypoints[i].idx != -1)
			cvCircle(ctx.color, cvPoint(ctx.keypoints[i].position.x, ctx.keypoints[i].position.y), 1, CV_RGB(255, 0, 255), -1);
	}
}

void ht_track_features(headtracker_t& ctx) {
	bool pyr_b_ready = false;

	if (!ctx.features)
		return;

	if (!ctx.last_image)
		return;

	int got_pyr = ctx.pyr_a != NULL;

	if (!ctx.pyr_a) {
		CvSize pyr_size = cvSize(ctx.grayscale->width+8, ctx.grayscale->height/3);
		ctx.pyr_a = cvCreateImage(pyr_size, IPL_DEPTH_32F, 1);
		ctx.pyr_b = cvCreateImage(pyr_size, IPL_DEPTH_32F, 1);
	}

	int sz = 0, max = ctx.model.count;
	CvPoint2D32f* old_features = new CvPoint2D32f[ctx.model.count];

	for (int i = 0; i < max; i++) {
		if (ctx.features[i].x != -1)
			old_features[sz++] = ctx.features[i];
	}

	if (sz > 0) {
		char* features_found = new char[sz];
		CvPoint2D32f* new_features = new CvPoint2D32f[sz];
	
		cvCalcOpticalFlowPyrLK(
			ctx.last_image,
			ctx.grayscale,
			ctx.pyr_a,
			ctx.pyr_b,
			old_features,
			new_features,
			sz,
			cvSize(ctx.config.pyrlk_win_size_w, ctx.config.pyrlk_win_size_h),
			ctx.config.pyrlk_pyramids,
			features_found,
			NULL,
			cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.04999),
			OPTFLOW_LK_GET_MIN_EIGENVALS | ((got_pyr && !ctx.restarted) ? CV_LKFLOW_PYR_A_READY : 0));
		
		pyr_b_ready = true;

		for (int i = 0, j = 0; i < sz; i++, j++) {
			for (; j < ctx.model.count; j++)
				if (ctx.features[j].x != -1)
					break;

			if (j == ctx.model.count)
				break;

			if (!features_found[i]) {
				ctx.features[j].x = -1;
				ctx.feature_count--;
				ctx.feature_failed_iters[j] = 0;
			} else
				ctx.features[j] = new_features[i];
		}

		delete[] features_found;
		delete[] new_features;
	}

	if (ctx.keypoint_count > 0) {
		int sz = ctx.keypoint_count;
		char* features_found = new char[sz];
		CvPoint2D32f* new_features = new CvPoint2D32f[sz];

		int k = 0;

		for (int i = 0; i < ctx.config.max_keypoints; i++) {
			if (ctx.keypoints[i].idx == -1)
				continue;
			old_features[k++] = ctx.keypoints[i].position;
		}

		if (k > 0) {
			cvCalcOpticalFlowPyrLK(ctx.last_image,
								   ctx.grayscale,
								   ctx.pyr_a,
								   ctx.pyr_b,
								   old_features,
								   new_features,
								   k,
								   cvSize(ctx.config.pyrlk_win_size_w, ctx.config.pyrlk_win_size_h),
								   ctx.config.pyrlk_pyramids,
								   features_found,
								   NULL,
								   cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.04999),
								   CV_LKFLOW_GET_MIN_EIGENVALS
								       | ((got_pyr && !ctx.restarted)
									       ? (CV_LKFLOW_PYR_A_READY | (pyr_b_ready && CV_LKFLOW_PYR_B_READY))
										   : 0));
			for (int i = 0, j = 0; i < k; i++, j++) {
				for (; j < ctx.config.max_keypoints && ctx.keypoints[j].idx == -1; j++)
					;;
				if (j == ctx.config.max_keypoints)
					break;
				if (!features_found[i]) {
					ctx.keypoints[j].idx = -1;
					ctx.keypoint_count--;
				} else {
					ctx.keypoints[j].position = new_features[i];
				}
			}
		}

		delete[] features_found;
		delete[] new_features;
	}

	IplImage* tmp = ctx.pyr_a;
	ctx.pyr_a = ctx.pyr_b;
	ctx.pyr_b = tmp;

	delete[] old_features;
}

static bool ht_feature_quality_level(const KeyPoint x, const KeyPoint y) {
	return x.response < y.response;
}

void ht_get_features(headtracker_t& ctx, model_t& model) {
	if (!(ctx.keypoint_count < ctx.config.max_keypoints ||
		  ctx.feature_count < ctx.config.max_tracked_features * ctx.config.features_detect_threshold))
		  return;

	float min_x = (float) ctx.grayscale->width, max_x = 0.0f;
	float min_y = (float) ctx.grayscale->height, max_y = 0.0f;

	int sz = model.count;

	if (!ctx.features) {
		ctx.features = new CvPoint2D32f[sz];
		for (int i = 0; i < sz; i++)
			ctx.features[i] = cvPoint2D32f(-1, -1);
		ctx.feature_failed_iters = new char[sz];
		for (int i = 0; i < sz; i++)
			ctx.feature_failed_iters[i] = 0;
	}

	for (int i = 0; i < ctx.model.count; i++) {
		float x = (model.projection[i].p1.x + model.projection[i].p2.x + model.projection[i].p3.x) / 3;
		float y = (model.projection[i].p1.x + model.projection[i].p2.y + model.projection[i].p3.y) / 3;
		if (x > max_x)
			max_x = x;
		if (x < min_x)
			min_x = x;
		if (y > max_y)
			max_y = y;
		if (y < min_y)
			min_y = y;
	}
	
	if (!model.projection)
		return;

	CvRect roi = cvRect(min_x, min_y, max_x - min_x, max_y - min_y);

	roi.x = max(0, min(roi.x, ctx.grayscale->width - 1));
	roi.y = max(0, min(roi.y, ctx.grayscale->height - 1));
	roi.width = max(0, min(ctx.grayscale->width - roi.x, roi.width));
	roi.height = max(0, min(ctx.grayscale->height - roi.y, roi.height));

	if (roi.width == 0 || roi.height == 0)
		return;

	vector<KeyPoint> corners;

	Mat mat = Mat(Mat(ctx.grayscale, false), roi);

	float max_dist = ctx.config.keypoint_distance * ctx.zoom_ratio;
start_keypoints:
	int good = 0;
	if (ctx.keypoint_count < ctx.config.max_keypoints) {
		max_dist *= max_dist;
		ORB detector = ORB(ctx.config.max_keypoints * 32, 1.1f, 16, ctx.config.keypoint_quality, 0, 2, 0, ctx.config.feature_quality_level);
		detector(mat, noArray(), corners);
		sort(corners.begin(), corners.end(), ht_feature_quality_level);
		int cnt = corners.size();
		CvPoint2D32f* keypoints_to_add = new CvPoint2D32f[cnt];

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
					delete[] keypoints_to_add;
					goto start_keypoints;
				}
			}
		} else {
			if (ctx.config.keypoint_quality < HT_FEATURE_MAX_QUALITY_LEVEL)
				ctx.config.keypoint_quality++;
		}

		if (good > 0) {
			if (roi.width > 17 && roi.height > 13)
				cvFindCornerSubPix(ctx.grayscale, keypoints_to_add, good, cvSize(8, 6), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.05));
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

				if (!ht_triangle_at(kp, &t, &idx, model))
					continue;

				for (; kpidx < ctx.config.max_keypoints; kpidx++) {
					if (ctx.keypoints[kpidx].idx == -1) {
						ctx.keypoints[kpidx].idx = idx;
						ctx.keypoints[kpidx].position = kp;
						ctx.keypoint_count++;
						ctx.keypoint_failed_iters[kpidx] = 0;
						break;
					}
				}
			}
		}

		delete[] keypoints_to_add;
	}

	corners.clear();
	
	if (ctx.feature_count >= ctx.config.max_tracked_features * ctx.config.features_detect_threshold)
		return;

	int max = ctx.state == HT_STATE_TRACKING
		? ctx.config.max_tracked_features
		: ctx.config.min_track_start_features;

redetect:
	good = 0;
	PyramidAdaptedFeatureDetector pyrfast(new FastFeatureDetector(ctx.config.feature_quality_level), 5);
	pyrfast.detect(mat, corners);

	sort(corners.begin(), corners.end(), ht_feature_quality_level);

	int count = corners.size(), k = 0;

	if (count == 0)
		return;

	CvPoint2D32f* features_to_add = new CvPoint2D32f[count];

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
				delete[] features_to_add;
				goto redetect;
			}
		}
	}

	if (k > 0) {
		if (roi.width > 17 && roi.height > 13)
			cvFindCornerSubPix(ctx.grayscale, features_to_add, k, cvSize(8, 6), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.05));
		for (int i = 0; i < k && ctx.feature_count < ctx.config.max_tracked_features; i++) {
			triangle_t t;
			int idx;

			for (int j = 0; j < model.count; j++)
				if (ctx.features[j].x != -1 && ht_distance2d_squared(features_to_add[i], ctx.features[j]) < max_distance)
					goto end2;

			if (!(ht_triangle_at(features_to_add[i], &t, &idx, model)))
				continue;

			if (ctx.features[idx].x != -1)
				continue;

			ctx.features[idx] = features_to_add[i];
			ctx.feature_count++;
	end2:
			;
		}
	}

	delete[] features_to_add;
}
