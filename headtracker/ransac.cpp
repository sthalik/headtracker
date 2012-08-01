#include "stdafx.h"

using namespace std;
using namespace cv;

static __inline error_t ht_avg_reprojection_error(headtracker_t& ctx, CvPoint3D32f* model_points, CvPoint2D32f* image_points, int point_cnt) {
	float rotation_matrix[9];
	float translation_vector[3];
	float focal_length = ctx.config.focal_length;

	error_t ret;

	if (!ht_posit(image_points,
				  model_points,
				  point_cnt,
				  rotation_matrix,
				  translation_vector,
				  cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, ctx.config.ransac_posit_iter, ctx.config.ransac_posit_eps),
				  focal_length)) {
		ret.avg = 1.0e10;
		return ret;
	}

	float avg = 0;

	for (int i = 0; i < point_cnt; i++)
		avg += ht_distance2d_squared(ht_project_point(model_points[i], rotation_matrix, translation_vector, focal_length), image_points[i]);

	ret.avg = (float) sqrt(avg / point_cnt);

	return ret;
}

void ht_fisher_yates(int* indices, int count) {
	int tmp;

	for (int i = count - 1; i > 0; i--) {
		int j = rand() % i;
		tmp = indices[i];
		indices[i] = indices[j];
		indices[j] = tmp;
	}
}

bool ht_ransac(headtracker_t& ctx,
			   int max_iter,
			   float max_error,
			   int min_consensus,
			   int* best_feature_cnt,
			   int* best_keypoint_cnt,
			   error_t* best_error,
			   int* best_indices,
			   int* best_keypoints,
			   float error_scale)
{
	if (ctx.keypoint_count == 0)
		return false;

	int mcnt = ctx.model.count;
	int k = 0;
	bool ret = false;
	float max_consensus_error = ctx.config.ransac_max_consensus_error;
	float importance = ctx.config.ransac_best_error_importance;
	int* keypoint_indices = new int[ctx.keypoint_count];
	int* model_feature_indices = new int[mcnt];
	int* model_keypoint_indices = new int[ctx.keypoint_count];
	int* indices = new int[mcnt];
	int bad = 0;

	best_error->avg = 1.0e10;
	*best_feature_cnt = 0;
	*best_keypoint_cnt = 0;

	for (int i = 0; i < mcnt; i++) {
		if (ctx.features[i].x != -1)
			indices[k++] = i;
	}

	int kppos = 0;

	for (int i = 0; i < ctx.keypoint_count; i++) {
		if (ctx.keypoints[i].idx != -1)
			keypoint_indices[kppos++] = i;
	}

	CvPoint2D32f* image_points = new CvPoint2D32f[k + kppos];
	CvPoint3D32f* model_points = new CvPoint3D32f[k + kppos];

	if ((k + kppos) < min_consensus || (k + kppos) < 4)
		goto end;

	for (int iter = 0; iter < max_iter; iter++) {
		ht_fisher_yates(indices, k);
		ht_fisher_yates(keypoint_indices, kppos);
		int ipos = 0;
		int fpos = 0;
		int kpos = 0;
		int gfpos = 0;
		int gkpos = 0;
		bool good = false;

		CvPoint3D32f first_point = ctx.feature_uv[indices[0]];

		error_t cur_error;
		cur_error.avg = max_consensus_error - 1.0e-2f;

		if (cur_error.avg >= max_consensus_error)
			continue;

		for (; fpos < k; fpos++) {
			int idx = indices[fpos];
			model_points[ipos] = ctx.feature_uv[idx];
			model_points[ipos].x -= first_point.x;
			model_points[ipos].y -= first_point.y;
			model_points[ipos].z -= first_point.z;
			image_points[ipos] = ctx.features[idx];
			model_feature_indices[gfpos] = idx;

			if (ipos >= ctx.config.ransac_min_features) {
				error_t e = ht_avg_reprojection_error(ctx, model_points, image_points, ipos+1);
				e.avg *= error_scale;
				if (e.avg*max_error > cur_error.avg)
					continue;
				cur_error.avg = e.avg;

				if (cur_error.avg > max_consensus_error)
					goto end2;

				good = true;
			}

			ipos++;
			gfpos++;

			float measure = (1.0 - importance) + importance * best_error->avg / cur_error.avg;

			if (ipos >= min_consensus && ipos * measure * measure > *best_feature_cnt + *best_keypoint_cnt) {
				ret = true;
				*best_error = cur_error;
				*best_feature_cnt = gfpos;
				*best_keypoint_cnt = gkpos;
				for (int i = 0; i < gfpos; i++)
					best_indices[i] = model_feature_indices[i];
				for (int i = 0; i < gkpos; i++)
					best_keypoints[i] = model_keypoint_indices[i];
			}
		}

		for (; kpos < kppos; kpos++) {
			int idx = keypoint_indices[kpos];
			ht_keypoint& kp = ctx.keypoints[idx];
			model_points[ipos] = ctx.keypoint_uv[idx];
			model_points[ipos].x -= first_point.x;
			model_points[ipos].y -= first_point.y;
			model_points[ipos].z -= first_point.z;
			image_points[ipos] = kp.position;
			model_keypoint_indices[gkpos] = idx;
			if (ipos >= ctx.config.ransac_min_features) {
				error_t e = ht_avg_reprojection_error(ctx, model_points, image_points, ipos+1);
				e.avg *= error_scale;
				if (e.avg*max_error > cur_error.avg)
					continue;
				cur_error.avg = e.avg;
				if (cur_error.avg > max_consensus_error)
					goto end2;
				good = true;
			}

			ipos++;
			gkpos++;

			float measure = (1.0 - importance) + importance * best_error->avg / cur_error.avg;

			if (ipos >= min_consensus && ipos * measure > *best_feature_cnt + *best_keypoint_cnt) {
				ret = true;
				*best_error = cur_error;
				*best_feature_cnt = gfpos;
				*best_keypoint_cnt = gkpos;
				for (int i = 0; i < gfpos; i++)
					best_indices[i] = model_feature_indices[i];
				for (int i = 0; i < gkpos; i++)
					best_keypoints[i] = model_keypoint_indices[i];
			}
		}
end2:
		if (!good)
			bad++;
	}

	if (ctx.config.debug)
		printf("RANSAC: %d out of %d iterations failed completely\n", bad, max_iter);

end:

	delete[] keypoint_indices;
	delete[] indices;
	delete[] image_points;
	delete[] model_points;
	delete[] model_feature_indices;
	delete[] model_keypoint_indices;

	return ret;
}

bool ht_ransac_best_indices(headtracker_t& ctx, error_t* best_error) {
	int min_features = ctx.state == HT_STATE_TRACKING ? ctx.feature_count * ctx.config.ransac_min_consensus : ctx.config.min_track_start_features;
	int* best_feature_indices = new int[ctx.feature_count];
	int* best_keypoint_indices = new int[ctx.keypoint_count];
	int best_feature_cnt, best_keypoint_cnt;
	if (ht_ransac(ctx,
				  ctx.config.ransac_iter,
				  ctx.config.ransac_max_error,
				  min_features,
				  &best_feature_cnt,
				  &best_keypoint_cnt,
				  best_error,
				  best_feature_indices,
				  best_keypoint_indices,
				  ctx.zoom_ratio)) {
		char* fusedp = new char[ctx.model.count];
		char* kusedp = new char[ctx.config.max_keypoints];
		for (int i = 0; i < ctx.model.count; i++)
			fusedp[i] = 0;
		for (int i = 0; i < best_feature_cnt; i++) {
			fusedp[best_feature_indices[i]] = 1;
		}
		for (int i = 0; i < ctx.model.count; i++) {
			if (!fusedp[i]) {
				if (ctx.features[i].x != -1) {
					if (++ctx.feature_failed_iters[i] > ctx.config.feature_max_failed_ransac) {
						ctx.features[i].x = -1;
						ctx.feature_count--;
						ctx.feature_failed_iters[i] = 0;
					}
				}
			} else {
				ctx.feature_failed_iters[i] = 0;
			}
		}
		for (int i = 0; i < ctx.config.max_keypoints; i++)
			kusedp[i] = 0;
		for (int i = 0; i < best_keypoint_cnt; i++)
			kusedp[best_keypoint_indices[i]] = 1;
		for (int i = 0; i < ctx.config.max_keypoints; i++) {
			if (!kusedp[i] && ctx.keypoints[i].idx != -1) {
				if (++ctx.keypoint_failed_iters[i] > ctx.config.keypoint_max_failed_ransac) {
					ctx.keypoints[i].idx = -1;
					ctx.keypoint_failed_iters[i] = 0;
					ctx.keypoint_count--;
				}
			} else {
				ctx.keypoint_failed_iters[i] = 0;
			}
		}
		delete[] fusedp;
		delete[] kusedp;
		delete[] best_keypoint_indices;
		delete[] best_feature_indices;
		return true;
	}
	delete[] best_keypoint_indices;
	delete[] best_feature_indices;
	return false;
}