#include "stdafx.h"

using namespace std;
using namespace cv;

error_t ht_avg_reprojection_error(headtracker_t& ctx, CvPoint3D32f* model_points, CvPoint2D32f* image_points, int point_cnt) {
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
			   int iter_points,
			   float max_error,
			   int min_consensus,
			   int* best_cnt,
			   error_t* best_error,
			   int* best_indices,
			   model_t& model,
			   float error_scale)
{
	int mcnt = ctx.model.count;
	int* indices = new int[mcnt];
	CvPoint2D32f* image_points = new CvPoint2D32f[mcnt];
	CvPoint3D32f* model_points = new CvPoint3D32f[mcnt];
	int k = 0;
	int* model_indices = new int[mcnt];
	bool ret = false;
	float max_consensus_error = ctx.config.ransac_max_consensus_error;
	float importance = ctx.config.ransac_best_error_importance;

	best_error->avg = 1.0e10;
	*best_cnt = 0;

	for (int i = 0; i < mcnt; i++) {
		if (ctx.features[i].x != -1)
			indices[k++] = i;
	}

	if (k < min_consensus || k < 4 || iter_points < 4)
		goto end;

	for (int iter = 0; iter < max_iter; iter++) {
		ht_fisher_yates(indices, k);

		int pos = 0;

		CvPoint3D32f first_point = model.centers[indices[0]];

		for (int i = 0; i < iter_points; i++) {
			int idx = indices[i];

			model_points[pos] = model.centers[idx];
			model_points[pos].x -= first_point.x;
			model_points[pos].y -= first_point.y;
			model_points[pos].z -= first_point.z;
			image_points[pos] = ctx.features[idx];
			model_indices[pos] = idx;
			pos++;
		}

		error_t cur_error = ht_avg_reprojection_error(ctx, model_points, image_points, pos);
		cur_error.avg *= error_scale;

		if (cur_error.avg >= max_consensus_error)
			continue;

		for (int i = iter_points; i < k; i++) {
			int idx = indices[i];
			model_points[pos] = model.centers[idx];
			model_points[pos].x -= first_point.x;
			model_points[pos].y -= first_point.y;
			model_points[pos].z -= first_point.z;
			image_points[pos] = ctx.features[idx];
			model_indices[pos] = idx;

			error_t e = ht_avg_reprojection_error(ctx, model_points, image_points, pos+1);
			e.avg *= error_scale;

			if (e.avg*max_error > cur_error.avg)
				continue;

			cur_error.avg = e.avg;
			pos++;

			if (cur_error.avg > max_consensus_error)
				goto end2;

			float measure = (1.0 - importance) + importance * best_error->avg / cur_error.avg;

			if (pos >= min_consensus && pos * measure * measure > *best_cnt) {
				ret = true;
				*best_error = e;
				*best_cnt = pos;
				for (int j = 0; j < pos; j++)
					best_indices[j] = model_indices[j];
			}
		}
end2:
		;
	}

end:

	delete[] indices;
	delete[] image_points;
	delete[] model_points;
	delete[] model_indices;

	return ret;
}

bool ht_ransac_best_indices(headtracker_t& ctx, int* best_cnt, error_t* best_error, int* best_indices) {
	int min_features = ctx.state == HT_STATE_TRACKING ? ctx.feature_count * ctx.config.ransac_min_consensus : ctx.config.min_track_start_features;
	if (ht_ransac(ctx,
				  ctx.config.ransac_iter,
				  ctx.config.ransac_min_features,
				  ctx.config.ransac_max_error,
				  min_features,
				  best_cnt,
				  best_error,
				  best_indices,
				  ctx.model,
				  ctx.zoom_ratio)) {
		char* usedp = new char[ctx.model.count];
		for (int i = 0; i < ctx.model.count; i++)
			usedp[i] = 0;
		for (int i = 0; i < *best_cnt; i++) {
			usedp[best_indices[i]] = 1;
		}
		for (int i = 0; i < ctx.model.count; i++) {
			if (!usedp[i]) {
				if (ctx.features[i].x != -1 && ctx.features[i].y != -1) {
					if (++ctx.feature_failed_iters[i] >= ctx.config.feature_max_failed_ransac) {
						ctx.features[i] = cvPoint2D32f(-1, -1);
						ctx.feature_count--;
					}
				}
			} else {
				ctx.feature_failed_iters[i] = 0;
			}
		}
		delete[] usedp;
		return true;
	}
	return false;
}