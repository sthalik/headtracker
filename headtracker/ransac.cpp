#include "stdafx.h"

using namespace std;
using namespace cv;

error_t ht_avg_reprojection_error(headtracker_t& ctx, CvPoint3D32f* model_points, CvPoint2D32f* image_points, int point_cnt) {
	float rotation_matrix[9];
	float translation_vector[3];

	error_t ret;

	if (!ht_posit(image_points, model_points, point_cnt, rotation_matrix, translation_vector, cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, HT_RANSAC_POSIT_ITER, HT_RANSAC_POSIT_EPS))) {
		ret.avg = 1.0e10;
		return ret;
	}

	float avg = 0;

	for (int i = 0; i < point_cnt; i++)
		avg += ht_distance2d_squared(ht_point_to_screen(model_points[i], rotation_matrix, translation_vector),
									      image_points[i]);

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

		if (cur_error.avg >= HT_RANSAC_MAX_CONSENSUS_ERROR)
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

			if (cur_error.avg > HT_RANSAC_MAX_CONSENSUS_ERROR)
				goto end2;

			if (pos >= min_consensus && pos > *best_cnt * ((1.0 - HT_RANSAC_BEST_ERROR_IMPORTANCE) + HT_RANSAC_BEST_ERROR_IMPORTANCE * cur_error.avg / best_error->avg)) {
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
	float error_scale = ctx.zoom_ratio;

	if (ht_ransac(ctx, HT_RANSAC_ITER, HT_RANSAC_MIN_FEATURES, HT_RANSAC_MAX_ERROR, (int) (ctx.feature_count * HT_RANSAC_MIN_CONSENSUS) + 1, best_cnt, best_error, best_indices, ctx.model, error_scale)) {
		char* usedp = new char[ctx.model.count];
		for (int i = 0; i < ctx.model.count; i++)
			usedp[i] = 0;
		for (int i = 0; i < *best_cnt; i++) {
			usedp[best_indices[i]] = 1;
		}
		for (int i = 0; i < ctx.model.count; i++) {
			if (!usedp[i]) {
				if (ctx.features[i].x != -1 && ctx.features[i].y != -1) {
					if (++ctx.feature_failed_iters[i] >= HT_FEATURE_MAX_FAILED_RANSAC) {
						ctx.features[i] = cvPoint2D32f(-1, -1);
						ctx.feature_count--;
					}
				}
			} else {
#if 0
				if (ctx.feature_failed_iters[i] != 0 && --ctx.feature_failed_iters[i] != 0) {
					for (int j = i; j < *best_cnt-1; j++)
						best_indices[j] = best_indices[j+1];
					(*best_cnt)--;
				}
#else
				ctx.feature_failed_iters[i] = 0;
#endif
			}
		}
		delete[] usedp;
		return *best_cnt >= HT_RANSAC_MIN_CONSENSUS * ctx.feature_count;
	}
	return false;
}