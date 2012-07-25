#include "stdafx.h"

using namespace std;
using namespace cv;

error_t ht_avg_reprojection_error(headtracker_t& ctx, CvPoint3D32f* model_points, CvPoint2D32f* image_points, int point_cnt) {
	float rotation_matrix[9];
	float translation_vector[3];

	error_t ret;

	if (!ht_posit(image_points, model_points, point_cnt, rotation_matrix, translation_vector, cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.06))) {
		ret.avg = 1.0e10;
		return ret;
	}

	float max = 0;
	float avg = 0;

	for (int i = 0; i < point_cnt; i++) {
		CvPoint2D32f p = ht_point_to_screen(model_points[i], rotation_matrix, translation_vector);
		float dist = ht_distance2d_squared(p, image_points[i]);
		if (dist > max)
			max = dist;
		avg += sqrt(dist);
	}

	ret.avg = avg / point_cnt;

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
	int mcnt = model.count;
	int* indices = new int[mcnt];
	CvPoint2D32f* image_points = new CvPoint2D32f[mcnt];
	CvPoint3D32f* model_points = new CvPoint3D32f[mcnt];
	int k = 0;
	CvPoint3D32f* model_centers = new CvPoint3D32f[mcnt];
	int* model_indices = new int[mcnt];
	bool ret = false;

	best_error->avg = 1.0e10;
	*best_cnt = 0;

	for (int i = 0; i < mcnt; i++) {
		if (ctx.features[i].x != -1)
			indices[k++] = i;
		model_centers[i] = cvPoint3D32f(
			(model.triangles[i].p1.x + model.triangles[i].p2.x + model.triangles[i].p3.x) / 3,
			(model.triangles[i].p1.y + model.triangles[i].p2.y + model.triangles[i].p3.y) / 3,
			(model.triangles[i].p1.z + model.triangles[i].p2.z + model.triangles[i].p3.z) / 3);
	}

	if (k < min_consensus || k < 4 || iter_points < 4)
		goto end;

	for (int iter = 0; iter < max_iter; iter++) {
		ht_fisher_yates(indices, k);

		int pos = 0;

		CvPoint3D32f first_point = model_centers[indices[0]];

		for (int i = 0; i < iter_points; i++) {
			int idx = indices[i];

			model_points[pos] = model_centers[idx];
			model_points[pos].x -= first_point.x;
			model_points[pos].y -= first_point.y;
			model_points[pos].z -= first_point.z;
			image_points[pos] = ctx.features[idx];
			model_indices[pos] = idx;
			pos++;
		}

		error_t cur_error = ht_avg_reprojection_error(ctx, model_points, image_points, pos);
		cur_error.avg *= error_scale;

		if (cur_error.avg >= 1e9)
			continue;

		for (int i = iter_points; i < k; i++) {
			int idx = indices[i];
			model_points[pos] = model_centers[idx];
			model_points[pos].x -= first_point.x;
			model_points[pos].y -= first_point.y;
			model_points[pos].z -= first_point.z;
			image_points[pos] = ctx.features[idx];
			model_indices[pos] = idx;

			error_t e = ht_avg_reprojection_error(ctx, model_points, image_points, pos+1);
			e.avg *= error_scale;

			if (e.avg*max_error * (pos+1) > cur_error.avg * pos)
				continue;

			cur_error.avg = max(e.avg, cur_error.avg);
			pos++;

			if (e.avg <= HT_RANSAC_AVG_BEST_ERROR) {
				if (pos >= min_consensus && pos > *best_cnt) {
					ret = true;
					*best_error = e;
					*best_cnt = pos;
					for (int j = 0; j < pos; j++)
						best_indices[j] = model_indices[j];
					if (pos == k)
						goto end;
				}
			} else {
				goto end2;
			}
		}
end2:
		;
	}

end:

	delete[] indices;
	delete[] image_points;
	delete[] model_points;
	delete[] model_centers;
	delete[] model_indices;

	return ret;
}

bool ht_ransac_best_indices(headtracker_t& ctx, int* best_cnt, error_t* best_error, int* best_indices) {
	float error_scale;

	if (ctx.depth_frame_count == 0)
		error_scale = 1.0f;
	else {
		error_scale = 0.0f;
		for (int i = 0; i < ctx.depth_frame_count; i++)
			error_scale += ctx.depths[i];
		error_scale /= ctx.depth_frame_count;
	}

	if (ht_ransac(ctx, HT_RANSAC_ITER, HT_RANSAC_MIN_POINTS, HT_RANSAC_MAX_ERROR, HT_RANSAC_MIN_CONSENSUS, best_cnt, best_error, best_indices, ctx.tracking_model, error_scale)) {
		char* usedp = new char[ctx.tracking_model.count];
		for (int i = 0; i < ctx.tracking_model.count; i++)
			usedp[i] = 0;
		for (int i = 0; i < *best_cnt; i++) {
			usedp[best_indices[i]] = 1;
		}
		for (int i = 0; i < ctx.tracking_model.count; i++) {
			if (!usedp[i]) {
				if (ctx.features[i].x != -1 && ctx.features[i].y != -1) {
					if (++ctx.feature_failed_iters[i] > HT_FEATURE_MAX_FAILED_RANSAC) {
						ctx.features[i] = cvPoint2D32f(-1, -1);
						ctx.feature_count--;
					}
				}
			} else {
				if (ctx.feature_failed_iters[i] != 0 && --ctx.feature_failed_iters[i] != 0) {
					for (int j = i; j < *best_cnt-1; j++)
						best_indices[j] = best_indices[j+1];
					(*best_cnt)--;
				}
			}
		}
		delete[] usedp;
		return true;
	}
	return false;
}