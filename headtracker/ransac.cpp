#include "stdafx.h"

using namespace std;
using namespace cv;

float ht_avg_reprojection_error(headtracker_t& ctx, CvPoint3D32f* model_points, CvPoint2D32f* image_points, int point_cnt) {
	float rotation_matrix[9];
	float translation_vector[3];

	if (!ht_posit(image_points, model_points, point_cnt, rotation_matrix, translation_vector, cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 40, 0.008)))
		return 1e10;

	double avg = 0;
	int cnt = 0;

	for (int i = 0; i < point_cnt; i++) {
		CvPoint2D32f p = ht_point_to_screen(model_points[i], rotation_matrix, translation_vector);
		float dist = ht_distance2d_squared(p, image_points[i]);
		avg += dist;
		cnt++;
	}

	if (cnt == 0)
		return 0;
	
	avg /= cnt;

	if (avg == 0)
		return 0;

	return sqrt(avg);
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
			   float* best_error,
			   int* best_indices,
			   model_t& model)
{
	int mcnt = model.count;
	int* indices = new int[mcnt];
	CvPoint2D32f* image_points = new CvPoint2D32f[mcnt];
	CvPoint3D32f* model_points = new CvPoint3D32f[mcnt];
	int k = 0;
	CvPoint3D32f* model_centers = new CvPoint3D32f[mcnt];
	int* model_indices = new int[mcnt];
	bool ret = false;

	*best_error = 1.0e9;
	*best_cnt = 0;

	for (int i = 0; i < mcnt; i++) {
		if (!(ctx.features[i].x == -1 || ctx.features[i].y == -1))
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

		float cur_error = ht_avg_reprojection_error(ctx, model_points, image_points, pos);

		if (cur_error >= 1e9)
			continue;

		for (int i = iter_points; i < k; i++) {
			int idx = indices[i];
			model_points[pos] = model_centers[idx];
			model_points[pos].x -= first_point.x;
			model_points[pos].y -= first_point.y;
			model_points[pos].z -= first_point.z;
			image_points[pos] = ctx.features[idx];
			model_indices[pos] = idx;

			float e = ht_avg_reprojection_error(ctx, model_points, image_points, pos+1);

			if (e*max_error > cur_error)
				continue;

			pos++;

			cur_error = max(cur_error, e);

			if (pos > *best_cnt && pos >= min_consensus && e <= HT_RANSAC_MAX_BEST_ERROR) {
				ret = true;
				*best_error = e;
				*best_cnt = pos;
				for (int j = 0; j < pos; j++)
					best_indices[j] = model_indices[j];
				if (pos == k)
					break;
			}
		}
	}

end:

	delete[] indices;
	delete[] image_points;
	delete[] model_points;
	delete[] model_centers;
	delete[] model_indices;

	return ret;
}

bool ht_ransac_best_indices(headtracker_t& ctx, int* best_cnt, float* best_error, int* best_indices) {
	if (ht_ransac(ctx, HT_RANSAC_ITER, HT_RANSAC_MIN_POINTS, HT_RANSAC_MAX_ERROR, HT_RANSAC_MIN_CONSENSUS, best_cnt, best_error, best_indices, ctx.tracking_model)) {
		char* usedp = new char[ctx.tracking_model.count];
		for (int i = 0; i < ctx.tracking_model.count; i++)
			usedp[i] = 0;
		for (int i = 0; i < *best_cnt; i++) {
			usedp[best_indices[i]] = 1;
		}
		for (int i = 0; i < ctx.tracking_model.count; i++) {
			if (!usedp[i] && ctx.features[i].x != -1 && ctx.features[i].y != -1) {
				ctx.features[i] = cvPoint2D32f(-1, -1);
				ctx.feature_count--;
			}
		}
		delete[] usedp;
		return true;
	}
	return false;
}