#include "stdafx.h"

using namespace std;
using namespace cv;

bool ht_estimate_pose(headtracker_t& ctx, float* rotation_matrix, float* translation_vector, int* indices, int count, CvPoint3D32f* offset) {
	CvPoint3D32f* model_points = new CvPoint3D32f[count];
	CvPoint2D32f* image_points = new CvPoint2D32f[count];
	CvPoint3D32f* tmp_model_points = new CvPoint3D32f[count];
	CvPoint2D32f* tmp_image_points = new CvPoint2D32f[count];
	int k = 0;
	bool ret = false;

	for (int i = 0; i < count; i++) {
		int idx = indices[i];
		model_points[k] = cvPoint3D32f(
			(ctx.tracking_model.triangles[idx].p1.x + ctx.tracking_model.triangles[idx].p2.x + ctx.tracking_model.triangles[idx].p3.x) / 3,
			(ctx.tracking_model.triangles[idx].p1.y + ctx.tracking_model.triangles[idx].p2.y + ctx.tracking_model.triangles[idx].p3.y) / 3,
			(ctx.tracking_model.triangles[idx].p1.z + ctx.tracking_model.triangles[idx].p2.z + ctx.tracking_model.triangles[idx].p3.z) / 3);
		image_points[k] = ctx.features[idx];
		k++;
	}

	if (k != 0) {
		int centermost = -1;
		float center_distance = 1e10;

		for (int i = 0; i < k; i++) {
			float d = ht_distance3d_squared(model_points[i], cvPoint3D32f(0, 0, 0));

			if (center_distance > d) {
				center_distance = d;
				centermost = i;
			}
		}

		if (centermost >= -1) {
			CvPoint3D32f c = model_points[centermost];

			tmp_model_points[0] = cvPoint3D32f(0, 0, 0);
			tmp_image_points[0] = image_points[centermost];

			for (int i = 0; i < centermost; i++) {
				tmp_model_points[i+1] = cvPoint3D32f(model_points[i].x - c.x, model_points[i].y - c.y, model_points[i].z - c.z);
				tmp_image_points[i+1] = image_points[i];
			}

			for (int i = centermost+1; i < k; i++) {
				tmp_model_points[i] = cvPoint3D32f(model_points[i].x - c.x, model_points[i].y - c.y, model_points[i].z - c.z);
				tmp_image_points[i] = image_points[i];
			}

			ret = ht_posit(tmp_image_points, tmp_model_points, k, rotation_matrix, translation_vector, cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 200, 0.1 * HT_PI / 180.0));

			if (ret) {
				*offset = c;
			}
		}
	}
	delete[] model_points;
	delete[] image_points;
	delete[] tmp_model_points;
	delete[] tmp_image_points;

	return ret;
}