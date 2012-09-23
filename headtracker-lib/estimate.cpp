#include "stdafx.h"

using namespace std;
using namespace cv;

bool ht_estimate_pose(headtracker_t& ctx,
					  float* rotation_matrix,
					  float* translation_vector,
					  float* rotation_matrix2,
					  float* translation_vector2,
					  CvPoint3D32f* offset,
					  CvPoint2D32f* image_centroid)
{
    int total = ctx.keypoint_count;
	CvPoint3D32f* model_points = new CvPoint3D32f[total+1];
	CvPoint2D32f* image_points = new CvPoint2D32f[total+1];
	CvPoint3D32f* tmp_model_points = new CvPoint3D32f[total+1];
	CvPoint2D32f* tmp_image_points = new CvPoint2D32f[total+1];
	int k = 0;
	bool ret = false;

	for (int i = 0; i < ctx.config.max_keypoints; i++) {
        if (ctx.keypoints[i].idx == -1)
			continue;
		model_points[k] = ctx.keypoint_uv[i];
		image_points[k] = ctx.keypoints[i].position;
		k++;
	}

	if (k >= 4) {
		int centermost = -1;
		float center_distance = 1e10;

		for (int i = 0; i < k; i++) {
			float d = ht_distance3d_squared(model_points[i], cvPoint3D32f(0, 24.21354, 25.75259));

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

			ret = ht_posit(tmp_image_points,
						   tmp_model_points,
						   k,
						   rotation_matrix,
						   translation_vector,
                           cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 1000, 1.0e-8),
						   ctx.focal_length);

			if (ret) {
				*offset = c;
				tmp_model_points[0] = cvPoint3D32f(0, 0, 0);
				tmp_image_points[0] = *image_centroid = ht_project_point(cvPoint3D32f(-c.x, -c.y - HT_CENTROID_Y, -c.z - HT_CENTROID_DEPTH),
																		 rotation_matrix,
																		 translation_vector,
																		 ctx.focal_length);
				for (int i = 0; i < k; i++) {
					tmp_model_points[i+1] = cvPoint3D32f(model_points[i].x, model_points[i].y + HT_CENTROID_Y, model_points[i].z + HT_CENTROID_DEPTH);
					tmp_image_points[i+1] = image_points[i];
				}
				ret = ht_posit(tmp_image_points,
							   tmp_model_points,
							   k+1,
							   rotation_matrix2,
							   translation_vector2,
                               cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 1000, 1.0e-8),
							   ctx.focal_length);
				if (ret)
					ht_update_zoom_scale(ctx, translation_vector2[2]);
			}
		}
    }

	delete[] model_points;
	delete[] image_points;
	delete[] tmp_model_points;
	delete[] tmp_image_points;

	return ret;
}

void ht_update_zoom_scale(headtracker_t& ctx, float translation_2) {
	int sz = ctx.config.depth_avg_frames;
    ctx.depths[ctx.depth_counter_pos] = translation_2;
	ctx.depth_counter_pos = (ctx.depth_counter_pos + 1) % sz;
	if (ctx.depth_frame_count < sz)
		ctx.depth_frame_count++;
	float zoom_scale = 0.0f;
	for (int i = 0; i < ctx.depth_frame_count; i++)
		zoom_scale += ctx.depths[i];
    zoom_scale /= ctx.depth_frame_count;
    ctx.zoom_ratio = zoom_scale / HT_STD_DEPTH;
}
