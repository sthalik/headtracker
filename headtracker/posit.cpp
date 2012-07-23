#include "stdafx.h"

using namespace std;
using namespace cv;

bool ht_posit(CvPoint2D32f* image_points, CvPoint3D32f* model_points, int point_cnt, float* rotation_matrix, float* translation_vector, CvTermCriteria term_crit) {
	if (point_cnt < 4)
		return false;

	float dx = 0, dy = 0, dz = 0;

	for (int i = 1; i < point_cnt; i++) {
		dx += fabs(model_points[i].x);
		dy += fabs(model_points[i].y);
		dz += fabs(model_points[i].z);
	}

	if (dx <= 1.0e-2 || dy <= 1.0e-2 || dz <= 1.0e-2)
		return false;

	CvPOSITObject* posit_obj = cvCreatePOSITObject(model_points, point_cnt);
	cvPOSIT(posit_obj, image_points, HT_FOCAL_LENGTH, term_crit, rotation_matrix, translation_vector);
	cvReleasePOSITObject(&posit_obj);

	for (int i = 0; i < 9; i++)
		if (_isnan(rotation_matrix[i]))
			return false;

	for (int i = 0; i < 3; i++)
		if (_isnan(translation_vector[i]))
			return false;

	return true;
}

euler_t ht_matrix_to_euler(float* rotation_matrix, float* translation_vector) {
	euler_t ret;

	if (rotation_matrix[0 * 3 + 2] > 0.9998) {
		ret.rotx = (float) (HT_PI/2.0);
		ret.roty = (float) -atan2(rotation_matrix[1 * 3 + 0], rotation_matrix[1 * 3 + 1]);
		ret.rotz = 0.0f;
	} else if (rotation_matrix[0 * 3 + 2] < -0.9998) {
		ret.rotx = (float) (HT_PI/-2.0);
		ret.roty = (float) -atan2(rotation_matrix[1 * 3 + 0], rotation_matrix[1 * 3 + 1]);
		ret.rotz = 0.0f;
	} else {
		ret.rotx = (float) asin(rotation_matrix[0 * 3 + 2]);
		ret.roty = (float) -atan2(rotation_matrix[1 * 3 + 2], rotation_matrix[2 * 3 + 2]);
		ret.rotz = (float) atan2(-rotation_matrix[0 * 3 + 1], rotation_matrix[0 * 3 + 0]);
	}

	ret.tx = translation_vector[0];
	ret.ty = translation_vector[1];
	ret.tz = translation_vector[2];

	return ret;
}