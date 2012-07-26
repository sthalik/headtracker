#include "stdafx.h"
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

CvPoint2D32f ht_point_to_screen(CvPoint3D32f p, float* rotation_matrix, float* translation_vector) {
	return ht_project_point(p, rotation_matrix, translation_vector);
}

void ht_project_model(headtracker_t& ctx,
					  float* rotation_matrix,
					  float* translation_vector,
					  model_t& model,
					  CvPoint3D32f origin)
{
	int sz = model.count;

	if (!model.projection)
		model.projection = new triangle2d_t[sz];

	for (int i = 0; i < sz; i++) {
		triangle_t& t = model.triangles[i];
		triangle2d_t t2d;
		t2d.p1 = ht_point_to_screen(cvPoint3D32f(t.p1.x + origin.x, t.p1.y + origin.y, t.p1.z + origin.z), rotation_matrix, translation_vector);
		t2d.p2 = ht_point_to_screen(cvPoint3D32f(t.p2.x + origin.x, t.p2.y + origin.y, t.p2.z + origin.z), rotation_matrix, translation_vector);
		t2d.p3 = ht_point_to_screen(cvPoint3D32f(t.p3.x + origin.x, t.p3.y + origin.y, t.p3.z + origin.z), rotation_matrix, translation_vector);

		model.projection[i] = t2d;
	}
}

bool ht_triangle_at(headtracker_t& ctx, CvPoint2D32f pos, triangle_t* ret, int* idx, float* rotation_matrix, float* translation_vector, model_t& model) {
	if (!model.projection)
		return false;

	float min_depth = -1e12;
	int sz = model.count;

	for (int i = 0; i < sz; i++) {
		float depth = model.centers[i].x * rotation_matrix[6] + model.centers[i].y * rotation_matrix[7] + model.centers[i].z * rotation_matrix[8];
		triangle2d_t& t = model.projection[i];
		if (depth > min_depth && ht_point_inside_triangle_2d(t.p1, t.p2, t.p3, pos)) {
			*ret = model.triangles[i];
			min_depth = depth;
			*idx = i;
		}
	}

	return min_depth > -9999;
}


void ht_draw_model(headtracker_t& ctx, float* rotation_matrix, float* translation_vector, model_t& model) {
	int sz = model.count;

	for (int i = 0; i < sz; i++) {
		triangle2d_t& t = model.projection[i];

		cvLine(ctx.color, cvPoint(t.p1.x, t.p1.y), cvPoint(t.p2.x, t.p2.y), CV_RGB(0, 0, 255));
		cvLine(ctx.color, cvPoint(t.p1.x, t.p1.y), cvPoint(t.p3.x, t.p3.y), CV_RGB(0, 0, 255));
		cvLine(ctx.color, cvPoint(t.p3.x, t.p3.y), cvPoint(t.p2.x, t.p2.y), CV_RGB(0, 0, 255));
	}
}

model_t ht_load_model(const char* filename, CvPoint3D64f scale, CvPoint3D64f offset) {
	FILE* stream = fopen(filename, "r");
	if (stream == NULL)
		throw exception("can't open model");
	char line[256];
	line[255] = '\0';
	vector<triangle_t> triangles;

	while (fgets(line, 255, stream) != NULL) {
		triangle_t triangle;

		int ret = sscanf(line,
						 "%f%f%f%f%f%f%f%f%f",
						 &triangle.p1.x, &triangle.p1.y, &triangle.p1.z,
						 &triangle.p2.x, &triangle.p2.y, &triangle.p2.z,
						 &triangle.p3.x, &triangle.p3.y, &triangle.p3.z);

		if (ret == EOF)
			break;

		if (ret != 9)
			throw new exception("parse error in model");

		triangle.p1.x += offset.x;
		triangle.p1.y += offset.y;
		triangle.p1.z += offset.z;

		triangle.p2.x += offset.x;
		triangle.p2.y += offset.y;
		triangle.p2.z += offset.z;

		triangle.p3.x += offset.x;
		triangle.p3.y += offset.y;
		triangle.p3.z += offset.z;

		triangle.p1.x *= scale.x;
		triangle.p1.y *= -scale.y;
		triangle.p1.z *= scale.z;

		triangle.p2.x *= scale.x;
		triangle.p2.y *= -scale.y;
		triangle.p2.z *= scale.z;

		triangle.p3.x *= scale.x;
		triangle.p3.y *= -scale.y;
		triangle.p3.z *= scale.z;

		triangles.push_back(triangle);

	}

	fclose(stream);

	model_t ret;

	int sz = triangles.size();
	ret.count = sz;
	ret.triangles = new triangle_t[sz];
	ret.centers = new CvPoint3D32f[sz];
	for (int i = 0; i < sz; i++) {
		ret.triangles[i] = triangles[i];
		ret.centers[i] = cvPoint3D32f(
			(triangles[i].p1.x + triangles[i].p2.x + triangles[i].p3.x) / 3,
			(triangles[i].p1.y + triangles[i].p2.y + triangles[i].p3.y) / 3,
			(triangles[i].p1.z + triangles[i].p2.z + triangles[i].p3.z) / 3);
	}

	ret.projection = NULL;

	return ret;
}

void ht_free_model(model_t& model) {
	delete model.triangles;
}

CvPoint2D32f ht_project_point(CvPoint3D32f point, float* rotation_matrix, float* translation_vector) {
	double x = point.x * rotation_matrix[0] + point.y * rotation_matrix[1] + point.z * rotation_matrix[2] + translation_vector[0];
	double y = point.x * rotation_matrix[3] + point.y * rotation_matrix[4] + point.z * rotation_matrix[5] + translation_vector[1];
	double z = point.x * rotation_matrix[6] + point.y * rotation_matrix[7] + point.z * rotation_matrix[8] + translation_vector[2];

#if 0
	CvPoint3D32f p3d;
	euler_t angles = ht_matrix_to_euler(rotation_matrix, translation_vector);

	double ox = angles.rotx;
	double oy = angles.roty;
	double oz = -angles.rotz;

	p3d.x = cos(oy)*(sin(oz)*y+cos(oz)*x)-sin(oy)*z;
	p3d.y = sin(ox)*(cos(oy)*z+sin(oy)*(sin(oz)*y+cos(oz)*x))+cos(ox)*(cos(oz)*y-sin(oz)*x);
	p3d.z = cos(ox)*(cos(oy)*z+sin(oy)*(sin(oz)*y+cos(oz)*x))-sin(ox)*(cos(oz)*y-sin(oz)*x);
#endif

	double bx = x * HT_FOCAL_LENGTH / z;
	double by = y * HT_FOCAL_LENGTH / z;
	
	return cvPoint2D32f(bx, by);
}

bool ht_point_inside_triangle_2d(CvPoint2D32f a, CvPoint2D32f b, CvPoint2D32f c, CvPoint2D32f point) {
	CvPoint2D32f v0 = cvPoint2D32f(
		c.x - a.x,
		c.y - a.y);
	CvPoint2D32f v1 = cvPoint2D32f(
		b.x - a.x,
		b.y - a.y);
	CvPoint2D32f v2 = cvPoint2D32f(
		point.x - a.x,
		point.y - a.y);
	float dot00 = ht_dot_product2d(v0, v0);
	float dot01 = ht_dot_product2d(v0, v1);
	float dot02 = ht_dot_product2d(v0, v2);
	float dot11 = ht_dot_product2d(v1, v1);
	float dot12 = ht_dot_product2d(v1, v2);
	float denom = dot00 * dot11 - dot01 * dot01;
	if (denom == 0)
		return false;
	float invDenom = 1 / denom;
	float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
	float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

	return u >= 0 && v >= 0 && u + v <= 1;
}

bool ht_point_inside_rectangle(CvPoint2D32f p, CvPoint2D32f topLeft, CvPoint2D32f bottomRight) {
	return p.x >= topLeft.x && p.x <= bottomRight.x && p.y >= topLeft.y && p.y <= bottomRight.y;
}