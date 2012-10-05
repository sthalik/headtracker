#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;

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
		t2d.p1 = ht_project_point(cvPoint3D32f(t.p1.x - origin.x, t.p1.y - origin.y, t.p1.z - origin.z), rotation_matrix, translation_vector, ctx.focal_length);
		t2d.p2 = ht_project_point(cvPoint3D32f(t.p2.x - origin.x, t.p2.y - origin.y, t.p2.z - origin.z), rotation_matrix, translation_vector, ctx.focal_length);
		t2d.p3 = ht_project_point(cvPoint3D32f(t.p3.x - origin.x, t.p3.y - origin.y, t.p3.z - origin.z), rotation_matrix, translation_vector, ctx.focal_length);

		model.projection[i] = t2d;
		model.projected_depths[i] = model.centers[i].x * rotation_matrix[6] + model.centers[i].y * rotation_matrix[7] + model.centers[i].z * rotation_matrix[8];
    }
}

bool ht_triangle_at(const CvPoint2D32f pos, triangle_t* ret, int* idx, const model_t& model, CvPoint2D32f& uv) {
	if (!model.projection)
		return false;

	float min_depth = -1e12f;
	int sz = model.count;

	for (int i = 0; i < sz; i++) {
		triangle2d_t& t = model.projection[i];
		if (model.projected_depths[i] > min_depth && ht_point_inside_triangle_2d(t.p1, t.p2, t.p3, pos, uv)) {
			*ret = model.triangles[i];
			min_depth = model.projected_depths[i];
			*idx = i;
		}
	}

	return min_depth > -9999;
}

bool ht_triangle_exists(CvPoint2D32f pos, const model_t& model) {
	if (!model.projection)
		return false;

	int sz = model.count;
    CvPoint2D32f uv;

	for (int i = 0; i < sz; i++) {
		triangle2d_t& t = model.projection[i];
		if (ht_point_inside_triangle_2d(t.p1, t.p2, t.p3, pos, uv))
			return true;
	}

	return false;
}


void ht_draw_model(headtracker_t& ctx, model_t& model) {
	int sz = model.count;

	for (int i = 0; i < sz; i++) {
		triangle2d_t& t = model.projection[i];

        line(ctx.color, cvPoint(t.p1.x, t.p1.y), cvPoint(t.p2.x, t.p2.y), Scalar(255, 0, 0));
        line(ctx.color, cvPoint(t.p1.x, t.p1.y), cvPoint(t.p3.x, t.p3.y), Scalar(255, 0, 0));
        line(ctx.color, cvPoint(t.p3.x, t.p3.y), cvPoint(t.p2.x, t.p2.y), Scalar(255, 0, 0));
	}
}

model_t ht_load_model(const char* filename, CvPoint3D32f scale, CvPoint3D32f offset) {
	ifstream stream(filename, ifstream::in);
	
	char line[256];
	line[255] = '\0';
	vector<triangle_t> triangles;
	triangle_t triangle;

	while (stream.good()) {
		stream.getline(line, 254);
		int ret = sscanf(line,
						 "%f%f%f%f%f%f%f%f%f",
						 &triangle.p1.x, &triangle.p1.y, &triangle.p1.z,
						 &triangle.p2.x, &triangle.p2.y, &triangle.p2.z,
						 &triangle.p3.x, &triangle.p3.y, &triangle.p3.z);

		if (ret == EOF)
			break;

		if (ret != 9)
			throw new exception();

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

	model_t ret;

	int sz = triangles.size();
	ret.count = sz;
	ret.triangles = new triangle_t[sz];
	ret.centers = new CvPoint3D32f[sz];
	ret.projected_depths = new float[sz];
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
	delete model.projected_depths;
}

bool ht_point_inside_triangle_2d(const CvPoint2D32f a, const CvPoint2D32f b, const CvPoint2D32f c, const CvPoint2D32f point, CvPoint2D32f& uv) {
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
	float invDenom = 1.0f / denom;
	float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
	float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

	if (u > 0 && v > 0 && u + v < 1) {
		uv.x = u;
		uv.y = v;
		return true;
	}

	return false;
}

bool ht_point_inside_rectangle(CvPoint2D32f p, CvPoint2D32f topLeft, CvPoint2D32f bottomRight) {
	return p.x >= topLeft.x && p.x <= bottomRight.x && p.y >= topLeft.y && p.y <= bottomRight.y;
}

CvPoint3D32f ht_get_triangle_pos(const CvPoint2D32f uv, const triangle_t& t) {
	float u = uv.x;
	float v = uv.y;
	CvPoint3D32f ret;

	ret.x = t.p1.x + u * (t.p3.x - t.p1.x) + v * (t.p2.x - t.p1.x);
	ret.y = t.p1.y + u * (t.p3.y - t.p1.y) + v * (t.p2.y - t.p1.y);
	ret.z = t.p1.z + u * (t.p3.z - t.p1.z) + v * (t.p2.z - t.p1.z);

	return ret;
}
