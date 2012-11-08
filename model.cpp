#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;

void ht_project_model(headtracker_t& ctx,
                      const Mat& rvec,
                      const Mat& tvec,
                      model_t& model)
{
    int sz = model.count;

    if (!model.projection)
        model.projection = new triangle2d_t[sz];

    Mat dist_coeffs = Mat::zeros(5, 1, CV_32FC1);

    Mat intrinsics = Mat::eye(3, 3, CV_32FC1);
    intrinsics.at<float> (0, 0) = ctx.focal_length_w;
    intrinsics.at<float> (1, 1) = ctx.focal_length_h;
    intrinsics.at<float> (0, 2) = ctx.grayscale.cols/2;
    intrinsics.at<float> (1, 2) = ctx.grayscale.rows/2;

    vector<Point3f> triangles;
    triangles.resize(sz * 3);

    for (int i = 0; i < sz; i++) {
        const triangle_t& t = model.triangles[i];
        triangles[i * 3 + 0] = t.p1;
        triangles[i * 3 + 1] = t.p2;
        triangles[i * 3 + 2] = t.p3;
    }

    vector<Point2f> image_points;

    projectPoints(triangles, rvec, tvec, intrinsics, dist_coeffs, image_points);

    for (int i = 0; i < sz; i++) {
        triangle2d_t t2d;
        t2d.p1 = image_points[i * 3 + 0];
        t2d.p2 = image_points[i * 3 + 1];
        t2d.p3 = image_points[i * 3 + 2];

        model.projection[i] = t2d;
    }
}

bool ht_triangle_at(const CvPoint2D32f pos, triangle_t* ret, int* idx, const model_t& model, CvPoint2D32f& uv) {
	if (!model.projection)
		return false;

	int sz = model.count;

    bool foundp = false;

	for (int i = 0; i < sz; i++) {
		triangle2d_t& t = model.projection[i];
        if (ht_point_inside_triangle_2d(t.p1, t.p2, t.p3, pos, uv)) {
			*ret = model.triangles[i];
			*idx = i;
            if (foundp)
                return false;
            foundp = true;
		}
	}

    return foundp;
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
	for (int i = 0; i < sz; i++) {
		ret.triangles[i] = triangles[i];
	}

	ret.projection = NULL;

	return ret;
}

void ht_free_model(model_t& model) {
	delete model.triangles;
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
    if (fabs(denom) < 1.0e-6)
        return false;
	float invDenom = 1.0f / denom;
	float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
	float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    if (u > 0 && v > 0 && u + v <= 1) {
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
