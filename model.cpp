#include "ht-api.h"
#include "ht-internal.h"
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;

bool ht_project_model(headtracker_t& ctx,
                      const Mat& rvec,
                      const Mat& tvec,
                      model_t& model)
{
    if (tvec.rows * tvec.cols != 3 || rvec.rows * rvec.cols != 3)
        return false;
    
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

    for (int i = 0; i < sz; i++) {
        const triangle_t& t = model.triangles[i];
        triangles.push_back(t.p1);
        triangles.push_back(t.p2);
        triangles.push_back(t.p3);
    }

    vector<Point2f> image_points;

    projectPoints(triangles, rvec, tvec, intrinsics, dist_coeffs, image_points);

    Mat rmat = Mat::zeros(3, 3, CV_64FC1);

    Rodrigues(rvec, rmat);

    for (int i = 0; i < sz; i++) {
        triangle2d_t t2d;
        t2d.p1 = image_points[i * 3 + 0];
        t2d.p2 = image_points[i * 3 + 1];
        t2d.p3 = image_points[i * 3 + 2];

        model.projection[i] = t2d;
    }
    
    return true;
}

bool ht_triangle_at(const Point2f pos, triangle_t* ret, int* idx, const model_t& model, Point2f& uv) {
	if (!model.projection)
		return false;

	int sz = model.count;

    bool foundp = false;

    float best_z = -1e10;
    Point2f best_uv;

	for (int i = 0; i < sz; i++) {
		triangle2d_t& t = model.projection[i];
        if (ht_point_inside_triangle_2d(t.p1, t.p2, t.p3, pos, uv))
        {
            Point3f tmp = ht_get_triangle_pos(uv, model.triangles[i]);
            float new_z = tmp.z;
            if (new_z > best_z)
            {
                best_uv = uv;
                *ret = model.triangles[i];
                *idx = i;
                best_z = new_z;
                foundp = true;
            }
		}
	}

    if (foundp)
        uv = best_uv;
    return foundp;
}

void ht_draw_model(headtracker_t& ctx, model_t& model) {
	int sz = model.count;

	float mult = ctx.color.cols / (float)ctx.grayscale.cols;

	for (int i = 0; i < sz; i++) {
		triangle2d_t& t = model.projection[i];

        line(ctx.color, mult * Point(t.p1.x, t.p1.y), mult * Point(t.p2.x, t.p2.y), Scalar(255, 0, 0), 2);
        line(ctx.color, mult * Point(t.p1.x, t.p1.y), mult * Point(t.p3.x, t.p3.y), Scalar(255, 0, 0), 2);
        line(ctx.color, mult * Point(t.p3.x, t.p3.y), mult * Point(t.p2.x, t.p2.y), Scalar(255, 0, 0), 2);
	}
}

model_t ht_load_model(const char* filename) {
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

        triangle.p1.x *= 100;
        triangle.p1.y *= 100;
        triangle.p1.z *= 100;
        triangle.p2.x *= 100;
        triangle.p2.y *= 100;
        triangle.p2.z *= 100;
        triangle.p3.x *= 100;
        triangle.p3.y *= 100;
        triangle.p3.z *= 100;

        triangle.p1.y *= -1;
        triangle.p2.y *= -1;
        triangle.p3.y *= -1;

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

static __inline double dot(const Point2d& p1, const Point2d& p2) {
    return p1.x * p2.x + p1.y * p2.y;
}

bool ht_point_inside_triangle_2d(const Point2d p1, const Point2d p2, const Point2d p3, const Point2d px, Point2f& uv) {
    Point2d v0(p3.x - p1.x, p3.y - p1.y);
    Point2d v1(p2.x - p1.x, p2.y - p1.y);
    Point2d v2(px.x - p1.x, px.y - p1.y);

    double dot00 = dot(v0, v0);
    double dot01 = dot(v0, v1);
    double dot02 = dot(v0, v2);
    double dot11 = dot(v1, v1);
    double dot12 = dot(v1, v2);

    double invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
    double u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    double v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    uv.x = u;
    uv.y = v;

    return (u >= 0) && (v >= 0) && (u + v <= 1);
}

Point3f ht_get_triangle_pos(const Point2f uv, const triangle_t& t) {
	float u = uv.x;
	float v = uv.y;
    Point3f ret;

    ret.x = t.p1.x + u * (t.p3.x - t.p1.x) + v * (t.p2.x - t.p1.x);
    ret.y = t.p1.y + u * (t.p3.y - t.p1.y) + v * (t.p2.y - t.p1.y);
    ret.z = t.p1.z + u * (t.p3.z - t.p1.z) + v * (t.p2.z - t.p1.z);

	return ret;
}
