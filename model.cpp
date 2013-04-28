#include "ht-api.h"
#include "ht-internal.h"
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;

static Point3f ht_rotate_point(const Mat& rmat, const Point3f p)
{
    Point3f ret;
    ret.x = p.x * rmat.at<double>(0, 0) + p.y * rmat.at<double>(1, 0) + p.z * rmat.at<double>(2, 0);
    ret.y = p.x * rmat.at<double>(0, 1) + p.y * rmat.at<double>(1, 1) + p.z * rmat.at<double>(2, 1);
    ret.z = p.x * rmat.at<double>(0, 2) + p.y * rmat.at<double>(1, 2) + p.z * rmat.at<double>(2, 2);
    return ret;
}

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

    if (!model.rotation)
        model.rotation = new triangle_t[sz];

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

        triangle_t t3d = model.triangles[i];
        triangle_t r;

        r.p1 = ht_rotate_point(rmat, t3d.p1);
        r.p2 = ht_rotate_point(rmat, t3d.p2);
        r.p3 = ht_rotate_point(rmat, t3d.p3);

        model.rotation[i] = r;
    }
    
    return true;
}

bool ht_triangle_at(const Point2f pos, triangle_t* ret, int* idx, const model_t& model, Point2f& uv) {
	if (!model.projection)
		return false;
    if (!model.rotation)
        return false;

	int sz = model.count;

    bool foundp = false;

    float best_z = -1e10;
    Point2f best_uv;

	for (int i = 0; i < sz; i++) {
		triangle2d_t& t = model.projection[i];
        if (ht_point_inside_triangle_2d(t.p1, t.p2, t.p3, pos, uv))
        {
            float new_z = ht_get_triangle_pos(uv, model.rotation[i]).z;
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

	float mult = ctx.color.cols > 320 ? 2 : 1;

	for (int i = 0; i < sz; i++) {
		triangle2d_t& t = model.projection[i];

        line(ctx.color, mult * Point(t.p1.x, t.p1.y), mult * Point(t.p2.x, t.p2.y), Scalar(255, 0, 0), 1);
        line(ctx.color, mult * Point(t.p1.x, t.p1.y), mult * Point(t.p3.x, t.p3.y), Scalar(255, 0, 0), 1);
        line(ctx.color, mult * Point(t.p3.x, t.p3.y), mult * Point(t.p2.x, t.p2.y), Scalar(255, 0, 0), 1);
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

#if 0
        triangle.p1.y *= -1;
        triangle.p2.y *= -1;
        triangle.p3.y *= -1;
#endif

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
    ret.rotation = NULL;

	return ret;
}

bool ht_point_inside_triangle_2d(const Point2f a, const Point2f b, const Point2f c, const Point2f point, Point2f& uv) {
    double x0 = a.x;
    double y0 = a.y;
    double x1 = b.x;
    double y1 = b.y;
    double x2 = c.x;
    double y2 = c.y;
    double xp = point.x;
    double yp = point.y;

    double det = x0*(y1 - y2) + x1*(y2 - y0) + x2*(y0 - y1);

    double v = ((x2*y0-x0*y2)+xp*(y2-y0)+yp*(x0-x2)) / det;
    double u = ((x0*y1-x1*y0)+xp*(y0-y1)+yp*(x1-x0)) / det;

    if (u >= 0 && v >= 0 && u + v <= 1) {
		uv.x = u;
		uv.y = v;
		return true;
	}

	return false;
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
