#undef NDEBUG
#include <cassert>

#include "ht-internal.h"
#include <iostream>
#include <fstream>
#include <vector>

void model::draw(model& model, cv::Mat& color, float scale)
{
	const int sz = model.triangles_and_projections.size();

	for (int i = 0; i < sz; i++) {
		triangle2d& t = model.triangles_and_projections[i].projection;

        cv::line(color, cv::Point(t.ps[0][0] * scale, t.ps[0][1] * scale), cv::Point(t.ps[1][0] * scale, t.ps[1][1] * scale), cv::Scalar(255, 0, 0), 1);
        cv::line(color, cv::Point(t.ps[0][0] * scale, t.ps[0][1] * scale), cv::Point(t.ps[2][0] * scale, t.ps[2][1] * scale), cv::Scalar(255, 0, 0), 1);
        cv::line(color, cv::Point(t.ps[2][0] * scale, t.ps[2][1] * scale), cv::Point(t.ps[1][0] * scale, t.ps[1][1] * scale), cv::Scalar(255, 0, 0), 1);
	}
}

model::model(const std::string& filename)
{
	std::ifstream stream(filename.c_str(), std::ifstream::in);
	
	char line[256];
	line[255] = '\0';
	std::vector<triangle> triangles;
	triangle triangle;

	while (stream.good()) {
		stream.getline(line, 255);
		int ret = sscanf(line,
						 "%f%f%f%f%f%f%f%f%f",
						 &triangle.ps[0][0], &triangle.ps[0][1], &triangle.ps[0][2],
						 &triangle.ps[1][0], &triangle.ps[1][1], &triangle.ps[1][2],
						 &triangle.ps[2][0], &triangle.ps[2][1], &triangle.ps[2][2]);

		if (ret == EOF)
			break;

		if (ret != 9)
            assert(!"corrupt model file");

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
                triangle.ps[i][j] *= 100;
            triangle.ps[i][1] *= -1;
        }

		triangles.push_back(triangle);
	}

	const int sz = triangles.size();
    triangles_and_projections.resize(sz);
	for (int i = 0; i < sz; i++)
    {
        struct projection p;
        p.t = triangles[i];
		triangles_and_projections[i] = p;
	}
}

