#pragma once

#ifdef HT_API
#   error "wrong include order"
#endif

#if defined(_WIN32) && !defined(__GNUC__)
#  define HT_API(t) __declspec(dllexport) t __stdcall
#else
#    if defined(_WIN32)
#        define HT_DECLSPEC __declspec(dllexport)
#    else
#        define HT_DECLSPEC
#    endif
# define HT_API(t) __attribute__ ((visibility ("default"))) HT_DECLSPEC t
#endif

#include "ht-api.h"

#include <string>
#include <vector>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>

#include "flandmark_detector.h"
#include "timer.hpp"

#include <memory>

using dist_coeffs = cv::Matx<float, 5, 1>;

struct triangle
{
    cv::Vec3f ps[3];
    triangle() : ps { cv::Vec3f(0,0,0), cv::Vec3f(0,0,0), cv::Vec3f(0,0,0) } {}
    triangle(const cv::Vec3f& p1, const cv::Vec3f& p2, const cv::Vec3f& p3) : ps { p1, p2, p3 } {}
};

struct triangle2d
{
    cv::Vec2f ps[3];
    triangle2d() : ps { cv::Vec2f(0,0), cv::Vec2f(0,0), cv::Vec2f(0,0) } {}
    triangle2d(const cv::Vec2f& p1, const cv::Vec2f& p2, const cv::Vec2f& p3) : ps { p1, p2, p3 } {}
};

struct projection
{
    triangle t;
    triangle2d projection;
};

struct model {
    std::vector<projection> triangles_and_projections;
    
    model(const std::string& filename);
    void draw(model& model, cv::Mat& color, float scale);
};

struct classifier
{
    cv::CascadeClassifier head_cascade;
    
    classifier() : head_cascade("haarcascade_frontalface_alt2.xml") {}
    bool classify(const cv::Mat& frame, cv::Rect& ret);
};

struct context
{
    enum state_t
    {
        STATE_INITIALIZING = 0, // waiting for RANSAC consensus
        STATE_TRACKING = 1, // ransac consensus established
        STATE_LOST = 2, // consensus lost; fall back to initializing
    };
    
    ht_config config;
    cv::VideoCapture camera;
    Timer timer_classify, timer_iter;
    model model, bbox;
    state_t state;
    float hz, iter_time;
    FLANDMARK_Model* flandmark_model;
    cv::Matx33f intrins;
    dist_coeffs dist;
    
    static constexpr double pi = 3.1415926535;
    
    context(const ht_config& conf);
    ht_result emit_result(const cv::Matx31d& rvec, const cv::Matx31d& tvec);
    bool estimate(cv::Mat& frame, const cv::Rect roi, cv::Matx31d& rvec_, cv::Matx31d& tvec_);
    bool get_image(cv::Mat& frame, cv::Mat& color);
    bool initial_guess(const cv::Rect rect_, cv::Mat& frame, cv::Matx31d& rvec_, cv::Matx31d& tvec_);
};

