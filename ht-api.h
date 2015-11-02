#pragma once
#ifndef HT_API
#   if defined(_WIN32) && !defined(__MINGW32__)
#     define HT_API(t) __declspec(dllexport) t __stdcall
#   else
#       if defined(_WIN32)
#           define HT_DECLSPEC __declspec(dllexport)
#       else
#           define HT_DECLSPEC
#       endif
#    define HT_API(t) __attribute__ ((visibility ("default"))) HT_DECLSPEC t
#   endif
#endif

#if defined(_MSC_VER)
#  define isnan _isnan
#endif

struct ht_config
{
    float field_of_view;
    float classification_delay;
    int   force_width;
    int   force_height;
    int   force_fps;
    int   camera_index;
    bool  debug;
    double dist_coeffs[5];
    
    ht_config()
        : field_of_view(69), classification_delay(2000), force_width(0), force_height(0),
          camera_index(0), debug(true), dist_coeffs { 0, 0, 0, 0, 0 }
    {}
};

struct ht_result
{
    double rotx, roty, rotz;
    double tx, ty, tz;
    bool filled;
    
    ht_result() : rotx(0), roty(0), rotz(0), tx(0), ty(0), tz(0), filled(false) {}
};
