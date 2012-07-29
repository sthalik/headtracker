#pragma once
#if defined(_WIN32) && !defined(MINGW)
#  define HT_API(t) __declspec(dllexport) t __cdecl
#else
#  define HT_API(t) t
#endif
#ifndef _WIN32
#  define _isnan isnan
#endif
#if defined(_WIN32) && !defined(MINGW)
#include "targetver.h"
#endif
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "common.h"
#include "internal.h"
