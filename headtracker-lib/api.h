#ifndef HT_API
#   if defined(_WIN32) && !defined(MINGW)
#     define HT_API(t) extern "C" __declspec(dllexport) t __cdecl
#   else
#    define HT_API(t) extern "C" t
#   endif
#endif
#if !defined(_WIN32) && !defined(_isnan)
#  define _isnan isnan
#endif
#if defined(_WIN32) && !defined(MINGW)
#include "targetver.h"
#endif
