#if defined(_WIN32) && !defined(MINGW)
#  define HT_API(t) __declspec(dllimport) t __cdecl
#else
#  define HT_API(t) t
#endif
#include "common.h"