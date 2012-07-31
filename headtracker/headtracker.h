#if defined __cplusplus
#  define HT_EXTERN extern "C"
#else
#  define HT_EXTERN 
#endif
#if defined(_WIN32) && !defined(MINGW)
#  define HT_API(t) HT_EXTERN __declspec(dllimport) t __cdecl
#else
#  define HT_API(t) HT_EXTERN t
#endif
#include "common.h"