#ifndef HT_API
#ifndef __cplusplus
# define HT_EXTERN 
#else
# define HT_EXTERN extern "C" 
#endif
#   if defined(_WIN32) && !defined(MINGW)
#     define HT_API(t) HT_EXTERN __declspec(dllexport) t __stdcall
#   else
#    define HT_API(t) HT_EXTERN t
#   endif
#endif
#if !defined(_WIN32) && !defined(_isnan)
#  define _isnan isnan
#endif
