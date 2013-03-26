project(headtracker)
cmake_minimum_required(VERSION 2.8)

FIND_PACKAGE( OpenCV REQUIRED )

if(OpenCV_FOUND)
	include_directories("${OpenCV_DIR}/include")
	include_directories("${OpenCV_CONFIG_PATH}/include")
endif()

configure_file(${CMAKE_SOURCE_DIR}/data/head.raw ${CMAKE_BINARY_DIR} COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/data/haarcascade_frontalface_alt2.xml ${CMAKE_BINARY_DIR} COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/data/flandmark_model.dat ${CMAKE_BINARY_DIR} COPYONLY)

SET(FLANDMARK_STATIC TRUE CACHE BOOL "Whether flandmark static library is used")
SET(LIBHEADTRACKER_STATIC TRUE CACHE BOOL "Whether flandmark static library is used")

if(LIBHEADTRACKER_STATIC)
	SET(HT_TYPE STATIC)
else()
	SET(HT_TYPE DYNAMIC)
endif()

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})

file(GLOB headtracker-SRC
    "*.h"
    "*.cpp"
)
add_library(${PROJECT_NAME} ${HT_TYPE} ${headtracker-SRC})

if(WIN32)
    set(HT_SONAME dll)
    set(HT_OBJNAME lib)
    set(HT_PREFIX "")
endif(WIN32)

if(MINGW)
    set(HT_OBJNAME a)
    set(HT_PREFIX lib)
endif()

if(UNIX)
    set(HT_SONAME so)
    set(HT_OBJNAME a)
    set(HT_PREFIX lib)
endif(UNIX)

if(APPLE)
    set(HT_SONAME dylib)
    set(HT_OBJNAME a)
    set(HT_PREFIX lib)
endif(APPLE)

if(FLANDMARK_STATIC)
	target_link_libraries(headtracker ${OpenCV_LIBS} ${CMAKE_BINARY_DIR}/${HT_PREFIX}flandmark_static.${HT_OBJNAME})
else()
	target_link_libraries(headtracker ${OpenCV_LIBS} ${CMAKE_BINARY_DIR}/${HT_PREFIX}flandmark_shared.${HT_SONAME})
endif()
file(GLOB headtracker-demo-SRC
    "demo/*.cpp"
)

#add_executable(headtracker-demo ${headtracker-demo-SRC})
#target_link_libraries(headtracker-demo headtracker ${OpenCV_LIBS})
file(GLOB headtracker-ftnoir-SRC
	"ftnoir/*.cpp"
)

add_executable(headtracker-ftnoir ${headtracker-ftnoir-SRC})
target_link_libraries(headtracker-ftnoir headtracker)
if(NOT WIN32)
	target_link_libraries(headtracker-ftnoir rt)
endif()
