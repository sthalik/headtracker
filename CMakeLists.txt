project(headtracker)
cmake_minimum_required(VERSION 2.8)

FIND_PACKAGE( OpenCV REQUIRED )

include_directories(${OpenCV2_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

configure_file(${CMAKE_SOURCE_DIR}/data/head.raw ${CMAKE_BINARY_DIR} COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/data/bounding-box.raw ${CMAKE_BINARY_DIR} COPYONLY)
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

link_directories(${CMAKE_BINARY_DIR})

if(FLANDMARK_STATIC)
	target_link_libraries(headtracker ${OpenCV_LIBS} flandmark_static)
else()
	target_link_libraries(headtracker ${OpenCV_LIBS} flandmark_shared)
endif()
file(GLOB headtracker-demo-SRC
    "demo/*.cpp"
)

set(ENABLE_DEMO FALSE CACHE BOOL "Whether to build the demo program")

if(ENABLE_DEMO)
    add_executable(headtracker-demo ${headtracker-demo-SRC})
    target_link_libraries(headtracker-demo headtracker ${OpenCV_LIBS})
endif()

file(GLOB headtracker-ftnoir-SRC
	"ftnoir/*.cpp"
)

add_executable(headtracker-ftnoir ${headtracker-ftnoir-SRC})
target_link_libraries(headtracker-ftnoir headtracker)
if(NOT WIN32)
	target_link_libraries(headtracker-ftnoir rt)
endif()
