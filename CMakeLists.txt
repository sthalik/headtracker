project(headtracker)
cmake_minimum_required(VERSION 2.8)

FIND_PACKAGE( OpenCV REQUIRED )

configure_file(${CMAKE_SOURCE_DIR}/data/head.raw ${CMAKE_BINARY_DIR} COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/data/haarcascade_frontalface_alt2.xml ${CMAKE_BINARY_DIR} COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/data/flandmark_model.dat ${CMAKE_BINARY_DIR} COPYONLY)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})

file(GLOB headtracker-SRC
    "*.h"
    "*.cpp"
)
add_library(${PROJECT_NAME} SHARED ${headtracker-SRC})

if(WIN32)
    set(HT_SONAME dll)
endif(WIN32)

if(UNIX)
    set(HT_SONAME so)
endif(UNIX)

if(APPLE)
    set(HT_SONAME dylib)
endif(APPLE)

target_link_libraries(headtracker ${OpenCV_LIBS} ${CMAKE_BINARY_DIR}/libflandmark_shared.${HT_SONAME})
file(GLOB headtracker-demo-SRC
    "demo/*.cpp"
)

add_executable(headtracker-demo ${headtracker-demo-SRC})
target_link_libraries(headtracker-demo headtracker ${OpenCV_LIBS})
file(GLOB headtracker-ftnoir-SRC
	"ftnoir/*.cpp"
)

add_executable(headtracker-ftnoir ${headtracker-ftnoir-SRC})
target_link_libraries(headtracker-ftnoir headtracker)
if(NOT WIN32)
	target_link_libraries(headtracker-ftnoir rt)
endif()