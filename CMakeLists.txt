project(headtracker)
cmake_minimum_required(VERSION 2.8)

FIND_PACKAGE(Qt COMPONENTS QtCore QtNetwork REQUIRED)
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
INCLUDE(${QT_USE_FILE})
ADD_DEFINITIONS(${QT_DEFINITIONS})
target_link_libraries(headtracker ${OpenCV_LIBS} ${CMAKE_BINARY_DIR}/libflandmark_shared.dll)
file(GLOB headtracker-demo-SRC
    "demo/*.h"
    "demo/*.cpp"
)
add_executable(headtracker-demo ${headtracker-demo-SRC})
target_link_libraries(headtracker-demo headtracker ${OpenCV_LIBS} ${QT_QTCORE_LIBRARY} ${QT_QTNETWORK_LIBRARY})

file(GLOB headtracker-ftnoir-SRC
    "ftnoir/*.h"
    "ftnoir/*.cpp"
)

add_executable(headtracker-ftnoir ${headtracker-ftnoir-SRC})
target_link_libraries(headtracker-ftnoir headtracker)

ADD_CUSTOM_TARGET(silly ALL)
