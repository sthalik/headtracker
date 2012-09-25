project(headtracker-lib)
file(GLOB headtracker-lib-SRC
    "*.h"
    "*.cpp"
)
add_library(${PROJECT_NAME} SHARED ${headtracker-lib-SRC})
INCLUDE(${QT_USE_FILE})
ADD_DEFINITIONS(${QT_DEFINITIONS})
target_link_libraries(headtracker-lib opencv_core opencv_calib3d opencv_features2d opencv_imgproc opencv_objdetect opencv_video ${QT_QTCORE_LIBRARY})
