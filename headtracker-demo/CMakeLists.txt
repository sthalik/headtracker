project(headtracker-demo)
file(GLOB headtracker-demo-SRC
    "*.h"
    "*.cpp"
)
add_executable(${PROJECT_NAME} ${headtracker-demo-SRC})
INCLUDE(${QT_USE_FILE})
ADD_DEFINITIONS(${QT_DEFINITIONS})
include_directories("${PROJECT_SOURCE_DIR}/../headtracker-lib/")
target_link_libraries(headtracker-demo headtracker-lib opencv_core opencv_highgui ${QT_QTCORE_LIBRARY} ${QT_QTNETWORK_LIBRARY})
