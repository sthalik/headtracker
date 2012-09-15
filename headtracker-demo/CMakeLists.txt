project(headtracker-demo)
file(GLOB headtracker-demo-SRC
    "*.h"
    "*.cpp"
)
add_executable(${PROJECT_NAME} ${headtracker-demo-SRC})
target_link_libraries(headtracker-demo headtracker-lib opencv_core opencv_highgui)
include_directories("${PROJECT_SOURCE_DIR}/../headtracker-lib/")
