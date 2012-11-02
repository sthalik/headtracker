project(headtracker)
cmake_minimum_required(VERSION 2.8)

find_package(Qt COMPONENTS QtCore QtNetwork REQUIRED)
FIND_PACKAGE( OpenCV REQUIRED )

macro(configure_files srcDir destDir)
    message(STATUS "Configuring directory ${destDir}")
    make_directory(${destDir})

    file(GLOB templateFiles RELATIVE ${srcDir} ${srcDir}/*)
    foreach(templateFile ${templateFiles})
        set(srcTemplatePath ${srcDir}/${templateFile})
        if(NOT IS_DIRECTORY ${srcTemplatePath})
            message(STATUS "Copying file ${templateFile}")
            configure_file(
                    ${srcTemplatePath}
                    ${destDir}/${templateFile}
                    @ONLY)
        endif(NOT IS_DIRECTORY ${srcTemplatePath})
    endforeach(templateFile)
endmacro(configure_files)

configure_files(${CMAKE_SOURCE_DIR}/data/ ${CMAKE_BINARY_DIR})

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})

file(GLOB headtracker-SRC
    "*.h"
    "*.cpp"
)
add_library(${PROJECT_NAME} SHARED ${headtracker-SRC})
INCLUDE(${QT_USE_FILE})
ADD_DEFINITIONS(${QT_DEFINITIONS})
target_link_libraries(headtracker ${OpenCV_LIBS} ${QT_QTCORE_LIBRARY})
file(GLOB headtracker-demo-SRC
    "demo/*.h"
    "demo/*.cpp"
)
add_executable(headtracker-demo ${headtracker-demo-SRC})
target_link_libraries(headtracker-demo headtracker opencv_core249 opencv_highgui249 ${OpenCV_LIBS} ${QT_QTNETWORK_LIBRARY})

ADD_CUSTOM_TARGET(silly ALL)
