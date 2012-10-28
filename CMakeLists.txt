project(headtracker)
cmake_minimum_required(VERSION 2.8)

find_package(Qt COMPONENTS QtCore QtNetwork REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
if(NOT cmake_build_type_tolower STREQUAL "debug"
   AND NOT cmake_build_type_tolower STREQUAL "release")
  message(FATAL_ERROR "Unknown build type")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug)

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
target_link_libraries(headtracker opencv_core249 opencv_calib3d249 opencv_features2d249 opencv_imgproc249 opencv_objdetect249 opencv_video249 opencv_highgui249 ${QT_QTCORE_LIBRARY})
file(GLOB headtracker-demo-SRC
    "demo/*.h"
    "demo/*.cpp"
)
add_executable(headtracker-demo ${headtracker-demo-SRC})
target_link_libraries(headtracker-demo headtracker opencv_core249 opencv_highgui249 ${QT_QTCORE_LIBRARY} ${QT_QTNETWORK_LIBRARY})

ADD_CUSTOM_TARGET(silly ALL)
