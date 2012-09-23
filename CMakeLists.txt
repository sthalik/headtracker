project(headtracker)
cmake_minimum_required(VERSION 2.8)

find_package(Qt COMPONENTS QtCore REQUIRED)

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

add_subdirectory(headtracker-lib)
add_subdirectory(headtracker-demo)

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

configure_files(${CMAKE_SOURCE_DIR}/data/ ${CMAKE_BINARY_DIR}/headtracker-demo)

ADD_CUSTOM_TARGET(silly ALL)
