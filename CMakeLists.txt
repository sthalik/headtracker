project(headtracker)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV 3.0 REQUIRED)

include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})
link_directories(${CMAKE_BINARY_DIR})

configure_file(${CMAKE_SOURCE_DIR}/data/head.raw ${CMAKE_BINARY_DIR} COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/data/bounding-box.raw ${CMAKE_BINARY_DIR} COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/data/haarcascade_frontalface_alt2.xml ${CMAKE_BINARY_DIR} COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/data/flandmark_model.dat ${CMAKE_BINARY_DIR} COPYONLY)

if(CMAKE_COMPILER_IS_GNUCXX OR APPLE)
    set(inlines "-fvisibility=hidden -fvisibility-inlines-hidden")
    set(CMAKE_CXX_FLAGS " -std=c++11 ${CMAKE_CXX_FLAGS} ${inlines} ")
    set(CMAKE_C_FLAGS " ${CMAKE_C_FLAGS} ${inlines} ")
endif()

if(APPLE)
    set(CMAKE_MACOSX_RPATH OFF)
    set(apple-frameworks "-stdlib=libc++ -framework Cocoa -framework CoreFoundation -lobjc -lz -framework Carbon")
    set(CMAKE_SHARED_LINKER_FLAGS " ${apple-frameworks} ${CMAKE_SHARED_LINKER_FLAGS}")
    #set(CMAKE_STATIC_LINKER_FLAGS " ${apple-frameworks} ${CMAKE_STATIC_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS " ${apple-frameworks} ${CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS " ${apple-frameworks} ${CMAKE_MODULE_LINKER_FLAGS}")
    set(CMAKE_CXX_FLAGS " -stdlib=libc++ ${CMAKE_CXX_FLAGS}")
endif()

# nix -rdynamic passed from Linux-GNU.cmake
if(CMAKE_COMPILER_IS_GNUCXX)
    set(__LINUX_COMPILER_GNU 1)
    set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)
    set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)
endif()

set(FLANDMARK_STATIC TRUE CACHE BOOL "Whether flandmark static library is used")
set(LIBHEADTRACKER_STATIC TRUE CACHE BOOL "Whether flandmark static library is used")

if(LIBHEADTRACKER_STATIC)
    set(HT_TYPE STATIC)
else()
    set(HT_TYPE DYNAMIC)
endif()

file(GLOB headtracker-SRC
    "*.[ch]"
    "*.[ch]pp"
)
add_library(${PROJECT_NAME} ${HT_TYPE} ${headtracker-SRC})

if(FLANDMARK_STATIC)
	target_link_libraries(headtracker flandmark_static ${OpenCV_LIBS})
else()
	target_link_libraries(headtracker flandmark_shared ${OpenCV_LIBS})
endif()
file(GLOB headtracker-demo-SRC
    "demo/*.cpp"
)

set(ENABLE_DEMO FALSE CACHE BOOL "Whether to build the demo program")

if(ENABLE_DEMO)
    add_executable(headtracker-demo ${headtracker-demo-SRC})
    target_link_libraries(headtracker-demo headtracker ${OpenCV_LIBS})
endif()
