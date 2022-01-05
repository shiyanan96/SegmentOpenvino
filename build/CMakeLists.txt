cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(segment)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-w -Wextra -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Ofast")


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/openvino/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/opencv/include)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/openvino/lib
                 ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/opencv/lib)

# build segment
add_executable(segment ${CMAKE_CURRENT_SOURCE_DIR}/../main.cpp)
target_link_libraries(segment PUBLIC
                     opencv_world
                     inference_engine_ir_reader inference_engine_legacy inference_engine_lp_transformations
                     inference_engine_preproc inference_engine inference_engine_transformations MKLDNNPlugin)
set_property(TARGET segment PROPERTY CXX_STANDARD 14)


