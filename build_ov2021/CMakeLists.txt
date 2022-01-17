cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(segment)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-w -Wextra -D_GLIBCXX_USE_CXX11_ABI=0")
#set(CMAKE_CXX_FLAGS "-w -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Ofast")


include_directories(/home/yangw/intel/openvino_2021/deployment_tools/ngraph/include
                    /home/yangw/intel/openvino_2021/deployment_tools/inference_engine/include
                    /home/yangw/intel/openvino_2021/deployment_tools/inference_engine/external/tbb/include
                    /home/yangw/intel/openvino_2021/deployment_tools/inference_engine/external/hddl/include
                    /home/yangw/local/opencv/include
                    )

link_directories(
                 /home/yangw/intel/openvino_2021/deployment_tools/inference_engine/lib/intel64
                 /home/yangw/intel/openvino_2021/deployment_tools/ngraph/lib
                 /home/yangw/intel/openvino_2021/deployment_tools/inference_engine/external/tbb/lib
                 /home/yangw/intel/openvino_2021/deployment_tools/inference_engine/external/hddl/lib
                 /home/yangw/local/opencv/lib64)

# build segment
add_executable(segment ${CMAKE_CURRENT_SOURCE_DIR}/../src/main.cpp)
target_link_libraries(segment PUBLIC
                     opencv_world
                     inference_engine_ir_reader inference_engine_legacy inference_engine_lp_transformations
                     inference_engine_preproc inference_engine inference_engine_transformations MKLDNNPlugin)
set_property(TARGET segment PROPERTY CXX_STANDARD 14)


