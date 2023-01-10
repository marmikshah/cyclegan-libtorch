
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(artium)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_executable(artium ./src/main.cpp)

target_include_directories(artium PRIVATE ./include/)
target_link_libraries(artium "${TORCH_LIBRARIES}")


set_property(TARGET artium PROPERTY CXX_STANDARD 20)