cmake_minimum_required(VERSION 3.5)

project(asio_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BOOST_ROOT /usr/include/boost)
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost COMPONENTS system thread)
include_directories(${Boost_INCLUDE_DIR})

add_executable(asio_tests src/main.cpp)
target_link_libraries(asio_tests ${Boost_LIBRARIES})
