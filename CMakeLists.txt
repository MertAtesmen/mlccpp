cmake_minimum_required(VERSION 3.25)

project("ML C-CPP Project")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(mlccpp)

add_subdirectory(tests)

add_subdirectory(examples)
