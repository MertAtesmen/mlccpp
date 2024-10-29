add_executable(cpp_example example.cpp)
add_executable(c_example example.c)

target_link_libraries(cpp_example mlccpp)
target_link_libraries(c_example mlccpp)