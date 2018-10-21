cmake_minimum_required(VERSION 2.6)
project(estimatetraj)

add_compile_options(-std=c++11)

find_package(OpenCV 3.0 QUIET)
find_package(Eigen3  REQUIRED)

include_directories( ${EIGEN3_INCLUDE_DIR} )

add_executable(estimatetraj main.cpp)
target_link_libraries( estimatetraj
 ${OpenCV_LIBS}
 ${EIGEN3_LIBS}
 )
 
install(TARGETS estimatetraj RUNTIME DESTINATION bin)
