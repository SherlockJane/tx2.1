cmake_minimum_required(VERSION 2.8)
project(MVCamera)

SET(CMAKE_C_COMPILER g++)
if(CMAKE_COMPILER_IS_GNUCXX)
  add_compile_options(-std=c++11)
message(STATUS "optional:-std=c++11")
endif(CMAKE_COMPILER_IS_GNUCXX)

set(CMAKE_BUILD_TYPE "Debug")

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()


include_directories(
        ${PROJECT_SOURCE_DIR}/include

)
### MV CAMERA
#add_executable(test_mv src/test_MVCamera.cpp src/MVCamera.cpp src/MarkSensor.cpp  src/MarkerParams.cpp src/serial.cpp  src/templetetracking.cpp)
#target_link_libraries(
#        test_mv
#        ${OpenCV_LIBRARIES}
#        ${PROJECT_SOURCE_DIR}/lib/arm64/libMVSDK.so

#        -lpthread
#)
### USB CAMERA
add_executable(usb_main src/main.cpp src/MarkerParams.cpp src/MarkSensor.cpp  src/serial.cpp src/templetetracking.cpp)
target_link_libraries(
    usb_main
    ${OpenCV_LIBS}
    -lpthread
    )





