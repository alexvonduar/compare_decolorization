
cmake_minimum_required (VERSION 3.1)

project(decolorize)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 99)

#set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -Wall -mfpu=neon")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -mfpu=neon")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -mfpu=neon")

# The version number.
set(decolorize_VERSION_MAJOR 0)
set(decolorize_VERSION_MINOR 1)

#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

#find_package(Eigen3 3.1.0 REQUIRED)
#find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  "${PROJECT_SOURCE_DIR}/include"
  "${OpenCV_INCLUDE_DIRS}"
  "${PROJECT_SOURCE_DIR}/rtcprgb2gray"
  )
list(APPEND LIBS ${OpenCV_LIBRARIES})

add_executable(decolorize
  src/main.cpp
  rtcprgb2gray/rtcprgb2gray.cpp
  )
target_link_libraries(decolorize ${LIBS})
