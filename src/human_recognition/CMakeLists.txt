cmake_minimum_required(VERSION 2.8.3)
project(human_recognition)

find_package(catkin REQUIRED COMPONENTS 
  roscpp
  geometry_msgs
  sensor_msgs
  message_filters 
  pcl_conversions
  pcl_ros
  cv_bridge
)

find_package(PCL REQUIRED)
find_package(PCL 1.6 REQUIRED)

add_message_files(
  FILES
  SkeletonMsg.msg
)

# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

generate_messages(
   DEPENDENCIES
   geometry_msgs  # Or other packages containing msgs
)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(human_recognition_INCLUDE_DIRECTORIES
  include
  ${CMAKE_CURRENT_BINARY_DIR}
)

catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES human_recognition
  CATKIN_DEPENDS geometry_msgs sensor_msgs
  DEPENDS glpk
)

###########
## Build ##
###########

include_directories(${human_recognition_INCLUDE_DIRECTORIES})

add_executable(human_recognition_node 
	src/human_recognition_node.cpp
	src/human_recognition.cpp
	src/a_star.cpp
	src/calc_skeleton_features.cpp
	src/sparse_face_recognition.cpp
#	src/face_recognition.cpp
#	src/face_detection.cpp
#	src/skeleton_detection.cpp
)

add_dependencies(human_recognition_node human_recognition_generate_messages_cpp)

target_link_libraries(human_recognition_node
  ${catkin_LIBRARIES}
  glpk
  yaml-cpp
)

#############
## Install ##
#############

## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS human_recognition human_recognition_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_human_recognition.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
