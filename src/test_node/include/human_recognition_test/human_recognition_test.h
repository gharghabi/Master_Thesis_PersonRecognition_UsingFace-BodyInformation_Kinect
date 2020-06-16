#ifndef _HUMAN_RECOGNITION_TEST_H_
#define _HUMAN_RECOGNITION_TEST_H_

#include <ros/ros.h>
#include <ros/package.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <pcl_ros/point_cloud.h>
#include <cv_bridge/cv_bridge.h>

#include <human_recognition/SkeletonMsg.h>

#include <yaml-cpp/node.h>
#include <yaml-cpp/emitter.h>
#include <yaml-cpp/parser.h>

#include <map>

namespace human_recognition_test {

class HumanRecognitionTest {
public:
  HumanRecognitionTest(ros::NodeHandle);
  void publish();

private:
  human_recognition::SkeletonMsg::Ptr extractSkeletonData (const std::string line, int& frameNum);

private:
  ros::NodeHandle nh_;
  ros::NodeHandle m_nh_;

  ros::Publisher scene_publisher_;
  ros::Publisher person_publisher_;
  ros::Publisher skeleton_publisher_;

  std::map<int, std::string> infoMap;
};

}

#endif
