#ifndef _HUMAN_RECOGNITION_H_
#define _HUMAN_RECOGNITION_H_

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <pcl_ros/point_cloud.h>
#include <cv_bridge/cv_bridge.h>

#include <human_recognition/calc_skeleton_features.h>
#include <human_recognition/sparse_face_recognition.h>

#include <yaml-cpp/node.h>
#include <yaml-cpp/emitter.h>
#include <yaml-cpp/parser.h>

#include <map>

namespace human_recognition {

class HumanRecognition {
public:
  HumanRecognition(ros::NodeHandle);
  bool init();
  void spin();

private:
  ros::NodeHandle nh_;
  ros::NodeHandle m_nh_;
  message_filters::Subscriber<sensor_msgs::PointCloud2> scene_subscriber_;
  message_filters::Subscriber<sensor_msgs::PointCloud2> person_subscriber_;
  message_filters::Subscriber<human_recognition::SkeletonMsg> skeleton_subscriber_;
  boost::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, human_recognition::SkeletonMsg> > sync_subscriber_;

  void testCallback(const sensor_msgs::PointCloud2ConstPtr& scene_msg_, const sensor_msgs::PointCloud2ConstPtr& person_msg_, const human_recognition::SkeletonMsg::ConstPtr& skeleton_test_);

  bool isReady;
  std::map<int, std::string> infoMap;
  std::map<int, std::vector<CalcSkeletonFeatures::AverageVariance> > trained_skeleton;

  SparseFaceRecognition* faceRecognition;
  CalcSkeletonFeatures* calcSkeletonFeatures;

private:
  human_recognition::SkeletonMsg extractSkeletonData(const std::string line, int& frameNum);
  double calculateFeatureProb(CalcSkeletonFeatures::AverageVariance trainData, double testData);
};

}

#endif
