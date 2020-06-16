#include <ros/ros.h>
#include <human_recognition/human_recognition.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "human_recognition");
  ros::NodeHandle nh_;

  human_recognition::HumanRecognition* rec = new human_recognition::HumanRecognition(nh_);
  rec->init();
  rec->spin();

  return 0; 
}
