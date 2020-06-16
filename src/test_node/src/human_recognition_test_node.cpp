#include <human_recognition_test/human_recognition_test.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "human_recognition_test");
  ros::NodeHandle nh_;

  human_recognition_test::HumanRecognitionTest* test = new human_recognition_test::HumanRecognitionTest(nh_);
  test->publish();

  return 0; 
}

     

