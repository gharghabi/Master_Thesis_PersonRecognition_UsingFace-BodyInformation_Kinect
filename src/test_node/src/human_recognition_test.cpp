#include <human_recognition_test/human_recognition_test.h>

namespace human_recognition_test {

HumanRecognitionTest::HumanRecognitionTest(ros::NodeHandle nh_) : 
  nh_(nh_),
  m_nh_("~")
{
  scene_publisher_ = nh_.advertise<sensor_msgs::PointCloud2>("/scene_cloud", 1);
  person_publisher_ = nh_.advertise<sensor_msgs::PointCloud2>("/person_cloud", 1);
  skeleton_publisher_ = nh_.advertise<human_recognition::SkeletonMsg>("/skeleton", 1);
}

void HumanRecognitionTest::publish()
{
  std::string prefix;
  m_nh_.param("test_path", prefix, ros::package::getPath("human_recognition_test") + "/test_set/");

  std::string infoPath = prefix + "info.yaml";
  std::ifstream fileInfo (infoPath.c_str());
  YAML::Parser p(fileInfo);
  YAML::Node n;
  std::string extension;

  if (! p.GetNextDocument(n)) {
     ROS_ERROR_STREAM("Could not read info file");
     if (fileInfo.fail())
        ROS_ERROR("The I/O error might have been: %s", strerror(errno));
     return;
  }

  for (YAML::Iterator it = n.begin(); it != n.end(); ++it) {
     infoMap[it.first().to<int>()] = it.second().to<std::string>();
  }

  for (std::map<int, std::string>::iterator it = infoMap.begin(); it != infoMap.end(); it++) {
     // map: class label -> directory

     int classLabel = it->first;

     std::string humanPrefix;
     humanPrefix = prefix + "/data/" + it->second;
    
     // read skeleton and pcd files
     std::string skeletonInfoPath = humanPrefix + "/skeleton.txt";
     std::ifstream skeletonFileInfo;
     skeletonFileInfo.open (skeletonInfoPath.c_str ());
     std::string line;

     getline (skeletonFileInfo, line);
     while (! skeletonFileInfo.eof()) {
        if (line.empty()) continue;

        // read skeleton file
        int frameNum;
        human_recognition::SkeletonMsg::Ptr newSkeletonData = extractSkeletonData(line, frameNum);

        extension = ".pcd";
        // read scene PCD file
        std::string pathToScenePCD = humanPrefix + "/ScenePC_" + boost::lexical_cast<std::string>(frameNum) + extension;
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene_cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
        pcl::io::loadPCDFile(pathToScenePCD, *scene_cloud);
        sensor_msgs::PointCloud2::Ptr scene_cloud_msg;
        pcl::toROSMsg(*scene_cloud, *scene_cloud_msg);
 
        // read person PCD file
        std::string pathToPersonPCD = humanPrefix + "/personPC_" + boost::lexical_cast<std::string>(frameNum) + extension;
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr person_cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
        pcl::io::loadPCDFile(pathToPersonPCD, *person_cloud);
        sensor_msgs::PointCloud2::Ptr person_cloud_msg;
        pcl::toROSMsg(*person_cloud, *person_cloud_msg);

        // publish msgs
        ros::Time now = ros::Time::now();
        newSkeletonData->header.stamp = now;
        scene_cloud_msg->header.stamp = now;
        person_cloud_msg->header.stamp = now;

        scene_publisher_.publish(scene_cloud_msg);
        person_publisher_.publish(person_cloud_msg);
        skeleton_publisher_.publish(newSkeletonData);

        getline (skeletonFileInfo, line);
     }
     skeletonFileInfo.close();
  }
}

human_recognition::SkeletonMsg::Ptr HumanRecognitionTest::extractSkeletonData (const std::string line, int& frameNum)
{
  human_recognition::SkeletonMsg::Ptr msg (new human_recognition::SkeletonMsg);
  geometry_msgs::Vector3 position;
  geometry_msgs::Quaternion orientation;
  float conf;

  std::vector<std::string> strs;
  boost::algorithm::split(strs,line,boost::algorithm::is_any_of(","));

  int jointNum = 5;
  int i = 0;

  frameNum = boost::lexical_cast<double>(strs.at(i++));
  while (i < strs.size()-1) {
     position.x = boost::lexical_cast<double>(strs.at(i++));
     position.y = boost::lexical_cast<double>(strs.at(i++));
     position.z = boost::lexical_cast<double>(strs.at(i++));

     orientation.w = boost::lexical_cast<double>(strs.at(i++));
     orientation.x = boost::lexical_cast<double>(strs.at(i++));
     orientation.y = boost::lexical_cast<double>(strs.at(i++));
     orientation.z = boost::lexical_cast<double>(strs.at(i++));

     float conf = boost::lexical_cast<float>(strs.at(i++));

     msg->confidence.push_back(conf);
     msg->orientation.push_back(orientation);
     msg->position.push_back(position);
  }
  return msg;
}

}
