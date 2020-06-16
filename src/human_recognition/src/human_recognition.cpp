#include <human_recognition/human_recognition.h>

namespace human_recognition {

HumanRecognition::HumanRecognition(ros::NodeHandle nh_) : 
  nh_(nh_),
  m_nh_("~"),
  isReady(false),
  faceRecognition(new SparseFaceRecognition()),
  calcSkeletonFeatures(new CalcSkeletonFeatures())
{
  scene_subscriber_.subscribe(nh_, "/scene_cloud", 1);
  person_subscriber_.subscribe(nh_, "/person_cloud", 1);
  skeleton_subscriber_.subscribe(nh_, "/skeleton", 1);

  sync_subscriber_ = boost::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2,human_recognition::SkeletonMsg> > (new message_filters::TimeSynchronizer<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, human_recognition::SkeletonMsg>(
    scene_subscriber_, person_subscriber_, skeleton_subscriber_, 1
  ));
  sync_subscriber_->registerCallback(boost::bind(&HumanRecognition::testCallback, this, _1, _2, _3));
}

void HumanRecognition::testCallback(const sensor_msgs::PointCloud2ConstPtr& scene_msg_, const sensor_msgs::PointCloud2ConstPtr& person_msg_, const human_recognition::SkeletonMsg::ConstPtr& skeleton_test_) {
  cv::Mat test;
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene_cloud;
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr person_cloud;

  if (scene_msg_->width != 0 && person_msg_->width != 0) {
     pcl::fromROSMsg (*person_msg_, *person_cloud);
     pcl::fromROSMsg (*scene_msg_, *scene_cloud);
     sensor_msgs::Image::Ptr image;
     pcl::toROSMsg(*scene_msg_, *image);
     cv_bridge::CvImagePtr cv_ptr;
     try {
       cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
       if (cv_ptr->image.size().width > 0) {
          cv_ptr->image.copyTo(test);
       } else {
          ROS_ERROR("cv_bridge: wrong image format!");
          return;
       }
     } catch (cv_bridge::Exception &e) {
       ROS_ERROR("cv_bridge: exception: %s", e.what());
       return;
     }
  } else {
    ROS_ERROR("pcl cloud: wrong input message!");
    return;
  }

  // TODO: crop face rect from image

  // Calculate Face Recognition Results
  std::map<int, double> faceResult = faceRecognition->test(test);

  // Calculate probs
  calcSkeletonFeatures->set_skeleton_data(*skeleton_test_);
  std::vector<double> skeleton_feature = calcSkeletonFeatures->get_features_skeleton_my();

  if (faceResult.size() != trained_skeleton.size()) {
    ROS_ERROR("problem in classes size!");
    return;
  }

  int classSize = trained_skeleton.size();
  int featuresSize = skeleton_feature.size() + 1; // +1 for face

  // weights
  std::vector<double> weights(featuresSize);
  weights[0]  = 1.0;
  weights[1]  = 0.0;
  weights[2]  = 0.0;
  weights[3]  = 0.0;
  weights[4]  = 0.0;
  weights[5]  = 0.0;
  weights[6]  = 0.0;
  weights[7]  = 0.0;
  weights[8]  = 0.0;
  weights[9]  = 0.0;
  weights[10] = 0.0;
  weights[11] = 0.0;
  weights[12] = 0.0;
  weights[13] = 0.0;
  weights[14] = 0.0;
  weights[15] = 0.0;
  weights[16] = 0.0;
  weights[17] = 0.0;

  double weightsSum = 0.0;
  for (int i=0; i < featuresSize; i++)
    weightsSum += weights[i];
  if (weightsSum != 1.0) {
    ROS_ERROR("wrong weight values, sum(weights) = 1.0");
    return;
  }
  //

  std::vector<double> results(classSize);
  for (int i=0; i < classSize; i++) {
    results[i] = 0.0;
    std::vector<double> probs(featuresSize);
    // skeleton features
    for (int j=0; j < skeleton_feature.size(); j++) {
      probs[j] = calculateFeatureProb(trained_skeleton[i][j], skeleton_feature[j]);
      results[i] += weights[j] * log(probs[j]);
    }
    // face feature
    probs[featuresSize-1] = faceResult[i];
    results[i] += weights[featuresSize-1] * log(probs[featuresSize-1]);

    std::cout << "result #" << i << ": " <<results[i] << std::endl;
  }
}

bool HumanRecognition::init()
{
  std::string prefix;
  m_nh_.param("train_path", prefix, ros::package::getPath("human_recognition") + "/train_set/");

  std::string infoPath = prefix + "info.yaml";
  std::ifstream fileInfo (infoPath.c_str());
  YAML::Parser p(fileInfo);
  YAML::Node n;
  std::string extension;

  if (! p.GetNextDocument(n)) {
     ROS_ERROR_STREAM("Could not read info file");
     if (fileInfo.fail())
        ROS_ERROR("The I/O error might have been: %s", strerror(errno));
     return false;
  }

  for (YAML::Iterator it = n.begin(); it != n.end(); ++it) {
     infoMap[it.first().to<int>()] = it.second().to<std::string>();
  }

  for (std::map<int, std::string>::iterator it = infoMap.begin(); it != infoMap.end(); it++) {
     // map: class label -> directory

     int classLabel = it->first;

     std::string humanPrefix;
     humanPrefix = prefix + "/data/" + it->second;

     // read face files
     extension = ".pgm"; // TODO rename this files
     for (boost::filesystem::directory_iterator it(humanPrefix+"/face/"); it != boost::filesystem::directory_iterator (); ++it)
     {
        if (boost::filesystem::is_regular_file(it->status()) && boost::filesystem::extension(it->path()) == extension)
        {
           std::string pathToImage = it->path().string();
           cv::Mat new_face_sample = cv::imread(pathToImage);
           faceRecognition->appendSample(classLabel, new_face_sample);
        }
     }
     
     // read skeleton and pcd files
     std::string skeletonInfoPath = humanPrefix + "/skeleton/skeleton.txt";
     std::ifstream skeletonFileInfo;
     skeletonFileInfo.open (skeletonInfoPath.c_str ());
     std::string line;
     std::vector<std::vector<double> > features(CalcSkeletonFeatures::featuresCount);

     getline (skeletonFileInfo, line);
     while (! skeletonFileInfo.eof()) {
        if (line.empty()) continue;

        // read skeleton file
        int frameNum;
        human_recognition::SkeletonMsg newSkeletonData = extractSkeletonData(line, frameNum);
        calcSkeletonFeatures->set_skeleton_data(newSkeletonData);

        // read pcd files
//        std::string pathToScenePCD = humanPrefix + "/skeleton/ScenePC_" + boost::lexical_cast<std::string>(frameNum) + ".pcd";
//        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene_cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
//        pcl::io::loadPCDFile(pathToScenePCD, *scene_cloud);

//        std::string pathToPersonPCD = humanPrefix + "/skeleton/personPC_" + boost::lexical_cast<std::string>(frameNum) + ".pcd";
//        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr person_cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
//        pcl::io::loadPCDFile(pathToPersonPCD, *person_cloud);

//        calcSkeletonFeatures->set_point_cloud(person_cloud);

        // calculate features
        std::vector<double> skeleton_feature = calcSkeletonFeatures->get_features_skeleton_my();

        for (int i=0; i < skeleton_feature.size(); i++) {
            features[i].push_back(skeleton_feature.at(i));
        }
        
        getline (skeletonFileInfo, line);
     }
     skeletonFileInfo.close();
     std::vector<CalcSkeletonFeatures::AverageVariance> person_skeleton(features.size());
     std::cout << std::endl;
     for (int i=0; i < features.size(); i++) {
        person_skeleton[i] = calcSkeletonFeatures->calc_variance_average(features.at(i));
        std::cout << person_skeleton[i].average << " " << person_skeleton[i].variance << ";" << std::endl;
     }
     trained_skeleton[classLabel] = person_skeleton;
  }

  faceRecognition->train();
  isReady = true;

  return isReady;
}

void HumanRecognition::spin()
{
  if (! isReady) {
     ROS_ERROR("Recognition Module is not ready!");
  }

  ros::spin();
}

human_recognition::SkeletonMsg HumanRecognition::extractSkeletonData (const std::string line, int& frameNum)
{
  human_recognition::SkeletonMsg msg;
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

     msg.confidence.push_back(conf);
     msg.orientation.push_back(orientation);
     msg.position.push_back(position);
  }
  return msg;
}

double HumanRecognition::calculateFeatureProb(CalcSkeletonFeatures::AverageVariance trainData, double testData)
{
  if (! isReady)
     return -1;

  // p = (1 / (variance * sqrt(2*PI))) * e^((-(f-average)^2) / (2*variance^2))
  double normalizedPow = -pow(testData - trainData.average, 2) / (2 * trainData.variance*trainData.variance);
  return (1 / (trainData.variance * sqrt(2 * M_PI))) * pow(M_E, normalizedPow);
}

}
