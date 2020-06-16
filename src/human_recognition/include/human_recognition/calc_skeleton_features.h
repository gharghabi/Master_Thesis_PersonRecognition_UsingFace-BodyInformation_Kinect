#ifndef _CALC_SKELETON_FEATURES_H_
#define _CALC_SKELETON_FEATURES_H_

#include <ros/ros.h>
#include <ros/package.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/vfh.h>
#include <boost/filesystem.hpp>
#include <fstream>
#include <human_recognition/SkeletonMsg.h>
#include <human_recognition/a_star.h>

namespace human_recognition {

struct floor_coefficient
{
    double a;
    double b;
    double c;
    double d;
    double normal;
};

class CalcSkeletonFeatures
{
public:
    enum JointName {
       HEAD = 0,
       NECK,
       TORSO,
       LEFT_SHOULDER,
       LEFT_ELBOW,
       LEFT_HAND,
       RIGHT_SHOULDER,
       RIGHT_ELBOW,
       RIGHT_HAND,
       LEFT_HIP,
       LEFT_KNEE,
       LEFT_FOOT,
       RIGHT_HIP,
       RIGHT_KNEE,
       RIGHT_FOOT
    };

    typedef struct AverageVariance {
       AverageVariance() : average(0), variance(0) {};
       double average;
       double variance;
    } AverageVariance;

public:
    static int const jointNum = 15;
    static int const featuresCount = 19;

private:
    pcl::PointCloud<pcl::PointXYZRGBA> cloud;
    floor_coefficient floorParameters;
    double body_features[featuresCount];
    double normalize_body_features[featuresCount];
    float confidence;
    human_recognition::SkeletonMsg SkData;
    AStar a_star;

public:
    CalcSkeletonFeatures();
    ~CalcSkeletonFeatures();

    void set_skeleton_data(const human_recognition::SkeletonMsg msg);
    void set_point_cloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudIn);

    std::vector<double> get_features_skeleton_my(); // [16]
    AverageVariance calc_variance_average(const std::vector<double> values);
    std::vector<double> calc_normalize_features(std::vector<double> personBodyFeatures);

    void set_floor(double a, double b, double c, double d);
};

}

#endif
