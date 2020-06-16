#include <human_recognition/calc_skeleton_features.h>

namespace human_recognition {

CalcSkeletonFeatures::CalcSkeletonFeatures()
{
    SkData.user_id = -1;
    confidence = 0.3;
    for (int i=0; i < featuresCount; ++i)
        body_features[i] = 0;
}

CalcSkeletonFeatures::~CalcSkeletonFeatures()
{ }

void CalcSkeletonFeatures::set_point_cloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudIn)
{
    pcl::VoxelGrid<pcl::PointXYZRGBA> sor;
    sor.setInputCloud (cloudIn);
    sor.setLeafSize (0.02f, 0.02f, 0.02f);
    sor.filter (cloud);
}

void CalcSkeletonFeatures::set_skeleton_data(const human_recognition::SkeletonMsg msg)
{
    SkData = msg;
}

std::vector<double> CalcSkeletonFeatures::get_features_skeleton_my()
{
    std::vector<double> features(featuresCount);

    if(SkData.user_id != -1)
    {
        int h = SkData.user_id;
        features[0] = 
            sqrt(pow(double(SkData.position[HEAD].x - SkData.position[NECK].x),2) +
                 pow(double(SkData.position[HEAD].y - SkData.position[NECK].y),2) +
                 pow(double(SkData.position[HEAD].z - SkData.position[NECK].z),2)
            );

        features[1] = 
            sqrt(pow(double(SkData.position[RIGHT_SHOULDER].x - SkData.position[TORSO].x),2) +
                 pow(double(SkData.position[RIGHT_SHOULDER].y - SkData.position[TORSO].y),2) +
                 pow(double(SkData.position[RIGHT_SHOULDER].z - SkData.position[TORSO].z),2)
            );

        features[2] = 
            sqrt(pow(double(SkData.position[LEFT_SHOULDER].x - SkData.position[TORSO].x),2) +
                 pow(double(SkData.position[LEFT_SHOULDER].y - SkData.position[TORSO].y),2) +
                 pow(double(SkData.position[RIGHT_SHOULDER].z - SkData.position[NECK].z),2)
            );

        features[3] = 
            sqrt(pow(double(SkData.position[HEAD].x - SkData.position[NECK].x),2) +
                 pow(double(SkData.position[HEAD].y - SkData.position[NECK].y),2) +
                 pow(double(SkData.position[HEAD].z - SkData.position[NECK].z),2)
            ) + 
            sqrt(pow(double(SkData.position[NECK].x - SkData.position[TORSO].x),2) +
                 pow(double(SkData.position[NECK].y - SkData.position[TORSO].y),2) +
                 pow(double(SkData.position[NECK].z - SkData.position[TORSO].z),2)
            ) + 
            sqrt(pow(double(SkData.position[TORSO].x - SkData.position[LEFT_HIP].x),2) +
                 pow(double(SkData.position[TORSO].y - SkData.position[LEFT_HIP].y),2) +
                 pow(double(SkData.position[TORSO].z - SkData.position[LEFT_HIP].z),2)
            ) +
            sqrt(pow(double(SkData.position[LEFT_HIP].x - SkData.position[LEFT_KNEE].x),2) +
                 pow(double(SkData.position[LEFT_HIP].y - SkData.position[LEFT_KNEE].y),2) +
                 pow(double(SkData.position[LEFT_HIP].z - SkData.position[LEFT_KNEE].z),2)
            ) +
            sqrt(pow(double(SkData.position[LEFT_KNEE].x - SkData.position[LEFT_FOOT].x),2) +
                 pow(double(SkData.position[LEFT_KNEE].y - SkData.position[LEFT_FOOT].y),2) +
                 pow(double(SkData.position[LEFT_KNEE].z - SkData.position[LEFT_FOOT].z),2)
            );

        features[4] = 
            sqrt(pow(double(SkData.position[NECK].x - SkData.position[TORSO].x),2) +
                 pow(double(SkData.position[NECK].y - SkData.position[TORSO].y),2) +
                 pow(double(SkData.position[NECK].z - SkData.position[TORSO].z),2)
            );

        features[5] = 
            sqrt(pow(double(SkData.position[RIGHT_HIP].x - SkData.position[RIGHT_KNEE].x),2) +
                 pow(double(SkData.position[RIGHT_HIP].y - SkData.position[RIGHT_KNEE].y),2) +
                 pow(double(SkData.position[RIGHT_HIP].z - SkData.position[RIGHT_KNEE].z),2)
            ) +
            sqrt(pow(double(SkData.position[RIGHT_KNEE].x - SkData.position[RIGHT_FOOT].x),2) +
                 pow(double(SkData.position[RIGHT_KNEE].y - SkData.position[RIGHT_FOOT].y),2) +
                 pow(double(SkData.position[RIGHT_KNEE].z - SkData.position[RIGHT_FOOT].z),2)
            );

        features[6] = 
            sqrt(pow(double(SkData.position[LEFT_HIP].x - SkData.position[LEFT_KNEE].x),2) +
                 pow(double(SkData.position[LEFT_HIP].y - SkData.position[LEFT_KNEE].y),2) +
                 pow(double(SkData.position[LEFT_HIP].z - SkData.position[LEFT_KNEE].z),2)
            ) +
            sqrt(pow(double(SkData.position[LEFT_KNEE].x - SkData.position[LEFT_FOOT].x),2) +
                 pow(double(SkData.position[LEFT_KNEE].y - SkData.position[LEFT_FOOT].y),2) +
                 pow(double(SkData.position[LEFT_KNEE].z - SkData.position[LEFT_FOOT].z),2)
            );

        features[7] = 
            sqrt(pow(double(SkData.position[NECK].x - SkData.position[TORSO].x),2) +
                 pow(double(SkData.position[NECK].y - SkData.position[TORSO].y),2) +
                 pow(double(SkData.position[NECK].z - SkData.position[TORSO].z),2)
            ) +
            sqrt(pow(double(SkData.position[TORSO].x - SkData.position[LEFT_HIP].x),2) +
                 pow(double(SkData.position[TORSO].y - SkData.position[LEFT_HIP].y),2) +
                 pow(double(SkData.position[TORSO].z - SkData.position[LEFT_HIP].z),2)
            ) +
            sqrt(pow(double(SkData.position[LEFT_HIP].x - SkData.position[LEFT_KNEE].x),2) +
                 pow(double(SkData.position[LEFT_HIP].y - SkData.position[LEFT_KNEE].y),2) +
                 pow(double(SkData.position[LEFT_HIP].z - SkData.position[LEFT_KNEE].z),2)
            ) +
            sqrt(pow(double(SkData.position[LEFT_KNEE].x - SkData.position[LEFT_FOOT].x),2) +
                 pow(double(SkData.position[LEFT_KNEE].y - SkData.position[LEFT_FOOT].y),2) +
                 pow(double(SkData.position[LEFT_KNEE].z - SkData.position[LEFT_FOOT].z),2)
            );

        features[8] = 
            sqrt(pow(double(SkData.position[LEFT_SHOULDER].x - SkData.position[NECK].x),2) +
                 pow(double(SkData.position[LEFT_SHOULDER].y - SkData.position[NECK].y),2) +
                 pow(double(SkData.position[LEFT_SHOULDER].z - SkData.position[NECK].z),2)
            );

        features[9] = 
            sqrt(pow(double(SkData.position[RIGHT_SHOULDER].x - SkData.position[NECK].x),2) +
                 pow(double(SkData.position[RIGHT_SHOULDER].y - SkData.position[NECK].y),2) +
                 pow(double(SkData.position[RIGHT_SHOULDER].z - SkData.position[NECK].z),2)
            );

        features[10] = 
            sqrt(pow(double(SkData.position[LEFT_SHOULDER].x - SkData.position[LEFT_ELBOW].x),2) +
                 pow(double(SkData.position[LEFT_SHOULDER].y - SkData.position[LEFT_ELBOW].y),2) +
                 pow(double(SkData.position[LEFT_SHOULDER].z - SkData.position[LEFT_ELBOW].z),2)
            ) +
            sqrt(pow(double(SkData.position[LEFT_ELBOW].x - SkData.position[LEFT_HAND].x),2) +
                 pow(double(SkData.position[LEFT_ELBOW].y - SkData.position[LEFT_HAND].y),2) +
                 pow(double(SkData.position[LEFT_ELBOW].z - SkData.position[LEFT_HAND].z),2)
            );

        features[11] = 
            sqrt(pow(double(SkData.position[RIGHT_SHOULDER].x - SkData.position[RIGHT_ELBOW].x),2) +
                 pow(double(SkData.position[RIGHT_SHOULDER].y - SkData.position[RIGHT_ELBOW].y),2) +
                 pow(double(SkData.position[RIGHT_SHOULDER].z - SkData.position[RIGHT_ELBOW].z),2)
            ) +
            sqrt(pow(double(SkData.position[RIGHT_ELBOW].x - SkData.position[RIGHT_HAND].x),2) +
                 pow(double(SkData.position[RIGHT_ELBOW].y - SkData.position[RIGHT_HAND].y),2) +
                 pow(double(SkData.position[RIGHT_ELBOW].z - SkData.position[RIGHT_HAND].z),2)
            );

        features[12] = 
            sqrt(pow(double(SkData.position[LEFT_HIP].x - SkData.position[LEFT_KNEE].x),2) +
                 pow(double(SkData.position[LEFT_HIP].y - SkData.position[LEFT_KNEE].y),2) +
                 pow(double(SkData.position[LEFT_HIP].z - SkData.position[LEFT_KNEE].z),2)
            );

        features[13] = 
            sqrt(pow(double(SkData.position[RIGHT_KNEE].x - SkData.position[RIGHT_HIP].x),2) +
                 pow(double(SkData.position[RIGHT_KNEE].y - SkData.position[RIGHT_HIP].y),2) +
                 pow(double(SkData.position[RIGHT_KNEE].z - SkData.position[RIGHT_HIP].z),2)
            );

        features[14] = features[LEFT_ELBOW] / features[RIGHT_KNEE];
        features[15] = features[LEFT_ELBOW] / features[RIGHT_HIP];

        a_star.set_cloud(cloud);
        features[16] = a_star.run(SkData.position[TORSO], SkData.position[RIGHT_SHOULDER]);  // torso and left shoulder
        features[17] = a_star.run(SkData.position[TORSO], SkData.position[LEFT_HIP]);  // torso and left hip
        features[18] = a_star.run(SkData.position[TORSO], SkData.position[RIGHT_HIP]); // torso and right hip
    }

    return features;
}

CalcSkeletonFeatures::AverageVariance CalcSkeletonFeatures::calc_variance_average(const std::vector<double> values)
{
    CalcSkeletonFeatures::AverageVariance feature;
    if (values.size() == 0) {
        return feature;
    }

    double Ex, Ex2;
    Ex = Ex2 = 0;
    for (int i=0; i < values.size() ;++i) {
        Ex += values.at(i);
        Ex2 += pow(values.at(i), 2);
    }

    Ex /= values.size();
    Ex2 /= values.size();
    feature.average = Ex;
    feature.variance = sqrt(Ex2 - pow(Ex,2));

    return feature;
}

std::vector<double> CalcSkeletonFeatures::calc_normalize_features(const std::vector<double> personBodyFeatures)
{
/*
    double* normalize_data = new double[features_count];
    double* variance_average;

    variance_average = calc_variance_average(normalize_data,features_count);
    for(int i = 0; i<features_count; ++i)
    {
        normalize_data[i] = (normalize_data[i] - variance_average[1])/variance_average[0];
        normalize_body_features[i] = normalize_data[i];
    }
    return normalize_data;
*/
}

void CalcSkeletonFeatures::set_floor(double a, double b, double c, double d)
{
    floorParameters.a = a;
    floorParameters.b = b;
    floorParameters.c = c;
    floorParameters.d = d;
    floorParameters.normal = sqrt(pow(floorParameters.a,2)+pow(floorParameters.b,2)+pow(floorParameters.c,2));
}

}
