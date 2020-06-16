#include <boost/filesystem.hpp>
#include <fstream>
#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include "upperbodycore_msgs/Skeleton.h"
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <pcl/features/normal_3d_omp.h>
#include <boost/filesystem.hpp>
#include <pcl/features/vfh.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/people/ground_based_people_detection_app.h>
#include <pcl/common/time.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
#define FACE_CASCADE_NAME  "/home/shaghayegh/catkin_ws/src/thesis/thesis_skeleton_face/haarcascade_frontalface_default.xml"

int main()
{
    cv::Rect rect;
    double X = 0.0355305;
    double Y = -0.936718;//-
    double Z = 2.70465;

//    0.0355305,0.936718,2.70465
    cout<<fabs(X)<<endl;
    pcl::PointCloud<pcl::PointXYZRGBA> cloud;
    pcl::io::loadPCDFile ("/media/2e34bb6b-909d-437c-a8ed-c11cdfc25ca5/shaghayegh/data/3/still/ScenePC_57.pcd", cloud);
    cv::Mat image = cv::imread("/media/2e34bb6b-909d-437c-a8ed-c11cdfc25ca5/shaghayegh/data/3/still/RGB_57.png");
    cv::Mat gray;

    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Rect> faces;
    cv::Mat faceCrop;
    int flags_face= CV_HAAR_DO_ROUGH_SEARCH && CV_HAAR_FEATURE_MAX && CV_HAAR_DO_CANNY_PRUNING;//|CV_HAAR_FIND_BIGGEST_OBJECT
    CascadeClassifier face_cascade;
    if (!face_cascade.load(FACE_CASCADE_NAME))
    {
        cout<<" not load "<<endl;
    };

    float factor = 0.9;
    face_cascade.detectMultiScale( gray, faces, 1.1, 2, flags_face|CASCADE_SCALE_IMAGE, Size(0, 0) );//khub ba yek ghalat gray, faces, 1.2, 10, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50) )
    for (size_t i = 0; i < faces.size(); i++) {
        cout<<" hi "<<endl;
        rect.width = faces[i].width*factor;
        rect.height = faces[i].height*factor;
        rect.x = (faces[i].width - rect.width) / 2 + faces[i].x;
        rect.y += (faces[i].height - rect.height) / 2 + faces[i].y;
        cout<<rect.x<<endl;
        double sumX=0;
        double sumY=0;
        double sumZ=0;
        double sumNum=0;

        cout<<" x "<<rect.x<<" y "<<rect.y<<" width "<<rect.width<<" height "<<rect.height<<endl;
        for(int w=rect.x;w<(rect.width+rect.x);++w)
            for(int h=rect.y;h<(rect.height+rect.y) ;++h)
            {
                if(!isnan(cloud.at(w,h).x) && !isnan(cloud.at(w,h).y) && !isnan(cloud.at(w,h).z))
                {
                    sumX += cloud.at(w,h).x;
                    sumY += cloud.at(w,h).y;
                    sumZ += cloud.at(w,h).z;
                    ++sumNum;
//                    cout<<cloud.at(w,h).z<<" z ";
                }
            }
        cout<<endl;
        sumX /= sumNum;
        sumY /= sumNum;
        sumZ /=  sumNum;
        cout<<" sum x "<<sumX<<" sum y "<<sumY<<" sumz "<<sumZ<<" sum Num "<<sumNum<<endl;
        cout<<(sumX-X)<<" x "<<(sumY- Y)<<" y "<<(sumZ-Z)<<"z "<<endl;
        double diffX = sumX - X;
        double diffY = sumY - Y;
        double diffZ = sumZ - Z;

        if(fabs(diffX)<0.2 && fabs(diffY)<0.2 && fabs(diffZ)<0.2)
        {
            cout<<" hi "<<endl;
            faceCrop = image(rect);
            Mat faceCropImage(rect.width,rect.height,CV_8UC3,Scalar(0));
            cout<<rect.width<<" width "<<rect.height<<" hight "<<endl;
            cout<<faceCrop.rows<<" rows "<<faceCrop.cols<<" cols "<<endl;

            for(int w=rect.x;w<(rect.width+rect.x);++w)
                for(int h=rect.y;h<(rect.height+rect.y) ;++h)
                {
                    if(fabs(cloud.at(w,h).x-X)<0.3 && fabs(cloud.at(w,h).y-Y)<0.3 && fabs(cloud.at(w,h).z-Z))
                    {
//                        cout<<w-rect.x<<" "<<h-rect.y<<" "<<endl;
                        cv::Vec3b gh = faceCrop.at<Vec3b>(w-rect.x,h-rect.y);
                        faceCropImage.at<Vec3b>(w-rect.x,h-rect.y) = gh;

//                        cv::Vec3b gh = faceCrop.at<Vec3b>(37,37);
//                        faceCropImage.at<Vec3b>(37,37) = gh;

//                                <<endl;
//                        cout<<faceCrop.at<Vec3b>(w-rect.x,h-rect.y)<<endl;
//                        faceCropImage.at<Vec3b>(w-rect.x,h-rect.y) = faceCrop.at<Vec3b>(w-rect.x,h-rect.y);
                    }
                }


            if(faceCrop.rows>0)
            {
                cv::imshow("crop",faceCrop);
                cv::imshow("image ",faceCropImage);
                cv::waitKey(0);
            }
        }
    }

    //    cv::imwrite("/media/2e34bb6b-909d-437c-a8ed-c11cdfc25ca5/shaghayegh/data/5/train/"+modelNum,faceCrop);
    //    for(int i=0;i<cloud.width;++i)
    //    {
    //        for(int j=0;j<cloud.height;++j)
    //        {
    //            if(abs(cloud.at(i,j).x-X)<0.00001 && abs(cloud.at(i,j).y- Y)<0.00001 && abs(cloud.at(i,j).z-Z)<0.00001)
    //            {
    //                cout<<" cloud i "<<i<<" cloud j "<<j<<endl;
    //            }
    //        }
    //    }
}

