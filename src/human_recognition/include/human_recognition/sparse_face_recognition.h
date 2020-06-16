#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <glpk.h>
#include <string>
#include <map>

#define M1 6
#define N1 3

namespace human_recognition {

class SparseFaceRecognition
{
public:
    SparseFaceRecognition();
    ~SparseFaceRecognition();

    void appendSample(int classLabel, cv::Mat new_sample);
    void train();
    std::map<int, double> test(cv::Mat testImage);

private:
    cv::Mat A;
    int sampleCount;
    std::map<int, int> classTrainSize;

    cv::Mat linprog(cv::Mat f, cv::Mat Aeq, cv::Mat y, cv::Mat lb);
    void testLinprog();
    template <typename T> double getMatrixAt(cv::Mat m, int pos);
    template <typename T> void setMatrixAt(cv::Mat m, int pos, double val);
};

}

#endif // SPARSEMATRIX_H
