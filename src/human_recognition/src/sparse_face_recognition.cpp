#include <human_recognition/sparse_face_recognition.h>

namespace human_recognition {

SparseFaceRecognition::SparseFaceRecognition()
{
    A = cv::Mat(M1*N1, 1, CV_64FC1);
    sampleCount = 0;
}

SparseFaceRecognition::~SparseFaceRecognition()
{}

void SparseFaceRecognition::appendSample(int classLabel, cv::Mat new_sample)
{
    if (A.cols == sampleCount) {
        cv::Mat newCol(M1*N1, 1, CV_64FC1);
        cv::hconcat(A, newCol, A);
    }

    cv::Mat image;
    cv::Mat downSampledImage;
    if (new_sample.channels() == 3) {
        cv::cvtColor(new_sample, image, CV_RGB2GRAY);
    } else {
        image = new_sample;
    }
    cv::resize(image, downSampledImage, cv::Size(N1, M1), 0, 0, cv::INTER_AREA);
    for (int i = 0; i < M1*N1; ++i) {
        A.at<double>(i, sampleCount) = getMatrixAt<uchar>(downSampledImage, i);
    }
    sampleCount ++;
    classTrainSize[classLabel] ++;
}

void SparseFaceRecognition::train()
{
    cv::Mat B;
    cv::sqrt((A.t() * A).diag(), B);
    A = A * B.diag(B).inv();
}

std::map<int, double> SparseFaceRecognition::test(cv::Mat testImage)
{
    if (A.empty()) {
        return std::map<int, double>();
    }
    cv::Mat image;
    cv::Mat downSampledImage;
    if (testImage.channels() == 3)
        cv::cvtColor(testImage, image, CV_RGB2GRAY);
    else
        image = testImage;

    cv::resize(image, downSampledImage, cv::Size(N1, M1), 0, 0, cv::INTER_AREA);
    cv::Mat y(M1*N1, 1, CV_64FC1);
    for (int i = 0; i < M1*N1; ++i) {
        y.at<double>(i, 0) = getMatrixAt<uchar>(downSampledImage, i);
    }
    int m = A.rows;
    int n = A.cols;
    cv::Mat f = cv::Mat::ones(n*2, 1, CV_64FC1);
    cv::Mat Aeq(m, n*2, CV_64FC1);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            Aeq.at<double>(i, j) = A.at<double>(i, j);
        }
    }
    for (int i = 0; i < m; ++i) {
        for (int j = n; j < 2 * n; ++j) {
            Aeq.at<double>(i, j) = - A.at<double>(i, j - n);
        }
    }

    cv::Mat lb = cv::Mat::zeros(2*n, 1, CV_64FC1);
    cv::Mat x1 = cv::Mat::zeros(2*n, 1, CV_64FC1);
    x1 = linprog(f, Aeq, y, lb);
    cv::Mat x11(n, 1, CV_64FC1);
    cv::Mat x12(n, 1, CV_64FC1);
    for (int i = 0; i < n; ++i) {
        x11.at<double>(i, 0) = getMatrixAt<double>(x1, i);
    }
    for (int i = n; i < 2*n; ++i) {
        x12.at<double>(i-n, 0) = getMatrixAt<double>(x1, i);
    }
    //
    cv::Mat x1r = x11 - x12;
    int* nn = new int[classTrainSize.size()];
    nn[0] = classTrainSize[0];
    for (int i = 1; i < classTrainSize.size(); ++i) {
        nn[i] = nn[i-1] + classTrainSize[i];
    }
    int tmp_var = 0;
    int k1 = classTrainSize.size();
    double* tmp = new double[k1];
    double* tmp1 = new double[k1];
    for (int i = 0; i < k1; ++i) {
        cv::Mat delta_xi = cv::Mat::zeros(x1r.rows, 1, CV_64FC1);
        if (i == 0) {
            for (int j = 0; j < nn[i]; ++j) {
                setMatrixAt<double>(delta_xi, j, getMatrixAt<double>(x1r, j));
            }
        } else {
            tmp_var = tmp_var + nn[i-1];
            int begs = nn[i-1];
            int ends = nn[i];
            for (int k = begs; k < ends; ++k) {
                setMatrixAt<double>(delta_xi, k, getMatrixAt<double>(x1r, k));
            }
        }

        tmp[i] = cv::norm(y-(A*delta_xi));
        tmp1[i] = cv::norm(delta_xi, 1) / cv::norm(x1r, 1);
    }

    std::map<int, double> result;
    double sum = 0;
    double avg;

    double minTmp = tmp[0];
    double maxTmp1 = tmp1[0];
    int classLabel = 0;
    for (int i = 0; i < k1; ++i) {
        if (tmp1[i] > maxTmp1)
            maxTmp1 = tmp1[i];
        if (tmp[i] < minTmp) {
            minTmp = tmp[i];
            classLabel = i;
        }

        sum += tmp[i];
    }
    avg = sum / k1;

    int sparseConcIndex = (k1*maxTmp1-1)/(k1-1);

    for (int i = 0; i < k1; ++i) {
        result[i] = (sum - tmp[i]) / ((k1-1)*sum);
    }
    return result;

    // float validity = ((avg - minTmp) / avg) * 100.0;
    // return classLabel;
}

cv::Mat SparseFaceRecognition::linprog(cv::Mat f, cv::Mat Aeq, cv::Mat y, cv::Mat lb)
{
    glp_prob *lp;
    const int m = y.rows;
    const int n = Aeq.cols;

    int ia[1+10000], ja[1+10000];
    double ar[1+10000];
    cv::Mat result(1, n, CV_64FC1);

    glp_term_out (GLP_OFF);

    lp = glp_create_prob();
    glp_set_prob_name(lp, "sample");
    glp_set_obj_dir(lp, GLP_MIN);
    glp_add_rows(lp, m);
    for (int i = 0; i < m; ++i) {
        glp_set_row_bnds(lp, i+1, GLP_FX, y.at<double>(i, 0), y.at<double>(i, 0));
    }

    glp_add_cols(lp, n);
    for (int i = 0; i < n; ++i) {
        glp_set_col_bnds(lp, i+1, GLP_LO, lb.at<double>(i, 0), lb.at<double>(i, 0));
        glp_set_obj_coef(lp, i+1, f.at<double>(i, 0));
    }

    int counter = 1;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            ia[counter]=i+1, ja[counter]=j+1, ar[counter] = Aeq.at<double>(i, j);
            counter ++;
        }
    }

    glp_load_matrix(lp, m*n, ia, ja, ar);
    glp_simplex(lp, NULL);
//    std::cout << "val: " << glp_get_obj_val(lp) << std::endl;
    for (int i = 1; i <= n; ++i) {
        result.at<double>(0, i-1) = glp_get_col_prim(lp, i);
    }
    glp_delete_prob(lp);
    return result;
}

void SparseFaceRecognition::testLinprog()
{
    glp_prob *lp;
    const int m = 3;
    const int n = 3;
    int ia[1+10000], ja[1+10000];
    double ar[1+10000];

    lp = glp_create_prob();
    glp_set_prob_name(lp, "sample");
    glp_set_obj_dir(lp, GLP_MIN);
    glp_add_rows(lp, m);
    glp_set_row_bnds(lp, 1, GLP_FX, 10.0, 10.0);
    glp_set_row_bnds(lp, 2, GLP_FX, 20.0, 20.0);
    glp_set_row_bnds(lp, 3, GLP_FX, 30.0, 30.0);

    glp_add_cols(lp, n);
    for (int i = 0; i < n; ++i) {
        glp_set_col_bnds(lp, i+1, GLP_LO, 0.0, 0.0);
        glp_set_obj_coef(lp, i+1, 1.0);
    }

    ia[1]=1, ja[1]=1, ar[1]=10;
    ia[2]=1, ja[2]=2, ar[2]=0;
    ia[3]=1, ja[3]=3, ar[3]=10;
    ia[4]=2, ja[4]=1, ar[4]=20;
    ia[5]=2, ja[5]=2, ar[5]=0;
    ia[6]=2, ja[6]=3, ar[6]=30;
    ia[7]=3, ja[7]=1, ar[7]=10;
    ia[8]=3, ja[8]=2, ar[8]=0;
    ia[9]=3, ja[9]=3, ar[9]=30;

    glp_load_matrix(lp, m*n, ia, ja, ar);
    glp_simplex(lp, NULL);
//    std::cout << glp_get_obj_val(lp) << std::endl;
//    for (int i = 1; i <= n; ++i) {
//        std::cout << "res: " << glp_get_col_prim(lp, i) << std::endl;
//    }
    glp_delete_prob(lp);
}


template <typename T>
double SparseFaceRecognition::getMatrixAt(cv::Mat m, int pos)
{
    int row, col;
    row = pos % m.rows;
    col = pos / m.rows;
    return m.at<T>(row, col);
}

template <typename T>
void SparseFaceRecognition::setMatrixAt(cv::Mat m, int pos, double val)
{
    int row, col;
    row = pos % m.rows;
    col = pos / m.rows;
    m.at<T>(row, col) = val;
}

}
