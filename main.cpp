#include <opencv2/opencv.hpp>
#include<iostream>
//#include <opencv2/shape.hpp>

using namespace std;
//using namespace cv;

int main()
{
    cv::Mat image= cv::imread("/home/luna/image.png");
    if (!image.data)
        return 0;
    //cv::namedWindow("Image");
    //cv::imshow("Image",image);

    // 定义边框矩形
    cv::Rect rectangle(61,78,210,210);
    // 定义前景、背景和分割结果
    cv::Mat bgModel,fgModel,result;

    // GrabCut分割
    cv::grabCut(image,
                result,
                rectangle,
                bgModel,
                fgModel,
                5,
                cv::GC_INIT_WITH_RECT); // use rectangle

    // 标记可能属于前景的区域
    cv::compare(result,cv::GC_PR_FGD,result,cv::CMP_EQ);
    // or:
    //	result= result&1;
    //cv::imshow("test",result);
    //cv::waitKey(0);
    // 创建前景图像
    //cv::Mat foreground(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    //image.copyTo(foreground,result); // 复制前景图像

    // 在原图像绘制矩形区域
    //cv::rectangle(image, rectangle, cv::Scalar(200,0,200),4);
    //cv::namedWindow("Rectangle");
    //cv::imshow("Rectangle",image);

    //cqv::namedWindow("Foreground");
    //cv::imshow("Foreground",foreground);

    //cv::waitKey();
    cv::Mat edges;
    Canny(result, edges, 50, 150, 5);
    // 找到所有轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // 计算每个轮廓的长度
    std::vector<double> lengths;
    for (const auto& contour : contours) {
        lengths.push_back(cv::arcLength(contour, true));
    }

    // 设置长度阈值
    double threshold = 100;

    // 过滤轮廓
    std::vector<std::vector<cv::Point>> filtered_contours;
    for (size_t i = 0; i < contours.size(); ++i) {
        if (lengths[i] > threshold) {
            filtered_contours.push_back(contours[i]);
        }
    }

    //double length = cv::arcLength(edges, true);
    // 绘制过滤后的轮廓
    cv::Mat filtered_edges = cv::Mat::zeros(edges.size(), CV_8UC1);
    cv::drawContours(filtered_edges, filtered_contours, -1, cv::Scalar(255), 1);

    //cv::imshow("Edges", edges);
    //cv::imshow("Filtered Edges", filtered_edges);
    //cv::waitKey(0);
    //cout << edges <<endl;

    cv::Mat gradient_x, gradient_y;
    Sobel(filtered_edges, gradient_x, CV_32F, 0, 1);
    Sobel(filtered_edges, gradient_y, CV_32F, 0, 1);

    cv::Mat gradient_magnitude;
    sqrt(gradient_x.mul(gradient_x) + gradient_y.mul(gradient_y), gradient_magnitude);

    double avg_smoothness = mean(gradient_magnitude)(0);
    cout << "Average edge smoothness: " << avg_smoothness << endl;

    //imshow("test", filtered_edges);
    //cv::waitKey(0);

    std::string smoothness_str = std::to_string(avg_smoothness);
    cv::Mat text_image = cv::Mat::zeros(cv::Size(200, 50), CV_8UC3);
    cv::putText(text_image, smoothness_str, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    //imshow("test", text_image);
    //cv::waitKey(0);
    cv::Mat sword;
    //double alpha = 0.7; double beta;
    //beta = ( 1.0 - alpha );
    //cv::addWeighted(filtered_edges, 0.5, text_image, 0.3, 0.0, filtered_edges);
    //addWeighted(filtered_edges, alpha, text_image, beta, 0.0, sword);
    //imshow("test", filtered_edges);
    cv::waitKey(0);
    cv::imwrite("/home/luna/text.png",text_image);
    cv::imwrite("/home/luna/edge.png",filtered_edges);

    double alpha = 0.5; double beta;
    cv::Mat src1 = cv::imread("/home/luna/text.png");
    cv::Mat src2 = cv::imread("/home/luna/edge.png");
    cv::Mat dst;

    //cout << src2 <<endl;
    cv::Mat src2_resized;
    int width = src1.rows;
    int height = src1.cols;
    int depth = src1.channels();

    cout << width << endl;
    cout << height << endl;
    cout << depth << endl;

    resize(src2, src2_resized, cv::Size(width, height), cv::INTER_LINEAR);
    //cv::resize(src2, src2_resized, (width,height));
    beta = (1.0 - alpha);
    //addWeighted(src1, alpha, src2_resized, beta, 0, src1);
    imshow("mixed image",filtered_edges);
    cv::waitKey(0);

    return 0;
}
// 为什么这里addwidgeted总是报错，同样size同样channels.需要了解清楚。
