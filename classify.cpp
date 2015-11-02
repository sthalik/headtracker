#include "ht-internal.h"

bool classifier::classify(const cv::Mat& frame, cv::Rect& ret) {
    std::vector<cv::Rect> seq;
    head_cascade.detectMultiScale(frame,
                                  seq,
                                  1.2,
                                  2,
                                  CV_HAAR_DO_CANNY_PRUNING | CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_DO_ROUGH_SEARCH,
                                  cv::Size(60, 60));

    if (seq.size() > 0)
        ret = seq[0];

    return seq.size() > 0;
}
