/*********************************************************************
 * This file is distributed as part of the C++ port of the APRIL tags
 * library. The code is licensed under GPLv2.
 *
 * Original author: Edwin Olson <ebolson@umich.edu>
 * C++ port and modifications: Matt Zucker <mzucker1@swarthmore.edu>
 ********************************************************************/

#include "TagDetector.h"

#include <sys/time.h>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <getopt.h>
#include <math.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/aruco.hpp>

#include "CameraUtil.h"

#define DEFAULT_TAG_FAMILY "Tag36h11"

typedef struct AprilTagOptions
{
    AprilTagOptions():
            params(),
            family_str(DEFAULT_TAG_FAMILY),
            error_fraction(1),
            device_num(0),
            focal_length(500), // TODO: Find out the focal length of James' camera lens, and convert it to pixels based on CCD dimensions
            tag_size(0.05233)  // This is the length of the edge of a tag: 0.07637 = 76.37 mm for Psi Swarm, 0.05233 = 52.33 mm for e-puck
            {}

    TagDetectorParams params;
    std::string family_str;
    double error_fraction;
    int device_num;
    double focal_length;
    double tag_size;
    int frame_width;
    int frame_height;

} AprilTagOptions;

void print_usage(const char* tool_name, FILE* output = stderr)
{
    TagDetectorParams p;
    AprilTagOptions o;

    fprintf(output, "\
        Usage: %s [OPTIONS]\n\
        Run a tool to test tag detection. Options:\n\
         -h              Show this help message.\n\
         -D              Use decimation for segmentation stage.\n\
         -S SIGMA        Set the original image sigma value (default %.2f).\n\
         -s SEGSIGMA     Set the segmentation sigma value (default %.2f).\n\
         -a THETATHRESH  Set the theta threshold for clustering (default %.1f).\n\
         -m MAGTHRESH    Set the magnitude threshold for clustering (default %.1f).\n\
         -V VALUE        Set adaptive threshold value for new quad algo (default %f).\n\
         -N RADIUS       Set adaptive threshold radius for new quad algo (default %d).\n\
         -b              Refine bad quads using template tracker.\n\
         -r              Refine all quads using template tracker.\n\
         -n              Use the new quad detection algorithm.\n\
         -f FAMILY       Look for the given tag family (default \"%s\")\n\
         -e FRACTION     Set error detection fraction (default %f)\n\
         -d DEVICE       Set camera device number (default %d)\n\
         -F FLENGTH      Set the camera's focal length in pixels (default %f)\n\
         -z SIZE         Set the tag size in meters (default %f)\n",
        tool_name,
        p.sigma,
        p.segSigma,
        p.thetaThresh,
        p.magThresh,
        p.adaptiveThresholdValue,
        p.adaptiveThresholdRadius,
        DEFAULT_TAG_FAMILY,
        o.error_fraction,
        o.device_num,
        o.focal_length,
        o.tag_size);

    fprintf(output, "Known tag families:");

    TagFamily::StringArray known = TagFamily::families();

    for(size_t i = 0; i < known.size(); ++i)
        fprintf(output, " %s", known[i].c_str());

    fprintf(output, "\n");
}

AprilTagOptions parse_options(int argc, char** argv)
{
    AprilTagOptions opts;
    const char* options_str = "hDS:s:a:m:V:N:brnf:e:d:F:z:W:H:M";
    int c;

    while((c = getopt(argc, argv, options_str)) != -1)
    {
        switch(c)
        {
            // Reminder: add new options to 'options_str' above and print_usage()!
            case 'h': print_usage(argv[0], stdout); exit(0); break;
            case 'D': opts.params.segDecimate = true; break;
            case 'S': opts.params.sigma = atof(optarg); break;
            case 's': opts.params.segSigma = atof(optarg); break;
            case 'a': opts.params.thetaThresh = atof(optarg); break;
            case 'm': opts.params.magThresh = atof(optarg); break;
            case 'V': opts.params.adaptiveThresholdValue = atof(optarg); break;
            case 'N': opts.params.adaptiveThresholdRadius = atoi(optarg); break;
            case 'b': opts.params.refineBad = true; break;
            case 'r': opts.params.refineQuads = true; break;
            case 'n': opts.params.newQuadAlgorithm = true; break;
            case 'f': opts.family_str = optarg; break;
            case 'e': opts.error_fraction = atof(optarg); break;
            case 'd': opts.device_num = atoi(optarg); break;
            case 'F': opts.focal_length = atof(optarg); break;
            case 'z': opts.tag_size = atof(optarg); break;
            default:
                fprintf(stderr, "\n");
                print_usage(argv[0], stderr);
                exit(1);
        }
    }

    opts.params.adaptiveThresholdRadius += (opts.params.adaptiveThresholdRadius+1) % 2;
    return opts;
}

// From: http://answers.opencv.org/question/16796/computing-attituderoll-pitch-yaw-from-solvepnp
void getEulerAngles(cv::Mat &rotCamerMatrix, cv::Vec3d &eulerAngles)
{
    cv::Mat cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ;
    double* _r = rotCamerMatrix.ptr<double>();
    double projMatrix[12] = {_r[0],_r[1],_r[2],0,
                             _r[3],_r[4],_r[5],0,
                             _r[6],_r[7],_r[8],0};

    cv::decomposeProjectionMatrix(cv::Mat(3, 4, CV_64FC1, projMatrix),
                                  cameraMatrix,
                                  rotMatrix,
                                  transVect,
                                  rotMatrixX,
                                  rotMatrixY,
                                  rotMatrixZ,
                                  eulerAngles);
}

using namespace cv;
using namespace std;

/**
  * @brief Convert input image to gray if it is a 3-channels image
  */
static void _convertToGrey(InputArray _in, OutputArray _out) {

    CV_Assert(_in.getMat().channels() == 1 || _in.getMat().channels() == 3);

    _out.create(_in.getMat().size(), CV_8UC1);
    if(_in.getMat().type() == CV_8UC3)
        cvtColor(_in.getMat(), _out.getMat(), COLOR_BGR2GRAY);
    else
        _in.getMat().copyTo(_out);
}

/**
  * @brief Threshold input image using adaptive thresholding
  */
static void _threshold(InputArray _in, OutputArray _out, int winSize, double constant) {

    CV_Assert(winSize >= 3);
    if(winSize % 2 == 0) winSize++; // win size must be odd
    adaptiveThreshold(_in, _out, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, winSize, constant);
}

/**
  * @brief Given a tresholded image, find the contours, calculate their polygonal approximation
  * and take those that accomplish some conditions
  */
static void _findMarkerContours(InputArray _in, vector< vector< Point2f > > &candidates,
                                vector< vector< Point > > &contoursOut, double minPerimeterRate,
                                double maxPerimeterRate, double accuracyRate,
                                double minCornerDistanceRate, int minDistanceToBorder) {

    CV_Assert(minPerimeterRate > 0 && maxPerimeterRate > 0 && accuracyRate > 0 &&
              minCornerDistanceRate >= 0 && minDistanceToBorder >= 0);

    // calculate maximum and minimum sizes in pixels
    unsigned int minPerimeterPixels =
            (unsigned int)(minPerimeterRate * max(_in.getMat().cols, _in.getMat().rows));
    unsigned int maxPerimeterPixels =
            (unsigned int)(maxPerimeterRate * max(_in.getMat().cols, _in.getMat().rows));

    Mat contoursImg;
    _in.getMat().copyTo(contoursImg);
    vector< vector< Point > > contours;
    findContours(contoursImg, contours, RETR_LIST, CHAIN_APPROX_NONE);
    // now filter list of contours
    for(unsigned int i = 0; i < contours.size(); i++) {
        // check perimeter
        if(contours[i].size() < minPerimeterPixels || contours[i].size() > maxPerimeterPixels)
            continue;

        // check is square and is convex
        vector< Point > approxCurve;
        approxPolyDP(contours[i], approxCurve, double(contours[i].size()) * accuracyRate, true);
        if(approxCurve.size() != 4 || !isContourConvex(approxCurve)) continue;

        // check min distance between corners
        double minDistSq =
                max(contoursImg.cols, contoursImg.rows) * max(contoursImg.cols, contoursImg.rows);
        for(int j = 0; j < 4; j++) {
            double d = (double)(approxCurve[j].x - approxCurve[(j + 1) % 4].x) *
                       (double)(approxCurve[j].x - approxCurve[(j + 1) % 4].x) +
                       (double)(approxCurve[j].y - approxCurve[(j + 1) % 4].y) *
                       (double)(approxCurve[j].y - approxCurve[(j + 1) % 4].y);
            minDistSq = min(minDistSq, d);
        }
        double minCornerDistancePixels = double(contours[i].size()) * minCornerDistanceRate;
        if(minDistSq < minCornerDistancePixels * minCornerDistancePixels) continue;

        // check if it is too near to the image border
        bool tooNearBorder = false;
        for(int j = 0; j < 4; j++) {
            if(approxCurve[j].x < minDistanceToBorder || approxCurve[j].y < minDistanceToBorder ||
               approxCurve[j].x > contoursImg.cols - 1 - minDistanceToBorder ||
               approxCurve[j].y > contoursImg.rows - 1 - minDistanceToBorder)
                tooNearBorder = true;
        }
        if(tooNearBorder) continue;

        // if it passes all the test, add to candidates vector
        vector< Point2f > currentCandidate;
        currentCandidate.resize(4);
        for(int j = 0; j < 4; j++) {
            currentCandidate[j] = Point2f((float)approxCurve[j].x, (float)approxCurve[j].y);
        }
        candidates.push_back(currentCandidate);
        contoursOut.push_back(contours[i]);
    }
}

/**
  * ParallelLoopBody class for the parallelization of the basic candidate detections using
  * different threhold window sizes. Called from function _detectInitialCandidates()
  */
class DetectInitialCandidatesParallel : public ParallelLoopBody {
public:
    DetectInitialCandidatesParallel(const Mat *_grey,
                                    vector< vector< vector< Point2f > > > *_candidatesArrays,
                                    vector< vector< vector< Point > > > *_contoursArrays,
                                    const Ptr<aruco::DetectorParameters> &_params)
            : grey(_grey), candidatesArrays(_candidatesArrays), contoursArrays(_contoursArrays),
              params(_params) {}

    void operator()(const Range &range) const {
        const int begin = range.start;
        const int end = range.end;

        for(int i = begin; i < end; i++) {
            int currScale =
                    params->adaptiveThreshWinSizeMin + i * params->adaptiveThreshWinSizeStep;
            // threshold
            Mat thresh;
            _threshold(*grey, thresh, currScale, params->adaptiveThreshConstant);

            // detect rectangles
            _findMarkerContours(thresh, (*candidatesArrays)[i], (*contoursArrays)[i],
                                params->minMarkerPerimeterRate, params->maxMarkerPerimeterRate,
                                params->polygonalApproxAccuracyRate, params->minCornerDistanceRate,
                                params->minDistanceToBorder);
        }
    }

private:
    DetectInitialCandidatesParallel &operator=(const DetectInitialCandidatesParallel &);

    const Mat *grey;
    vector< vector< vector< Point2f > > > *candidatesArrays;
    vector< vector< vector< Point > > > *contoursArrays;
    const Ptr<aruco::DetectorParameters> &params;
};

/**
 * @brief Initial steps on finding square candidates
 */
static void _detectInitialCandidates(const Mat &grey, vector< vector< Point2f > > &candidates,
                                     vector< vector< Point > > &contours,
                                     const Ptr<aruco::DetectorParameters> &params) {

    CV_Assert(params->adaptiveThreshWinSizeMin >= 3 && params->adaptiveThreshWinSizeMax >= 3);
    CV_Assert(params->adaptiveThreshWinSizeMax >= params->adaptiveThreshWinSizeMin);
    CV_Assert(params->adaptiveThreshWinSizeStep > 0);

    // number of window sizes (scales) to apply adaptive thresholding
    int nScales =  (params->adaptiveThreshWinSizeMax - params->adaptiveThreshWinSizeMin) /
                   params->adaptiveThreshWinSizeStep + 1;

    vector< vector< vector< Point2f > > > candidatesArrays((size_t) nScales);
    vector< vector< vector< Point > > > contoursArrays((size_t) nScales);

    ////for each value in the interval of thresholding window sizes
    // for(int i = 0; i < nScales; i++) {
    //    int currScale = params.adaptiveThreshWinSizeMin + i*params.adaptiveThreshWinSizeStep;
    //    // treshold
    //    Mat thresh;
    //    _threshold(grey, thresh, currScale, params.adaptiveThreshConstant);
    //    // detect rectangles
    //    _findMarkerContours(thresh, candidatesArrays[i], contoursArrays[i],
    // params.minMarkerPerimeterRate,
    //                        params.maxMarkerPerimeterRate, params.polygonalApproxAccuracyRate,
    //                        params.minCornerDistance, params.minDistanceToBorder);
    //}

    // this is the parallel call for the previous commented loop (result is equivalent)
    parallel_for_(Range(0, nScales), DetectInitialCandidatesParallel(&grey, &candidatesArrays,
                                                                     &contoursArrays, params));

    // join candidates
    for(int i = 0; i < nScales; i++) {
        for(unsigned int j = 0; j < candidatesArrays[i].size(); j++) {
            candidates.push_back(candidatesArrays[i][j]);
            contours.push_back(contoursArrays[i][j]);
        }
    }
}

/**
  * @brief Assure order of candidate corners is clockwise direction
  */
static void _reorderCandidatesCorners(vector< vector< Point2f > > &candidates) {

    for(unsigned int i = 0; i < candidates.size(); i++) {
        double dx1 = candidates[i][1].x - candidates[i][0].x;
        double dy1 = candidates[i][1].y - candidates[i][0].y;
        double dx2 = candidates[i][2].x - candidates[i][0].x;
        double dy2 = candidates[i][2].y - candidates[i][0].y;
        double crossProduct = (dx1 * dy2) - (dy1 * dx2);

        if(crossProduct < 0.0) { // not clockwise direction
            swap(candidates[i][1], candidates[i][3]);
        }
    }
}

/**
  * @brief Check candidates that are too close to each other and remove the smaller one
  */
static void _filterTooCloseCandidates(const vector< vector< Point2f > > &candidatesIn,
                                      vector< vector< Point2f > > &candidatesOut,
                                      const vector< vector< Point > > &contoursIn,
                                      vector< vector< Point > > &contoursOut,
                                      double minMarkerDistanceRate) {

    CV_Assert(minMarkerDistanceRate >= 0);

    vector< pair< int, int > > nearCandidates;
    for(unsigned int i = 0; i < candidatesIn.size(); i++) {
        for(unsigned int j = i + 1; j < candidatesIn.size(); j++) {

            int minimumPerimeter = min((int)contoursIn[i].size(), (int)contoursIn[j].size() );

            // fc is the first corner considered on one of the markers, 4 combinations are possible
            for(int fc = 0; fc < 4; fc++) {
                double distSq = 0;
                for(int c = 0; c < 4; c++) {
                    // modC is the corner considering first corner is fc
                    int modC = (c + fc) % 4;
                    distSq += (candidatesIn[i][modC].x - candidatesIn[j][c].x) *
                              (candidatesIn[i][modC].x - candidatesIn[j][c].x) +
                              (candidatesIn[i][modC].y - candidatesIn[j][c].y) *
                              (candidatesIn[i][modC].y - candidatesIn[j][c].y);
                }
                distSq /= 4.;

                // if mean square distance is too low, remove the smaller one of the two markers
                double minMarkerDistancePixels = double(minimumPerimeter) * minMarkerDistanceRate;
                if(distSq < minMarkerDistancePixels * minMarkerDistancePixels) {
                    nearCandidates.push_back(pair< int, int >(i, j));
                    break;
                }
            }
        }
    }

    // mark smaller one in pairs to remove
    vector< bool > toRemove(candidatesIn.size(), false);
    for(unsigned int i = 0; i < nearCandidates.size(); i++) {
        // if one of the marker has been already markerd to removed, dont need to do anything
        if(toRemove[nearCandidates[i].first] || toRemove[nearCandidates[i].second]) continue;
        size_t perimeter1 = contoursIn[nearCandidates[i].first].size();
        size_t perimeter2 = contoursIn[nearCandidates[i].second].size();
        if(perimeter1 > perimeter2)
            toRemove[nearCandidates[i].second] = true;
        else
            toRemove[nearCandidates[i].first] = true;
    }

    // remove extra candidates
    candidatesOut.clear();
    unsigned long totalRemaining = 0;
    for(unsigned int i = 0; i < toRemove.size(); i++)
        if(!toRemove[i]) totalRemaining++;
    candidatesOut.resize(totalRemaining);
    contoursOut.resize(totalRemaining);
    for(unsigned int i = 0, currIdx = 0; i < candidatesIn.size(); i++) {
        if(toRemove[i]) continue;
        candidatesOut[currIdx] = candidatesIn[i];
        contoursOut[currIdx] = contoursIn[i];
        currIdx++;
    }
}

/**
 * @brief Detect square candidates in the input image
 */
static void _detectCandidates(InputArray _image, OutputArrayOfArrays _candidates,
                              OutputArrayOfArrays _contours, const Ptr<aruco::DetectorParameters> &_params) {

    Mat image = _image.getMat();
    CV_Assert(image.total() != 0);

    /// 1. CONVERT TO GRAY
    Mat grey;
    _convertToGrey(image, grey);

    vector <vector<Point2f> > candidates;
    vector <vector<Point> > contours;
    /// 2. DETECT FIRST SET OF CANDIDATES
    _detectInitialCandidates(grey, candidates, contours, _params);

    /// 3. SORT CORNERS
    _reorderCandidatesCorners(candidates);

    /// 4. FILTER OUT NEAR CANDIDATE PAIRS
    vector <vector<Point2f> > candidatesOut;
    vector <vector<Point> > contoursOut;
    _filterTooCloseCandidates(candidates, candidatesOut, contours, contoursOut,
                              _params->minMarkerDistanceRate);

    // parse output
    _candidates.create((int) candidatesOut.size(), 1, CV_32FC2);
    _contours.create((int) contoursOut.size(), 1, CV_32SC2);
    for (int i = 0; i < (int) candidatesOut.size(); i++) {
        _candidates.create(4, 1, CV_32FC2, i, true);
        Mat m = _candidates.getMat(i);
        for (int j = 0; j < 4; j++)
            m.ptr<Vec2f>(0)[j] = candidatesOut[i][j];

        _contours.create((int) contoursOut[i].size(), 1, CV_32SC2, i, true);
        Mat c = _contours.getMat(i);
        for (unsigned int j = 0; j < contoursOut[i].size(); j++)
            c.ptr<Point2i>()[j] = contoursOut[i][j];
    }
}

int main(int argc, char** argv)
{
    AprilTagOptions opts = parse_options(argc, argv);

    TagFamily family(opts.family_str);

    if(opts.error_fraction >= 0 && opts.error_fraction <= 1)
        family.setErrorRecoveryFraction(opts.error_fraction);

    std::cout << "family.minimumHammingDistance = " << family.minimumHammingDistance << "\n";
    std::cout << "family.errorRecoveryBits = " << family.errorRecoveryBits << "\n";

    cv::VideoCapture vc;
//    std::string videoFilePath = "/mnt/hgfs/millard/Desktop/epucks.MTS";
    std::string videoFilePath = "/mnt/hgfs/millard/Desktop/aruco.MTS";

    try
    {
        vc.open(videoFilePath); // open the video file

        if(!vc.isOpened())  // check if we succeeded
            CV_Error(CV_StsError, "Cannot open video file");

        std::cout << "Frame count: " << vc.get(CV_CAP_PROP_FRAME_COUNT) << std::endl;
    }
    catch(cv::Exception& e)
    {
        std::cerr << e.msg << std::endl;
        exit(1);
    }

    std::cout << "Set camera to resolution: "
              << vc.get(CV_CAP_PROP_FRAME_WIDTH) << "x"
              << vc.get(CV_CAP_PROP_FRAME_HEIGHT) << "\n";

    cv::Mat frame;
    cv::Point2d opticalCenter;

    vc >> frame;

    if(frame.empty())
    {
        std::cerr << "no frames!\n";
        exit(1);
    }

//    cv::transpose(frame, frame);
//    cv::flip(frame, frame, 1);

    opticalCenter.x = frame.cols * 0.5;
    opticalCenter.y = frame.rows * 0.5;

    std::string win = "AprilTags optimised";

    TagDetectorParams& params = opts.params;
    TagDetector detector(family, params);
  
    TagDetectionArray detections;
  
    while(true)
    {
        vc >> frame;
//        cv::transpose(frame, frame);
//        cv::flip(frame, frame, 1);

        if(frame.empty())
            break;

//        cv::Mat grey;
//        _convertToGrey(frame, grey);
//
//        std::vector<std::vector<cv::Point2f> > candidates;
//        std::vector<std::vector<cv::Point> > contours;
//        cv::Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();
//
//        _detectCandidates(grey, candidates, contours, params);
//        aruco::drawDetectedMarkers(frame, candidates, noArray(), Scalar(100, 0, 255));

        std::vector<int> markerIds;
        std::vector<std::vector<Point2f> > markerCorners, rejectedCandidates;
        cv::Ptr<aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();
        cv::Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50);
        cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

        aruco::drawDetectedMarkers(frame, markerCorners, markerIds);

        cv::Mat show;

        if(detections.empty())
            show = frame;
        else
        {
            show = family.superimposeDetections(frame, detections);

            double s = opts.tag_size;
            double f = opts.focal_length;

            for(size_t i = 0; i < detections.size(); ++i)
            {
                cv::Mat r, t;

                CameraUtil::homographyToPoseCV(f, f, s,
                                               detections[i].homography,
                                               r, t);

                cv::Mat R;
                cv::Rodrigues(r, R);

                cv::Vec3d eulerAngles;
                getEulerAngles(R, eulerAngles);

                float bottomlineangle = std::atan2(detections[i].p[1].y - detections[i].p[0].y, detections[i].p[1].x - detections[i].p[0].x)* 180.0f / M_PI; // atan2 returns angle in radians [-pi, pi]
                if(std::isnan(bottomlineangle))
                {
                    std::cout << " angle is nan for tag points " << detections[i].p << std::endl;
                    exit(-1);
                }

                std::ostringstream x, y, orientation, taglineangle;
                x << std::fixed << std::setprecision(2) << detections[i].cxy.x;
                y << std::fixed << std::setprecision(2) << detections[i].cxy.y;
                orientation << std::fixed << std::setprecision(2) << eulerAngles[2]; // roll
                taglineangle << std::fixed << std::setprecision(2) << bottomlineangle;
                std::string text = "x: " + x.str() + ", y: " + y.str() + ", a: " + orientation.str() + ", b: " + taglineangle.str();



                cv::putText(show,
                            text,
                            detections[i].cxy,
                            cv::FONT_HERSHEY_SIMPLEX,
                            1,
                            CV_RGB(255, 255, 255),
                            2,
                            CV_AA);

                // always points to the bottom-left corner of the tag
                cv::putText(show,
                            "x",
                            detections[i].p[0],
                            cv::FONT_HERSHEY_SIMPLEX,
                            1,
                            CV_RGB(255, 0, 0),
                            2,
                            CV_AA);

                // always points to the bottom-right corner of the tag
                cv::putText(show,
                            "y",
                            detections[i].p[1],
                            cv::FONT_HERSHEY_SIMPLEX,
                            1,
                            CV_RGB(255, 0, 0),
                            2,
                            CV_AA);
            }
        }

//        cv::resize(show, show, cv::Size(720, 1280), 0, 0, cv::INTER_CUBIC);
        cv::resize(show, show, cv::Size(1280, 720), 0, 0, cv::INTER_CUBIC);
        cv::imshow(win, show);

        int k = cv::waitKey(1);

        if(k % 256 == 27) // Escape key
            break;
    }

    detector.reportTimers();
    return 0;
}
