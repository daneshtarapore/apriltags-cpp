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
#include <stdio.h>
#include <getopt.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "CameraUtil.h"

#include "opencv2/opencv.hpp"
#include <string>     // std::string, std::to_string


#include <iomanip>
//#include <sstream>

using namespace cv;
using namespace std;



#define DEFAULT_TAG_FAMILY "Tag36h11"


void getEulerAngles(Mat &rotCamerMatrix, Mat& transVect, Vec3d &eulerAngles);
void RotationMatrixToEulerAngles(Mat &rotmatrix, double &theta1, double &theta2, double &theta3);

typedef struct CamTestOptions
{
    CamTestOptions() :
        params(),
        family_str(DEFAULT_TAG_FAMILY),
        error_fraction(1),
        device_num(0),
        focal_length(500),
        tag_size(0.1905),
        frame_width(0),
        frame_height(0),
        mirror_display(true)
    {
    }
    TagDetectorParams params;
    std::string family_str;
    double error_fraction;
    int device_num;
    double focal_length;
    double tag_size;
    int frame_width;
    int frame_height;
    bool mirror_display;
} CamTestOptions;


void print_usage(const char* tool_name, FILE* output=stderr)
{

    TagDetectorParams p;
    CamTestOptions o;

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
                                                           -z SIZE         Set the tag size in meters (default %f)\n\
                                                           -W WIDTH        Set the camera image width in pixels\n\
                                                           -H HEIGHT       Set the camera image height in pixels\n\
                                                           -M              Toggle display mirroring\n",
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
    for (size_t i = 0; i < known.size(); ++i) {
        fprintf(output, " %s", known[i].c_str());
    }
    fprintf(output, "\n");
}

CamTestOptions parse_options(int argc, char** argv)
{
    CamTestOptions opts;
    const char* options_str = "hDS:s:a:m:V:N:brnf:e:d:F:z:W:H:M";
    int c;
    while ((c = getopt(argc, argv, options_str)) != -1) {
        switch (c) {
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
        case 'W': opts.frame_width = atoi(optarg); break;
        case 'H': opts.frame_height = atoi(optarg); break;
        case 'M': opts.mirror_display = !opts.mirror_display; break;
        default:
            fprintf(stderr, "\n");
            print_usage(argv[0], stderr);
            exit(1);
        }
    }
    opts.params.adaptiveThresholdRadius += (opts.params.adaptiveThresholdRadius+1) % 2;
    return opts;
}

int main(int argc, char** argv)
{

    CamTestOptions opts = parse_options(argc, argv);

    TagFamily family(opts.family_str);

    if (opts.error_fraction >= 0 && opts.error_fraction <= 1)
    {
        family.setErrorRecoveryFraction(opts.error_fraction);
    }

    std::cout << "family.minimumHammingDistance = " << family.minimumHammingDistance << "\n";
    std::cout << "family.errorRecoveryBits = " << family.errorRecoveryBits << "\n";


    cv::VideoCapture vc;
    /*vc.open(opts.device_num);

    if (opts.frame_width && opts.frame_height)
    {
        // Use uvcdynctrl to figure this out dynamically at some point?
        vc.set(CV_CAP_PROP_FRAME_WIDTH, opts.frame_width);
        vc.set(CV_CAP_PROP_FRAME_HEIGHT, opts.frame_height);
    }

    std::cout << "Set camera to resolution: "
              << vc.get(CV_CAP_PROP_FRAME_WIDTH) << "x"
              << vc.get(CV_CAP_PROP_FRAME_HEIGHT) << "\n";*/


    std::string videoFilePath = "/home/millard/apriltags-cpp/videos/00001.MTS";

    try
    {
        //open the video file
        vc.open(videoFilePath); // open the video file
        if(!vc.isOpened())  // check if we succeeded
            CV_Error(CV_StsError, "Can not open Video file");

        std::cout << " FRAME_COUNT " << vc.get(CV_CAP_PROP_FRAME_COUNT);
    }
    catch( cv::Exception& e )
    {
        cerr << e.msg << endl;
        exit(1);
    }






    cv::Mat frame;
    cv::Point2d opticalCenter;

    vc >> frame;
    if (frame.empty())
    {
        std::cerr << "no frames!\n";
        exit(1);
    }

    opticalCenter.x = frame.cols * 0.5;
    opticalCenter.y = frame.rows * 0.5;

    std::string win = "video tag test";

    TagDetectorParams& params = opts.params;
    TagDetector detector(family, params);

    TagDetectionArray detections;

    int cvPose = 0;

    //while (1)
    for(int frameNum = 0; frameNum < vc.get(CV_CAP_PROP_FRAME_COUNT);frameNum++)
    {
        vc >> frame;
        if (frame.empty())
        {
            break;
        }

        cv::transpose(frame, frame);
	    cv::flip(frame, frame, 1);

        detector.process(frame, opticalCenter, detections);

        cv::Mat show = frame;

        if(!detections.empty())
        {
            //show = family.superimposeDetections(frame, detections);
            // show = frame;

            double s = opts.tag_size;
            double ss = 0.5*s;
            double sz = s;

            enum { npoints = 8, nedges = 12 };

            cv::Point3d src[npoints] =
            {
                cv::Point3d(-ss, -ss, 0),
                cv::Point3d( ss, -ss, 0),
                cv::Point3d( ss,  ss, 0),
                cv::Point3d(-ss,  ss, 0),
                cv::Point3d(-ss, -ss, sz),
                cv::Point3d( ss, -ss, sz),
                cv::Point3d( ss,  ss, sz),
                cv::Point3d(-ss,  ss, sz),
            };

            int edges[nedges][2] =
            {

                { 0, 1 },
                { 1, 2 },
                { 2, 3 },
                { 3, 0 },

                { 4, 5 },
                { 5, 6 },
                { 6, 7 },
                { 7, 4 },

                { 0, 4 },
                { 1, 5 },
                { 2, 6 },
                { 3, 7 }
            };

            cv::Point2d dst[npoints];

            double f = opts.focal_length;

            double K[9] = {
                f, 0, opticalCenter.x,
                0, f, opticalCenter.y,
                0, 0, 1
            };

            cv::Mat_<cv::Point3d> srcmat(npoints, 1, src);
            cv::Mat_<cv::Point2d> dstmat(npoints, 1, dst);

            cv::Mat_<double>      Kmat(3, 3, K);

            cv::Mat_<double>      distCoeffs = cv::Mat_<double>::zeros(4,1);

            std::cout << "Number of tags detected in frame " << detections.size() << std::endl;
            for (size_t i=0; i<detections.size(); ++i) {

                //for (cvPose=0; cvPose<2; ++cvPose) {
                if (1) {

                    cv::Mat r, t;



                    if (cvPose)
                    {


                        CameraUtil::homographyToPoseCV(f, f, s,
                                                       detections[i].homography,
                                                       r, t);

                    }
                    else
                    {

                        cv::Mat_<double> M =
                                CameraUtil::homographyToPose(f, f, s,
                                                             detections[i].homography,
                                                             false);

                        cv::Mat_<double> R = M.rowRange(0,3).colRange(0, 3);

                        t = M.rowRange(0,3).col(3);

                        cv::Rodrigues(R, r);
 
                        {
                            ostringstream Convert1,  Convert2,  Convert3;
                            Convert1 << fixed << setprecision(2) << (t.at<double>(0,0)); // Use some manipulators
                            string Result1 = Convert1.str(); // Give the result to the string
                            Convert2 << fixed << setprecision(2) << (t.at<double>(1,0)); // Use some manipulators
                            string Result2 = Convert2.str();
                            Convert3 << fixed << setprecision(2) << (t.at<double>(2,0)); // Use some manipulators
                            string Result3 = Convert3.str();
                            string Result = Result1 + ", " + Result2;
                            // string Result = Result1 + ',' + Result2 + ',' + Result3 ;

    						cv::putText(show, Result, detections[i].cxy, cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 255, 255), 2, CV_AA);
                        }




                        cv::Mat_<double> rotationmatrix, translationmatrix;
                        Vec3d eulerAngles;

                        rotationmatrix = R;
                        getEulerAngles(rotationmatrix, translationmatrix, eulerAngles);
                        //std::cout << "Detector id " << i << " t[0] " << t[0]  << " t[1] " << t[1]  << " t[2] " << t[2] << std::endl;
                        std::cout << "Index " << i << std::endl;
                        std::cout << "Detector observed code " << detections[i].obsCode << " matched code " << detections[i].code << " id " << detections[i].id << std::endl;
                        //std::cout << "rotation to align code " << detections[i].rotation << std::endl;
                        std::cout << "tag center in pixel coordinates" << detections[i].cxy << std::endl;// << "  Position (in fractional pixel coordinates) of the detection " <<
                        //              detections[i].p[0] << " " << detections[i].p[1] << " " << detections[i].p[2] <<  " " << detections[i].p[3] << std::endl;
                        //std::cout << "The homography is relative to image center, whose coordinates are "  << detections[i].hxy << std::endl;


                        //std::cout << "Pose matrix "  << M << std::endl;


                        std::cout << " Rotation Angles (pitch, yaw, roll)" << eulerAngles << std::endl;
                        std::cout << " position  " << t << std::endl;

                        double theta1 = 0.0f, theta2 = 0.0f,theta3 = 0.0f;
                        cv::Mat_<double> rotationmatrix1;
                        rotationmatrix1 = R;
                        RotationMatrixToEulerAngles(rotationmatrix, theta1, theta2, theta3);
                        /*std::cout << " RotationMatrixToEulerAngles: " << theta1 << " " << theta2 << " " << theta3 << std::endl;*/

                        {
                            ostringstream Convert1,  Convert2,  Convert3;
                            Convert1 << fixed << setprecision(2) << (eulerAngles[0]); // Use some manipulators
                            string Result1 = Convert1.str(); // Give the result to the string
                            Convert2 << fixed << setprecision(2) << (eulerAngles[1]); // Use some manipulators
                            string Result2 = Convert2.str();
                            Convert3 << fixed << setprecision(2) << (eulerAngles[2]); // Use some manipulators
                            string Result3 = Convert3.str();
                            string Result = Result1 + ',' + Result2 + ',' + Result3 ;



                            /*cv::putText(show, Result, detections[i].cxy, cv::FONT_HERSHEY_SIMPLEX,
                                    1.6, CV_RGB(0,0,0), 3, CV_AA);*/
                        }


                    }

                    cv::projectPoints(srcmat, r, t, Kmat, distCoeffs, dstmat);

                    // for (int j=0; j<nedges; ++j)
                    // {
                    //     cv::line(show,
                    //              dstmat(edges[j][0],0),
                    //             dstmat(edges[j][1],0),
                    //             cvPose ? CV_RGB(0,255,0) : CV_RGB(255,0,0),
                    //             1, CV_AA);
                    // }

                }

            }


        }

        // cv::transpose(show, show);

        // if (opts.mirror_display) {
        //     cv::flip(show, show, 1);
        // }

        resize(show, show, Size(720, 1280), 0, 0, INTER_CUBIC);

        cv::imshow(win, show);
        int k = cv::waitKey(5);
        if (k % 256 == 's') {
            cv::imwrite("frame.png", frame);
            std::cout << "wrote frame.png\n";
        } else if (k % 256 == 'p') {
            cvPose = !cvPose;
        } else if (k % 256 == 27 /* ESC */) {
            break;
        }

    }

    detector.reportTimers();

    return 0;
}

void getEulerAngles(Mat &rotCamerMatrix, Mat& transVect, Vec3d &eulerAngles){

    Mat cameraMatrix,rotMatrix,rotMatrixX,rotMatrixY,rotMatrixZ;
    double* _r = rotCamerMatrix.ptr<double>();
    double projMatrix[12] = {_r[0],_r[1],_r[2],0,
                          _r[3],_r[4],_r[5],0,
                          _r[6],_r[7],_r[8],0};

    decomposeProjectionMatrix( Mat(3,4,CV_64FC1,projMatrix),
                               cameraMatrix,
                               rotMatrix,
                               transVect,
                               rotMatrixX,
                               rotMatrixY,
                               rotMatrixZ,
                               eulerAngles);
}


void RotationMatrixToEulerAngles(Mat &rotmatrix, double &theta1, double &theta2, double &theta3)
{
    double m00, m01, m02, m10, m11, m12, m20, m21, m22;
    double* _r = rotmatrix.ptr<double>();

    m00 = _r[0]; m01 = _r[1]; m02 = _r[2];
    m10 = _r[3]; m11 = _r[4]; m12 = _r[5];
    m20 = _r[6]; m21 = _r[7]; m22 = _r[8];

    theta1 = atan2(m12, m22);
    double c2 = sqrt( (m00*m00)  +  (m01*m01));
    theta2 = atan2(-m02, c2);
    double s1 = sin(theta1); double c1 = cos(theta1);
    theta3 = atan2( s1*m20 - c1*m10, c1*m11 - s1*m21);

    theta1 *= 180.0f/3.142; theta2 *= 180.0f/3.142; theta3 *= 180.0f/3.142;


    //std::cout << "RotMatToAngles: " << theta1 << " " << theta2 << " " << theta3 << std::endl;



    theta1 = atan2(m21, m22) * 180.0f/3.142;
    theta2 = atan2(-m20, sqrt( (m21*m21) + (m22*m22))) * 180.0f/3.142;
    theta3 = atan2(m10, m00) * 180.0f/3.142;

    //std::cout << "RotMatToAngles: " << theta1 << " " << theta2 << " " << theta3 <<  std::endl << std::endl << std::endl;



}
