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
            tag_size(0.07637)  // This is the length of the edge of a tag: 0.07637 = 76.37 mm for Psi Swarm, 0.05233 = 52.33 mm for e-puck
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

int main(int argc, char** argv)
{
    AprilTagOptions opts = parse_options(argc, argv);

    TagFamily family(opts.family_str);

    if(opts.error_fraction >= 0 && opts.error_fraction <= 1)
        family.setErrorRecoveryFraction(opts.error_fraction);

    std::cout << "family.minimumHammingDistance = " << family.minimumHammingDistance << "\n";
    std::cout << "family.errorRecoveryBits = " << family.errorRecoveryBits << "\n";

    cv::VideoCapture vc;
    std::string videoFilePath = "/home/danesh/work/apriltags-cpp/videos/00002.MTS";

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

    cv::transpose(frame, frame);
    cv::flip(frame, frame, 1);

    opticalCenter.x = frame.cols * 0.5;
    opticalCenter.y = frame.rows * 0.5;

    std::string win = "AprilTags optimised";

    TagDetectorParams& params = opts.params;
    TagDetector detector(family, params);
  
    TagDetectionArray detections;
  
    while(true)
    {
        vc >> frame;
        cv::transpose(frame, frame);
        cv::flip(frame, frame, 1);

        if(frame.empty())
            break;

        detector.process(frame, opticalCenter, detections);

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

        cv::resize(show, show, cv::Size(720, 1280), 0, 0, cv::INTER_CUBIC);
        cv::imshow(win, show);

        int k = cv::waitKey(1);

        if(k % 256 == 27) // Escape key
            break;
    }

    detector.reportTimers();
    return 0;
}
