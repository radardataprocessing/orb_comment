/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>


using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

/*
 * if the image has 3 or 4 channels, convert them to gray images
 * use the left and right images and the correspondding extractors, parameters as well as thresholds tp construct a frame
 * 
 */
cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        // ORBextractor* mpIniORBextractor; 
        // mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        // ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
        // mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

/*
 * 
 */
void Tracking::Track()
{
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }
    /*
    the possible values for mState
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };
    */

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            MonocularInitialization();

        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        // mbOnlyTracking is true if local mapping is deactivated and we are performing only localization
        if(!mbOnlyTracking)// mapping mode: local mapping is activated
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                // cv::Mat mVelocity;  representes the motion model
                // if the motion model is unknown or less than two frames away from the lastreloc frame 
                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    // track the current frame against the reference frame, if the matched map points exceeds a given threshold, bOK is true
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else
            {
                bOK = Relocalization();
            }
        }
        else// localization mode: local mapping is deactivated
        {
            // Localization Mode: Local Mapping is deactivated

            if(mState==LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                /* In case of localization mode, this flag is true when there are no matches to points in the map. Still tracking will continue if there are
                   enough matches with temporal points. In that case we are doing visual odometry. The system will try to do relocalization to recover 
                   "zero-drift" localization to the map */
                if(!mbVO) // !mbVO means that we matched enough mappoints in the map
                {
                    // In last frame we tracked enough MapPoints in the map

                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else // mbVO is true, means that we can not match enough mappoints in the map, we will track vo temporal points
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    /*  */
                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if(!mVelocity.empty())// if the motion model is not empty, track with motion model
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    // after tracking with motion model, do the relocalization
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)// if tracking with motion model is sucess and relocalization is not success
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        // if mbVO is true, for every keypoint in current frame, if it is not a null pointer and it is not marked as an outlier, increase
                        // its found times by 1
                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)// if relocalization is success, set mbVO to false
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;// if bOKReloc or bOKMM is true, then bOK is true       MM here means motion model
                }
            }
        }// end the assume for localization mode

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)// mapping mode: local mapping is activated
        {
            if(bOK)// is bOK is true
                bOK = TrackLocalMap();
        }
        else// localization mode
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            /*
             * in localization mode, if mbVO is true, means we can not localize to the map, so we cannot retrieve a local map and therefore we do not
             * perform TrackLocalMap(). Once the system relocalizes the camera, we will use the local map again
             */
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if(bOK)// if tracking was good
        {
            // Update motion model
            if(!mLastFrame.mTcw.empty())// if the pose of last frame is not empty
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                // mVelocity = T_world_to_current * T_last_to_world = T_last_to_current
                mVelocity = mCurrentFrame.mTcw*LastTwc;// the rotation matrix from the last to the current 
            }
            else// if the pose of last frame is empty, set mVelocity to empty mat
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches for every keypoint in mCurrentFrame
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];// get the mappoint
                if(pMP)// if the mappoint is not a null pointer
                    if(pMP->Observations()<1)// if observation of the mappoint is less than 1
                    {
                        mCurrentFrame.mvbOutlier[i] = false;//set the mvbOutlier of the mappoint to false
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);// set the mappoint tobe null pointer
                    }
            }

            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();//clear the list mlpTemporalPoints

            // Check if we need to insert a new keyframe
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function) pass to the new keyframe, so that bundle adjustment
            // will finally decide if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            /* for all keypoints in mCurrentFrame, if the mappoint is not null and is marked as an outlier, set the mappoint to null */
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST) // if the mState is LOST
        {
            if(mpMap->KeyFramesInMap()<=5) //if keyframes in the map is no more than 5
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();// reset the system
                return;// return from this function
            }
        }

        if(!mCurrentFrame.mpReferenceKF)// if the mpReferenceKF of mCurrentFrame is null, set it tobe mpReferenceKF
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);// construct a frame using mCurrentFrame to set mLastFrame
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())// if the pose of mCurrentFrame is not empty
    {
        //GetPoseInverse() returns Twc   world_to_current*referencce_to_world = reference_to_current 
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else// if the pose of mCurrentFrame is empty
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}


/*
 * 1. if the number of keypoints in the left image is no more than 500, return from this function
 * 2. set the pos of current frame to 0, create a keyframe as the keyframe ini using the current frame, add keyframe ini to map
 * 3. for all keypoints in the current frame, get their depth, if the depth is bigger than 0, compute the 3D position of the keypoint in the world coordinate, create 
 *    a mappoint using the 3D point,compute the descriptor of the mappoint as well as the normal, the max and min distance; then add the point to map
 * 4. set the related variables 
 */
void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)// number of keypoints in the left image more than 500
    {
        // Set Frame pose to the origin    Frame mCurrentFrame
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame    Map* mpMap;  KeyFrameDatabase* mpKeyFrameDB
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // Insert KeyFrame in the map  Map* mpMap
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and associate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)// for all left image keypoints in the current frame
        {
            float z = mCurrentFrame.mvDepth[i];// get the depth of the keypoint
            if(z>0)//if the depth is bigger than 0
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);//compute the 3D position of the keypoint in the world coordinate
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);// this pNewMP mappoint is observed by the keyframe pKFini as the ith keypoint
                pKFini->AddMapPoint(pNewMP,i);// this pKFini keyframe has a ith keypoint corresponding to the map point pNewMP 
                pNewMP->ComputeDistinctiveDescriptors();//compute the descriptor for the mappoint
                //compute the normal using all keyframes observe this mappoint, and compute the maxdistance and mindistance using the reference keyframe
                pNewMP->UpdateNormalAndDepth(); 
                mpMap->AddMapPoint(pNewMP);// insert the mappoint to the set in the map class

                mCurrentFrame.mvpMapPoints[i]=pNewMP;// put the mappoint to the vector in the curent keyframe
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;//get the number of map points in the map

        mpLocalMapper->InsertKeyFrame(pKFini);// insert the key frame to localmapper

        mLastFrame = Frame(mCurrentFrame);// use the copy constructor of class Frame to set last frame the same as current frame
        mnLastKeyFrameId=mCurrentFrame.mnId;// set the last key frame id to current frame id
        mpLastKeyFrame = pKFini;//set the last keyframe to keyframe ini

        mvpLocalKeyFrames.push_back(pKFini);//add the keyframe ini to local keyframes vector
        mvpLocalMapPoints=mpMap->GetAllMapPoints();// get map points from map, and set them to be local map points
        mpReferenceKF = pKFini;// set the reference keyframe to keyframe ini
        mCurrentFrame.mpReferenceKF = pKFini;// set the reference key frame of the current frame to keyframe ini

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);// set the reference mappoints of map to be local map points

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);// insert the keyframe ini to mvpKeyFrameOrigins of the map

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);// set the transform to the map drawer

        mState=OK;
    }
}

void Tracking::MonocularInitialization()
{

    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;

            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
    }
    else
    {
        // Try to initialize
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);


    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    // Set median depth to 1
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

/*
 * 1. compute bag of words vector for the current frame
 * 2. perform an orb matching with the reference keyframe, that is, use the node index in the tree to search the feature point in the frame corresponding to the point in keyframe
 *    and then use the angle statistic information to delete those pairs have angle difference values away from other pairs. If the number of matching feature points is less
 *    than 15, just return false
 * 3. set the map points computed in the last step as the map points of current frame
 * 4. set the initial pose of current frame to be the pose of the last frame then use the map points to optimize the reprojection error and get the pose estimation
 * 5. for all map points in the current frame, if it is marked as an outlier in the last step, set the pointer to be null, and decrease nmatches by 1; else, if the
 *    observation times of the map point is bigger than 0, increase nmatchesMap by 1
 * 6. if nmatchesMap >= 10, return true; else return false 
 */
bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector   compute the bow vector for the current frame
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    // set mfNNratio=0.7, mbCheckOrientation=true
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    // KeyFrame *mpRefrenceKF           Frame mCurrentFrame
    /* use the node index in the tree to search the feature point in the frame corresponding to the point in keyframe, and then use the angle statistic information
       to delete those pairs have angle difference values away from other pairs*/
   int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    // if the number of matching feature points is less than 15, return false
    if(nmatches<15)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;//set the map points computed in the last step as the map points of current frame
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    return nmatchesMap>=10;
}

/*
 * 1. set the pose of last frame to be mlRelativeFramePoses.back()*reference_pose
 * 2. if last frame is a keyframe, or the sensor is monocular, or the node does not only tracking but also local mapping, return from this function
 * 3. for every keypoint in the last frame, if it has a positive depth, put its depth and index into vector vDepthIdx, after the for circulation, if the 
 *    vDepthIdx is empty, return from this function, else sort vDepthIdx according to the depth
 * 4. for every element in vector vDepthIdx, use the index in the pair to get the related map point, if the map pointer is null or the number of observations is less 
 *    than 1, use the depth to compute the 3D position of the map point, use the position computed to create a new map point, set the newly created map point to be 
 *    the mappoint in last frame and put the newly created map point into mlpTemporalPoints. if the point's depth is no more than the threshold or the number of points is 
 *    no more than 100, continue the for circulation; else break from the circulation 
 */
void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    // list<cv::Mat> mlRelativeFramePoses , lists used to recover the full camera trajectory at the end of the execution
    // Basically we store the reference keyframe for each frame and its reletive transformation
    // list::back() returns a reference to the last element in the list container
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    // if last frame is a keyframe, or the sensor is monocular, or the node does not only tracking but also local mapping, return from this funtion
    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    // for every keypoints in the last frame, if it has a positive depth, put its depth and index into the vector
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    // if the above mentioned vector is empty, return from this function
    if(vDepthIdx.empty())
        return;

    // sort the vector vDepthIdx according to the depth
    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    /*
    for every element in vector vDepthIdx, use the index in the pair to get the related map point, if the map pointer is null or the number of observations is less 
    than 1, use the depth to compute the 3D position of the map point, use the position computed to create a new map point, set the newly created map point to be 
    the mappoint in last frame and put the newly created map point into mlpTemporalPoints. if the point's depth is no more than the threshold or the number of points is 
    no more than 100, continue the for circulation; else break from the circulation 
    */
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

/*
 * 1. update last frame pose according to its reference keyframe and relative frame poses, if in the localization mode, create mappoints for applying 
 *    visual odometry
 * 2. using the motion model and the pose of last frame to set pose for current frame
 * 3. if this is the stereo case, set the threshold to 7, else set the threshold to 15, compute tlc using the last frame pose and the current frame pose,
 *    search the corresponding in current frame corresponding to last frame by projecting keypoints in last frame to current frame, for keypoints in current 
 *    frame in the window centered at the projected keypoint,compute the distance between descriptor of candidate point and descriptor of keypoint in last 
 *    frame, if the best distance among the candidates is less than the threshold, set mapppoint for the candidate and add the nmatches by 1. If 
 *    mbCheckOrientation is true, set the ones without the main direction of orientation to invalid
 * 4. optimize the pose of PFrame using the map point, add unary edges to the optimization graph, then optimize the problem and reset the pFrame pose，the 
 *    returned int type value is (nInitialCorrespondences - nBad)(i.e. the number of inliers), where
 *    nInitialCorrespondences is the number of valid mappoints in pFrame
 *    nBad is the number of edges who has a chi2 bigger than thres after optimazation
 * 5. discard outliers, if the number of matches is bigger than some certain threshold, return true, else return false
 */
bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    // update last frame pose according to its reference keyframe, if in the localization mode, create mappoints for applying visual odometry
    UpdateLastFrame();

    // using the motion model and the pose of last frame to set pose for current frame
    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    /*
    std::fill(first, last, val) first is the beginner iterator of the container, last is the end iterator of the container, val is the new value used to replace the old ones
    */
    // fill mvpMapPoints of current frame with null pointers
    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    // if this is the stereo dataset, set the threshold to 7, else set the threshold to 15
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;
    // Frame mCurrentFrame;     Frame mLastFrame
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, clear mCurrentFrame, uses a wider window search
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    /*
     * optimize the pose of PFrame using the map point, add unary edges to the optimization graph, then optimize the problem and reset the pFrame pose，the returned
     * int type value is (nInitialCorrespondences - nBad)(i.e. the number of inliers), where
     * nInitialCorrespondences is the number of valid mappoints in pFrame
     * nBad is the number of edges who has a chi2 bigger than thres after optimazation
     */
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    // for all keypoints in the current frame
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i]) // if the mappoint pointer is not a null pointer
        {
            if(mCurrentFrame.mvbOutlier[i]) // if the map point is an outlier
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];// get the mappoint

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)// if the number of the observation is bigger than 0, add nmatchesMap by 1
                nmatchesMap++;
        }
    }    

    // mbOnlyTracking is true if local mapping is deactivated and performing only localization
    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;// if nmatchesMap<10, mbVO is true; else mbVO is false
        return nmatches>20;// if nmatches>20, return true; else return false
    }

    return nmatchesMap>=10;// if nmatchesMap>=10, return true; else return false
}

/*
 * 1. update mvpLocalKeyFrames and mvpLocalMapPoints
 * 2. find match for mCurrentFrame with mappoints in mvpLocalMapPoints, if match found, set the mappoint of mCurrentFrame tobe the one in mvpLocalMapPoints
 * 3. optimize the pose of mCurrentFrame using the mappoint, add unary edges to the optimization graph, then optimize the problem and reset mCurrentFrame pose
 * 4. for every keypoint in mCurrentFrame, if the mappoint is not a null pointer
 *    4.1 if the mappoint is not marked as an outlier, increase the mnFound of the mappoint by 1. If this is the mapping mode, if the observation of the
 *        mappoint is bigger than 0, add mnMatchesInliers by 1; else, add mnMatchesInliers by 1
 *    4.2 if the sensor is stereo, set the mappoint tobe null pointer
 * 5. if mCurrentFrame is neaer the relocFrame and mnMatchesInliers is less than 50, return false
 * 6. if mnMatchesINliers is less than 30, return false
 * 7. else, return true
 */
bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    // update mvpLocalKeyFrames and mvpLocalMapPoints
    UpdateLocalMap();

    // find match for mCurrentFrame with mappoints in mvpLocalMapPoints, if match found, set the mappoint of mCurrentFrame tobe the one in mvpLocalMapPoints
    SearchLocalPoints();

    // Optimize Pose
    /* optimize the pose of mCurrentFrame using the map point, add unary edges to the optimization graph, then optimize the problem and reset the mCurrentFrame 
    pose，the returned int type value is (nInitialCorrespondences - nBad)(i.e. the number of inliers)*/
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)// for every keypoint in mCurrentFrame
    {
        if(mCurrentFrame.mvpMapPoints[i])// if the mappoint is not a null pointer
        {
            if(!mCurrentFrame.mvbOutlier[i])// if the mappoint is not marked as an outlier
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();// increase the mnFound of the mappoint by 1
                if(!mbOnlyTracking)// if this is in mapping mode
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)// if the observation of the mappoint is bigger than 0
                        mnMatchesInliers++;// add mnMatchesInliers by 1
                }
                else// if this is in localization mode
                    mnMatchesInliers++;// add mnMatchesInliers by 1
            }
            else if(mSensor==System::STEREO)// if the mappoint is marked as an outlier and the sensor is stereo, set the mappoint tobe null pointer
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    // if mCurrentFrame is near the reloc frame and mnMatchesInliers is less than 50, return false
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if(mnMatchesInliers<30)// if mnMatchesInliers is less than 30, return false
        return false;
    else// else, return true
        return true;
}

/*
 * lots of principles to help determine whether a new keyframe is needed, did not list the whole process here
 * 1. if this is the localization mode, return false
 * 2. if the local mapper is stopped or stop requested, return false
 * 3. get the number of keyframes in map as nKFs
 * 4. if the current frame is near a reloc frame and the number of keyframes of the map is bigger than the threshold, return false
 * 5. compute how many mappoints in mapReferenceKF was observed more than nMinObs times as nRefMatches
 * 6. set bLocalMappingIdle tobe mBAcceptKeyFrames of mpLocalMapper
 */
bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking)// if this is the localization mode, return false
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    // if the local mapper is stopped or stop requested, return false
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();// get the number of keyframes in map

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    // if the curent frame is near a reloc frame and the number of keyframes of the map is bigger than the threshold, return false  
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    // compute how many map points in mpReferenceKF was observed more than nMinObs times
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?  idle闲置的
    // set bLocalMappingIdle tobe mbAcceptKeyFrames of mpLocalMapper
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)// if the sensor is not monocular
    {
        for(int i =0; i<mCurrentFrame.N; i++)// for every keypoint in mCurrentFrame
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)// if the depth of the keypoint is in a certain range
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])// if the mappoint is not a null pointer and is not marked as an outlier
                    nTrackedClose++;// add nTrackedClose by 1
                else
                    nNonTrackedClose++;// add nNonTrackedClose by 1
            }
        }
    }

    // if nTrackedClose is less than 100 and nNonTrackedClose is bigger than 70, set bNeedToInsertClose to true
    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)// if the number of keyframes in the map is less than 2, set thRefRatio tobe 0.4
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)//if the sensor is monocular, set the thRefRatio tobe 0.9
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    // if mCurrentFrame index has passed the mnLastKeyFrameId more than mMaxFrames, cla is true
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    // if mCurrentFrame index has passed the mnLastKeyFrameId more than nMinFrames and bLocalMappingIdle is true, clb is true
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    // Condition 1c: tracking is weak
    // if the sensor is not monocular, and (mnMatchesInliers is less than threshold or bNeedToInsertClose is true), clc is true
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    // if (mnMatchesInliers is less than threshold or bNeedToInsertClose is true) and mnMatchesInliers is more than 15, c2 is true
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2) //if (cla or clb or clc is true) and c2 is true
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        // if bLocalMappintIdle is true, return true
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();// send a signal to interruptBA
            // if the sensor is not monocular, if mlNewKeyFrames of mpLocalMapper is less than 3, return true;else return false
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else// if the sensor is monocular, return false
                return false;
        }
    }
    else
        return false;
}

/*
 * 1. if mpLocalMapper->SetNotStop(true) returns false, return from this function
 * 2. construct a new keyframe pKF using mCurrentFrame, set mpReferenceKF tobe pKF and mpReferenceKF of mCurrentFrame tobe pKF
 * 3. if the sensor is not monocular
 *    3.1 update mRcw, mRwc, mtcw, mOw for mCurrentFrame
 *    3.2 for all keypoints in mCurrentFrame , if the depth of the keypoint is bigger than 0, make pair of the depth and index of the keypoint and 
 *        put the pair into vector vDepthIdx
 *    3.3 if vDepthIdx is not null, sort vDepthIdx by the depth of keypoint, for every pair in vDepthIdx, if the mappoint is null or the number of
 *        observation of the mapppoinf is less than 1, set bCreateNew tobe true. If bCreateNew is true, use the depth to create mappoint in the
 *        world coordinate, use the 3D coordinate computed above to construct a new map point, use the key frame and the index to add observation 
 *        for the mappoint, add the new mappoint for the keyframe, set descriptor for the mappoint, update the normal and depth for the mappoint,
 *        add new mappoint to map, set the new mappoint tobe the mappoint of the keypoint in the current frame. In this step, we create mappoints
 *        whose depth is less than thres, if there are less than 100 points we create 100 closest
 * 4. insert the new keyframe constructed by mCurrentFrame to mpLocalMapper, set mnLastKeyFrameId tobe the mnId of the mCurrentFrame, set mpLastKeyFrame 
 *    tobe the new keyframe constructed by mCurrentFrame, i.e. pKF
 */
void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);// use mCurrentFrame to construct a new keyframe

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mSensor!=System::MONOCULAR)// if the sensor is not monocular
    {
        mCurrentFrame.UpdatePoseMatrices();// update mRcw, mRwc, mtcw, mOw for mCurrentFrame

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)// if the depth of the keypoint is bigger than 0
            {
                vDepthIdx.push_back(make_pair(z,i));// make pair of the depth and index of the keypoint, and put the pair into a vector
            }
        }

        if(!vDepthIdx.empty())// if the vector of pairs is not empty
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());// sort the vector of pairs using the depth

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)// for all pairs in the vector
            {
                int i = vDepthIdx[j].second;// get the index in the frame

                bool bCreateNew = false;// initialize bCreateNew tobe false

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];// get the mappoint of the index
                if(!pMP)// if the mappoint is null, set bCreateNew to true
                    bCreateNew = true;
                else if(pMP->Observations()<1)// else if the number of observations is less than 1, set bCreateNew tobe true and set the mappoint to null pointer
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)// if bCreateNew is true
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);// use the depth to compute the mappoint in the world coordinate
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);// use the 3D coordinate computed above to construct a new map point
                    pNewMP->AddObservation(pKF,i);// use the key frame and the index to add observation for the mappoint
                    pKF->AddMapPoint(pNewMP,i);// add the new mappoint for the keyframe
                    //a mappoint may have a lot of observations, choose the best descriptor among all the descriptors
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();// update the normal and depth
                    mpMap->AddMapPoint(pNewMP);// add the new mappoint to the map

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;// set the new mappoint tobe the mappoint of the keypoint in the current frame
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                // if the depth od the point is bigger than thres and the number of mappoints is bigger than 100, break from the for circulation
                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);// insert the new keyframe constructed by mCurrentFrame to mpLocalMapper

    mpLocalMapper->SetNotStop(false);// 

    mnLastKeyFrameId = mCurrentFrame.mnId;// set mnLastKeyFrameId tobe the mnId of the mCurrentFrame
    mpLastKeyFrame = pKF;// set mpLastKeyFrame tobe the new keyframe constructed by mCurrentFrame
}

/*
 * 1. for every mapppoint in mCurrentFrame.mvpMapPoints, get the mappoint, if it is not a null pointer, if the mappoint is bad, set the mappoint tobe a null
 *    pointer; else, increase mnVisible of the mappoint by 1, set the mnLastFrameSeen of the mappoint tobe mCurrentFrame, set the mbTrackInView of the mappoint 
 *    to false.
 *    this step mark those mappoints already observed by mCurrentFrame, set its mnLastFrameSeen tobe mCurrentFrame, so in the following step, we don't need
 *    to check whethre the mappoint is in frustum of mCurrentFrame
 * 2. set nToMatch tobe 0
 * 3. for every mappoint in mvpLocalMapPoints
 *    3.1 get the mappoint. If the mnLastFrameSeen of the mappoint is mCurrentFrame, continue to process next mappoint. If the mappoint is a bad point, 
 *        continue to process next mappoint
 *    3.2 check whether the mappoint in the frustum of mCurrentFrame, if the mappoint is in, increase mnVisible of the mappoint by 1 and add nToMatch by 1
 * 4. if nToMatch is bigger than 0, set th=1, if the sensor is rgbd type, set th=3; if the camera has been relocalised recently, set th=5. Use the projected
 *    coordinate of the mappoint to find candidate for it in the input frame mCurrentFrame, if it really matches, set the mappoint of frame mCurrentFrame 
 *    tobe the mappoint in the input vector mvpLocalMapPoints
 */
/*
 * find match for mCurrentFrame with mappoints in mvpLocalMapPoints, if match found, set the mappoint of mCurrentFrame tobe the one in mvpLocalMapPoints
 */
void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    // for those map points already observed by mCurrentFrame, set its mnLastFrameSeen to mCurrentFrame so it won't be searched in the following step
    // for every map point in mCurrentFrame 
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)// if the mappoint is not a null pointer
        {
            if(pMP->isBad())// if the mappoint is a bad point, set it tobe null pointer
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();// increase mnVisible of the mappoint by 1
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;// set the mnLastFrameSeen of the mappoint tobe mCurrentFrame
                pMP->mbTrackInView = false;// set the mbTrackInView of the map point to false
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    // for every mappoint in mvpLocalMapPoints
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)// if the mnLastFrameSeen of the mappoint is mCurrentFrame, continue to process next mappoint
            continue;
        if(pMP->isBad())// if the mappoint is a bad point, continue to process next mappoint
            continue;
        // Project (this fills MapPoint variables for matching)
        // check whether the map point in the frustum of the frame, if the mappoint is in, increase mnVisible of the map point by 1 and add nToMatch by 1
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    // if nToMatch is bigger than 0
    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        // th is a value related to the search radius in the function SearchByProjection
        /*
         * use the projected coordinate of the mappoint to find candidate for it in the input frame F, if it really matches, set the mappoint of frame F tobe the 
         * mappoint in the input vector vpMapPoints, finally after process all mapppoints in vpMapPoints, return the number of matches found
         */
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

// update mvpLocalKeyFrames and mvpLocalMapPoints
void Tracking::UpdateLocalMap()
{
    // This is for visualization
    /* set mvpReferenceMapPoints tobe mvpLocalMapPoints*/
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

/*
 * 1. clear mvpLocalMapPoints
 * 2. for keyframes in mvpLocalKeyFrames
 *    2.1 get the keyframe and its mappoints
 *    2.2 for every mappoint got in 2.1, if it is a null pointer, continue to process next mappoint, if the track reference is already the mCurrentFrame, continue
 *        to process next mappoint, if the mappoint is not a bad point, push the mappoint into mvpLocalMapPoints and set its track reference to mCurrentFrame
 */
void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

/*
 * 1. make pair of keyframe who shares mappoint with mCurrentFrame and how many mapppoints they share, put the pairs into a map keyframeCounter, if the map 
 *    is empty, return from this function
 * 2. for every pair in keyframeCounter, get the keyframe, if the keyframe is a bad frame, continue to process next keyframe, push the keyframe into vector
 *    mvpLocalKeyFrames and set its track reference to mCurrentFrame. At the same time find the keyframe who shares most mappoints with mCurrentFrame
 * 3. for every keyframe in mvpLocalKeyFrames, i.e. keyframe who shares mappoint with mCurrentFrame and is not a bad keyframe itself
 *    3.1 if the size of mvpLocalKeyFrames is bigger than 80, break from this for circulation
 *    3.2 get covisible keyframes for local keyframe as vNeighs
 *    3.3 for every neighbor keyframe of the local keyframe, if the neighbor frame is not a bad frame and the track reference of the neighbor frame is 
 *        not mCurrentFrame, push the neighbor frame into mvpLocalKeyFrames and set its track reference to mCurrentFrame and break from 3.3 for circulation
 *    3.4 for every child keyframe of the local keyframe, if the child frame is not a bad frame and the track reference of the neighbor frame is not
 *        mCurrentFrame, push the child frame into mvpLocalKeyFrames and set its track reference to mCurrentFrame and break from 3.4 for circulation
 *    3.5 get the parent frame of this local frame, if the parent frame is not a bad frame and the track reference of the neighbor frame is not 
 *        mCurrentFrame, push the parent frame into mvpLocalKeyFrame and set its track reference to mCurrentFrame and break from 3 for circulation
 * 4. if the pKFmax i.e. local keyframe sharing most mappoints with mCurrentFrame is not null, set mpReferenceKF tobe pKFmax, and set the mpReferenceKF
 *    of mCurrentFrame tobe mpReferenceKF
 */
void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)// for every keypoint in the current keyframe
    {
        if(mCurrentFrame.mvpMapPoints[i])// if the mappoint is not a null pointer
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())// if the map point is not a bad point
            {
                // get the keyframe who observe the map point and the index of the keypoint
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
                // keyframeCounter stores the keyframe who observe the mappoint in mCurrentFrame and how many mappoints in mCurrentFrame is observed 
                // by that keyframe
            }
            else// if the mappoint in mCurrentFrame is a bad point, set the mappoint to null pointer
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }
    // if no keyframe observe mappoint in mCurrentFrame, return from this function
    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    // std::vector<KeyFrame*> mvpLocalKeyFrames; set the size of mvpLocalKeyFrames to 3 times the size of keyframeCounter
    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;// get the keyframe who observe a mappoint in mCurrentFrame

        if(pKF->isBad())// if the keyframe is a bad frame, continue to process next keyframe
            continue;

        if(it->second>max)// find the keyframe who shares most mappoints with mCurrentFrame
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;// set the track reference of the keyframe to mCurrentFrame
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    // for every keyframe who shares mappoint with mCurrentFrame and is not a bad keyframe itself
    // mvpLocalKeyFrames contains keyframes who share mappoint with mCurrentFrame and is not a bad frame itself
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)// if mvpLocalKeyframe has more than 80 keyframe, break from this for circulation
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);// get covisible keyframes of the local keyframe

        // for every neighbor keyframe of the local keyframe
        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            // get the neighbor keyframe of the local keyframe
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())// if the neighbor keyframe is not a bad keyframe
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)// if the track reference of the neighbor frame is not current frame
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);//put the neighbor frame to mvpLocalKeyFrames
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;// set the track reference of the neighbor frame to current frame
                    break;
                }
            }
        }

        /* for a keyframe A, if a keyframe B has most covisible map points with A, set B tobe A's parent and set A tobe B's child */
        const set<KeyFrame*> spChilds = pKF->GetChilds();
        // for localKeyframe's every child keyframe
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())// if the child keyframe is not a bad frame 
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)// if the track reference of the child frame is not mCurrentFrame
                {
                    mvpLocalKeyFrames.push_back(pChildKF);//push the child frame into mvpLocalKeyFrames
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;// set the track reference of the child frame tobe mCurrentFrame
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();// get the parent of the local keyframe
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)// if the local keyframe sharing most mappoints with mCurrentFrame is not null
    {
        mpReferenceKF = pKFmax;// set the mpReferenceKF tobe the local keyframe sharing most mapppoints with mCurrentFrame
        mCurrentFrame.mpReferenceKF = mpReferenceKF;// set the mpReferenceKF of mCurrentFrame tobe mpReferenceKF
    }
}

/*
 * 1. compute the bow vector of current frame, the bow vector is in the form of {(w1,weight1),(w2,weight2),...,(wn,weightn)}
 * 2. detect candidate keyframes for relocalization(find candidates among those keyframes who share words with mCurrentFrame, and take covisible 
 *    keyframes into consideration), if no candidate keyframes found, return false
 * 3. for every keyframe in vpCandidateKFs
 *    3.1 get the keyframe, if the frame is bad, set the corresponding element in vbDiscarded tobe true
 *    3.2 for keypoints in pKF and mCurrentFrame corresponding to the same node in the bow tree, use their distance of descriptor to find matches, if
 *        the number of matches is less than 15, set the corresponding element in vbDiscarded tobe true and continue to process next keyframe; else
 *        construct a pnp solver with mCurrentFrame and matched mappoints found in this step
 * 4. for every keyframe in vpCandidateKFs
 *    4.1 if the corresponding element in vbDiscarded is true, continue to process next keyframe
 *    4.2 for the pnp solver created in 3.2, if if ransac reaches max, discard the candidate keyframe, and decrease nCandidates by 1
 *    4.3 if the Tcw computed by pnpsolver is not empty, apply the optimize and projection step again and again
 * 5. if bMatch is false, it represents that we can not find a match, return false; else set mnLastRelocFrameId tobe the id of current frame and return true 
 */
bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector, the bow vector is in the form of {(w1,weight1),(w2,weight2),...,(wn,weightn)}
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalization
    // find candidate keyframes for relocalization
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    // if there is no candidate keyframe for relocalization, return false
    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();//get the number of the candidate keyframes

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)// for every keyframe in vpCandidateKFs
    {
        KeyFrame* pKF = vpCandidateKFs[i]; // get the keyframe
        if(pKF->isBad())
            vbDiscarded[i] = true;// if the keyframe is bad, the correspoding element in vbDiscarded is true
        else
        {
            /* for keypoints in pKF and mCurrentFrame corresponding to the same node in the bow tree, compute the descriptor between them, for the one with the 
             descriptor distance satisfying the condition, set it as the match for keypoint in mCurrentFrame, and check the orientation for the matches*/
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)// if the keypoint matches is very few, set the corresponding element in vbDiscarded as true
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                // every valid candidate keyframe make a pnp solver
                // get keyframes from mCurrentFrame and its corresponding  pyramid level, get mapppoints from vvpMapPointMatches
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                //compute a series of parameters for the pnp problem 
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)// for every candidate keyframe
        {
            if(vbDiscarded[i])// if the corresponding element in vbDiscarded is true, continue to process next keyframe
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)// if ransac reaches max, discard the candidate keyframe, and decrease nCandidates by 1
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);// set the pose of current frame using the pnp result

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)// for every inlier pnp found
                {
                    if(vbInliers[j])// if the map point is marked as an inlier
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];// set the mappoint to be the mappoint of current frame 
                        sFound.insert(vvpMapPointMatches[i][j]);// insert the map point to set
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;// set the corresponding element of current frame to null
                }

                /* optimize the pose of mCurrentFrame using the map point, add unary edges to the optimization graph, then optimize the problem and reset
                the mCurrentFrame pose, the returned int value is the number of inliers*/
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)// if the inliers of optimization is less than 10, continue to process next candidate keyframe
                    continue;

                // for every keypoint in the current frame, if the keypoint is marked as an outlier, set the corresponding mappoint to null
                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)// if the number of inliers in the optimization problem is less than 50
                {
                    // sFound is the mapppoints found by pnp solver
                    // sFound in the parameter list means set of mappoints already found
                    // project mappoint in vpCandidateKFs to mCurrentFrame to find matches
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    // if the (inliers in the optimition + those ones find in the projection step) >= 50
                    if(nadditional+nGood>=50)
                    {
                        /* optimize the pose of mCurrentFrame using the map point, add unary edges to the optimization graph, then optimize the problem and
                           reset the mCurrentFrame pose, the returned int value is the number of inliers*/
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            // if the (those ones find in the projection step + inliers in the optimition) >= 50
                            // 
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                // for all keypoints in the current frame, if it is marked as an outlier, set the corresponding mappoint pointer to null
                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    // if bMatch is false, it represents that we can not find a match, return false
    if(!bMatch)// if bMatch is false, we can not find a match, return false
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

/* the reset operation of the tracking class include
 * 1. stop the viewer
 * 2. reset local mapping
 * 3. reset loop closing
 * 4. clear the keyframe database
 * 5. clear map(erase map points and keyframes)
 * 6. delete and clear a series of pointers and lists
 */
void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}



} //namespace ORB_SLAM
