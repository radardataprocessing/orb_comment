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

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}

//stereo constructor
/*
 * 1. extract keypoints(image was divided into octree nodes, for every node retain the best keypoint in it) and and compute their corresponding descriptors
 * 2. try to find the depth and corresponding right keypoint(in subpixel) for every left keypoint
 * 3. if this is the first frame, compute some parameters
 * 4. compute which grid the keypoint is in and put the keypoint into the corresponding container
 */
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();// number of levels of the pyramid
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    /*
     * the process is defined in ORBextractor.cc operator()
     */
    /* 1. pre-compute the image pyramid 
     * 2. compute keypoints for every level of the pyramid, and compute their orientation(image was divided into octree nodes, for every node retain the best keypoint in it)
     * 3. count the number of keypoints in all levels of the pyramid, set the descriptor mat to size(number of keypoints, 32),type CV_8UC1
     * 4. for every level of the pyramid, gaussian blur the corresponding image, using the preset pattern to compute descriptor in the blurred image, if this is not the 0
     *    level in the pyramid, multiply the keypoint cordinate with the scale factor of current level
     * 5. insert the keypoints into the vector _keypoints, i.e. the third parameter of this function
     */
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);// flag is 0, use the left extractor
    thread threadRight(&Frame::ExtractORB,this,1,imRight);// flag is 1, use the right extractor
    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // try to find the depth and corresponding right keypoint(in subpixel) for every left keypoint
    ComputeStereoMatches();

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    // get some parameters
    if(mbInitialComputations)
    {
        // undistort the image and compute the new bounding points
        ComputeImageBounds(imLeft);// Undistort corners 对角点去畸变，求取去畸变后图片有效的行列范围

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);//FRAME_GRID_COLS 64
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);//FRAME_GRID_ROWS 48

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    //compute which grid the keypoint is in and put the keypoint into the corresponding container
    AssignFeaturesToGrid();//将特征点存入栅格中
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);//若flag即第一个参数为0，则计算左图，否则计算右图

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();//对关键点去畸变

    ComputeStereoFromRGBD(imDepth);//在深度图中取出深度并用disparity = f * baseline/depth恢复视差求出虚拟的右图坐标

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}


Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info , get scale level info from the extrctor in the parameter list
    mnScaleLevels = mpORBextractorLeft->GetLevels();// an int value
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();// a float value
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();// a vector
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();// a vector
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();// a vector
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();// a vector

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;
    // mbf = fx*baseline

    AssignFeaturesToGrid();
}

//compute which grid the keypoint is in and put the keypoint into the corresponding container
void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);//N为关键点的个数
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);//将每个栅格的存储能力置为nReserve

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        // compute which grid the keypoint is in
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

/* */
void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        // defined in ORBextractor.cc void ORBextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints, OutputArray _descriptors)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);//世界坐标系到相机坐标系的旋转
    mRwc = mRcw.t();//转置//相机坐标系到世界坐标系的旋转
    mtcw = mTcw.rowRange(0,3).col(3);//世界坐标系到相机坐标系的平移
    mOw = -mRcw.t()*mtcw;//相机坐标系到世界坐标系的平移    Pw = mRwc*Pc+mOw相机中心在相机坐标系下的坐标为（0， 0， 0），则mOw为相机中心在世界坐标系下的坐标
    // P2 = R21*p1 + T21 则 P1 = R21.inverse() * P2 - R21.inverse() * T21
}

/*
 * 1. set the mbTrackInView of the mappoint to false
 * 2. get the position of the mappoint in world coordinate, use the transformation between camera coordinate and world coordinate to transform the 
 *    mapppoint position from the world coordinate to camera coordinate, if the depth in the camera coordinate is negative, return false
 * 3. project the mappoint from the camera coordinate to image plane, if the projected point is out of the image bound, return false
 * 4. get the maxDistance and minDistance of the mappoint, compute PO as a vector in the world coordinate pointing to mappoint from camera center, compute 
 *    the norm of the vector, if the norm is not in the bound of minDistance and maxDistance, return false
 * 5. get the mean view direction of the mappoint Pn, if the cos value of Pn and PO is less than the threshold, return false
 * 6. predict which pyramid level the mappoint is in for the mappoint, set some params for the mappoint, and return true
 */
bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)//frustum截头锥体，平截头体
{
    pMP->mbTrackInView = false;//set the mbTrackInView of the map point to false

    // 3D in absolute coordinates  get the position of the mappoint in world coordinate
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    // using transformation between camera coordinate and world coordinate to transform the mappoint from world frame to camera frame
    const cv::Mat Pc = mRcw*P+mtcw;//将点坐标从世界坐标系转到相机坐标系
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth，查看深度是否为正，若深度为负则退出
    // if the depth of the mappoint in camera frame is negative, return false
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside，投影到图片坐标下，看其是否在图片范围内
    // project the point from the camera coordinate to image plane
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    // if the projected point is out of the image bound, return false
    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)//若投影后发现其范围超出预计算的最大最小，则返回false
        return false;

    // Check distance is in the scale invariance region of the MapPoint       invariance不变性，不变式
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    // mOw is the camera position in the world coordinate, so PO is a vector in the world coordinate pointing to mappoint from the camera center
    const cv::Mat PO = P-mOw;//P为该关键点在世界坐标系下的坐标   PO即为经过相机到世界坐标系的旋转的相机坐标系下的坐标
    // Pc = mRcw*P+mtcw     P = mRcw.inverse()*Pc - mRcw.inverse()*mtcw = mPwc*Pc + mOw     p-mOw = mRwc*Pc
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)// if the distance between the point and camera center is out of the threshold bound, return false
        return false;

   // Check viewing angle       GetNormal() returns the mean view direction 
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    // if the direction of the vector in the world coordinate pointing to mappoint from the camera center is away from the mean view direction, return false
    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image, predict which pyramid level the mappoint is in
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;// set the mbTrackInView of the mappoint tobe true
    pMP->mTrackProjX = u;// set mTrackProjX of the mappoint tobe u
    pMP->mTrackProjXR = u - mbf*invz;//在右图中的坐标// use u and disparity computed by depth to compute the u in the right view and set mTrackProjXR
    pMP->mTrackProjY = v;// set mTrackProjY of the mappoint tobe v
    pMP->mnTrackScaleLevel= nPredictedLevel;// set the mnTrackScaleLevel of the mappoint tobe nPredictedLevel
    pMP->mTrackViewCos = viewCos;// set mTrackViewCos of the mappoint tobe viewCos

    return true;
}

/*
 * 1. compute the grid range the window is in, if none of the grid is in the bound of the image, return an empty vector
 * 2. if the minLevel or the maxLevel is positive, set the bCheckLevels to true  
 * 3. for all grids the window is in 
 *    3.1 get the vector of keypoint indexes of the certain grid, if the vector is empty, continue to process next grid
 *    3.2 for all keypoints in that cell, get the undistorted keypoint
 *        3.2.1 if the bCheck is true and the keypoint octave is out of the level range, continue to process next keypoint
 *        3.2.2 if the keypoint is close to the point in the parameter list, put the point index into list
 * 4. return the vector of indexes
 */
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const//取出[x-r, x+r],[y-r, y+r]范围内去畸变的关键点
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    // compute the grid range which the window is in 
    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));//mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;
    //求出x,y被区间(-r, r)作用后所在栅格的范围

    // if the minLevel or the maxLevel is positive, set the bCheckLevels to true 
    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    // for all the grids the window is in
    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];// get the vector of keypoint indexs of the certain grid
            if(vCell.empty())// if the vector is empty, continue to process the next cell
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)// for all keypoints in that cell
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];// get the undistorted keypoint
                // if bCheckLevels is true and the keypoint octave is out of the level range, continue to process next keypoint
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                // if the keypoint is close to the point in the parameter list, put the point index into list
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    // return the vector of indexes
    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        // convert descriptor from cv::Mat to std::vector<cv::Mat>
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // DBoW2 computes the BowVector of an image as {(w1,weight1),(w2,weight2),...,(wn,weightn)}
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}
/*
 * TF_IDF
 * IDF:  in the process of creating tree, N represents the number of all the features, Ni represents word i has Ni features, IDFi=log(N/Ni)
 * TF:   an image has m features, the word wi has mi features, as a result, TFi=mi/m
 * the final weight of wi is TFi*IDFi
 * IDF is decided by the process of creating tree
 * TF is decided by the certain image features
 */

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }
    /*
    C++: Mat Mat::reshape(int cn, int rows=0 const)

    cn：目标通道数，如果是0则保持和原通道数一致；

    rows：目标行数，同上是0则保持不变；
    */
    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)//if the first distortion parameter is not 0 
    {
        // construct a mat using the four image bound corners of the left image
        cv::Mat mat(4,2,CV_32F);//将左图像的四个角点按照x,y的顺序存入mat中
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;//
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners 求取去畸变后图片有效的行列范围
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

/*
 * try to find the depth and corresponding right keypoint(in subpixel) for every left keypoint
 */
void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;//TH_HIGH = 100    TH_LOW = 50

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;// the number of rows in the origin image

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();//std::vector<cv::KeyPoint> mvKeys, mvKeysRight;

    /*
    cv::KeyPoint
    float angle: computed orientation of the keypoint (-1 if not applicable); it's in [0,360) degrees and measured relative to image coordinate system, ie in clockwise.
    int class_id: object class (if the keypoints need to be clustered by an object they belong to) 
    int octave: octave (pyramid layer) from which the keypoint has been extracted
    Point2f pt: coordinates of the keypoints
    float response: the response by which the most strong keypoints have been selected. Can be used for the further sorting or subsampling
    float size: diameter of the meaningful keypoint neighborhood
    */

    for(int iR=0; iR<Nr; iR++)// for every keypoint in the right image
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];//取出一个右图中的关键点
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);//最大行范围
        const int minr = floor(kpY-r);//最小行范围

        for(int yi=minr;yi<=maxr;yi++)//for every row from the min row range to the max row range, put the index of this keypoint into the corresponding vRowIndices
            vRowIndices[yi].push_back(iR);//在最大范围到最小范围的行中都加入这个关键点的序号
    }

    // Set limits for search
    const float minZ = mb;// stereo baseline in meters
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image 对于左图中每个关键点，找右图中的一个对应
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);//N means number of keypoints in the left image

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;//row index of the keypoint
        const float &uL = kpL.pt.x;//column index of the keypoint

        const vector<size_t> &vCandidates = vRowIndices[vL];//get the indexes of the keypoints in the right image which may match this keypoint

        if(vCandidates.empty())//if no candidates index found, continue to process next keypoint
            continue;

        const float minU = uL-maxD;// use the max disparity to get the minimum column index
        const float maxU = uL-minD;// use the min disparity to get the maxmum column index, candidates in this range may match the current keypoint

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;//TH_HIGH= 100
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);//get the descriptor of the currrent key point

        // Compare descriptor to right keypoints 比较左右关键点的描述子
        /*
         * the candiate must meet two cinditions:
         * 1. lie in the same or neighbor level of the pyramid with the current keypoint
         * 2. lie in the column range conputed using the disparity range
         */
        for(size_t iC=0; iC<vCandidates.size(); iC++)//for all candidate keypoints in the right image of the current point
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            // if the candidate don't lie in the same of neighbor level of pyramid with the current keypoint, continue to process the next candidate
            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)//如果候选点与本关键点不在金字塔的同一层或相邻层，则不再处理本候选点
                continue;

            const float &uR = kpR.pt.x;

            // if the candidate lie in the column range computed using the disparity range, compute the distance of the two descriptor 
            if(uR>=minU && uR<=maxU)//若候选点所在列在视差允许的列范围内
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                // compute the distance of the descriptor of the current keypoint and the candidate keypoint 
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)//找出所有候选点中与关键点描述子距离最小的点
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation 用相关性进行亚像素的匹配
        // const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;//TH_HIGH = 100    TH_LOW = 50
        // if the best distance of the current keypoint and the candidate is less than the threshold
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            // get the scale factor of the current keypoint, multiply the scale factor to the row and column of the current keypoint and the column of the best candidate
            const float uR0 = mvKeysRight[bestIdxR].pt.x;//取出最佳候选点的列坐标
            const float scaleFactor = mvInvScaleFactors[kpL.octave];//取出本关键点所在金字塔层的缩放因子
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);//取出缩放因子作用后的关键点的行列坐标
            const float scaleduR0 = round(uR0*scaleFactor);//取出缩放因子作用后的最佳候选点列坐标

            // sliding window search
            const int w = 5;
            // from the image of the current keypoint pyramid level, get a 11*11 patch centered at the current keypoint
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);//取出缩放因子作用后关键点附近的一个窗
            IL.convertTo(IL,CV_32F);//将图中元素转换为浮点型
            // for the 11*11 patch centered at the current keypoint, make every pixel in patch substract the center pixel
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);//对窗内每个点都减去中心点的值

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;//scaleduR0为缩放因子作用后的最佳候选点列坐标
            const float endu = scaleduR0+L+w+1;
            // if can not extract a patch of the same size from the keypoint level right image, continue to process next keypoint
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)//在[scaleduR0-L,scaleduR0+L]范围内加窗处理
            {
                // from the right image of the current keypoint pyramid level, get a 11*11 patch centered at the (best_candidate.x + incR, best_candidate.y)
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                //Mat.rowRange（int x，int y）和Mat.rowRange(range(int x,int y)得到的结果一样，函数取的实际行数y-x，只取到范围的左边界，而不取右边界。
                IR.convertTo(IR,CV_32F);
                // for the 11*11 patch centered at the best candidate keypoint, make every pixel in patch substract the center pixel
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                
                float dist = cv::norm(IL,IR,cv::NORM_L1);//求左右图像窗内元素的一范数
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;//将incR求出的一范数存储下来
            }

            if(bestincR==-L || bestincR==L)//若最佳疑犯数对应的元素在边界，则只有一个邻域，不能拟合抛物线所以不再处理这个点
                continue;

            // Sub-pixel match (Parabola fitting)拟合抛物线然后求抛物线最低点
            // fit a parabola using the column index of the best dist and its two neighbors, then find the minimum 
            const float dist1 = vDists[L+bestincR-1];//取出最佳范数的邻域元素的范数
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));
            /*
            y1 = a*(x-1)*(x-1) + b*(x-1) + c
            y2 = a*x*x + b*x + c
            y3 = a*(x+1)*(x+1) + b*(x+1) + c

            y1 - y2 = -2*a*x + a - b
            y3 - y2 = 2*a*x + a＋ b

            (y1 + y3) - 2*y2 = 2*a
            y3 - y1 = 4*a*x + 2*b

            2*a = (y1 + y3) - 2 * y2
            2*b = y3 - y1 - 4*a*x

            对称轴 -b/(2*a) = (y1-y3+4*a*x)/(4*a) = (y1 - y3)/(4*a)＋ｘ
            对称轴与ｘ的距离为 (y1 - y3)/(4*a)
            */

            if(deltaR<-1 || deltaR>1)//若最小值超出了左右一个像素，则不合理
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);//在重新缩放的坐标系下最佳的右侧列坐标为

            float disparity = (uL-bestuR);//disparity in the current keypoint level of the pyramid

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;//左图中关键点对应的深度
                mvuRight[iL] = bestuR;//左图中关键点对应的右图列坐标
                vDistIdx.push_back(pair<int,int>(bestDist,iL));//bestDist存储的是最佳的左右图像窗内元素一范数
            }
        }
    }
    // vector<pair<int, int> > vDistIdx; vDistIdx sotres pair<best_norm_L1_of_patch, index of the left keypoint>
    sort(vDistIdx.begin(),vDistIdx.end()); // sort the vDistIdx by norm
    const float median = vDistIdx[vDistIdx.size()/2].first;//取窗内左右元素一范数的中值
    const float thDist = 1.5f*1.4f*median;//阈值设置为2.1倍中值

    /* for every keypoint from the biggest norm to the smallest, if the norm is bigger than the threshold,set the depth and 
     * corrresponding right keypoint of the left keypoint to -1
     */
    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else//若一范数大于阈值，则设置对应右图坐标以及深度为-1
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

/*
 * get the depth of the key point, if the depth is bigger than 0, use the depth, undistorted pixel coordinate and K matrix to compute the 3D position in the camera coordinate, then
 * use the transform matrix between camera coordinate and world coordinate to compute the 3D position in the world coordinate
 */
cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}
/*

*/

} //namespace ORB_SLAM
