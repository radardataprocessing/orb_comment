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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include<mutex>

namespace ORB_SLAM2
{

LocalMapping::LocalMapping(Map *pMap, const float bMonocular):
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::Run()
{

    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy            跟踪模块会得知局部建图模块忙碌
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue   查看队列中是否有新的关键帧
        if(CheckNewKeyFrames())
        {
            // BoW conversion and insertion in Map   进行Bow转换并插入到地图
            ProcessNewKeyFrame();

            // Check recent MapPoints    检查最近的地图点
            MapPointCulling();

            // Triangulate new MapPoints    三角化新地图点
            CreateNewMapPoints();

            if(!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors();
            }

            mbAbortBA = false;

            if(!CheckNewKeyFrames() && !stopRequested())//如果没有检查到新关键帧且未中断请求
            {
                // Local BA
                if(mpMap->KeyFramesInMap()>2)  //如果地图中的关键帧大于两个
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap);

                // Check redundant local Keyframes
                KeyFrameCulling();
            }

            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if(Stop())
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())
                break;
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }

    SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)//将关键帧插入mlNewKeyFrames链表中
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}


bool LocalMapping::CheckNewKeyFrames()//检查mlNewKeyFrames链表是否为空
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures  对当前关键帧计算词袋结构
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor     
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();//取出当前帧中的地图点

    for(size_t i=0; i<vpMapPointMatches.size(); i++)//迭代每一个地图点
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())//若地图点不是坏点
            {
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking 只发生在跟踪模块插入新双目点时
                {
                    mlpRecentAddedMapPoints.push_back(pMP);//将地图点加入最近新添地图点中
                }
            }
        }
    }    

    // Update links in the Covisibility Graph   更新当前关键帧的共视图关系
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map    在地图中插入当前关键帧
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints    检查新添的地图点
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)//若为单目要求观察到的次数阈值为2否则为3
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    while(lit!=mlpRecentAddedMapPoints.end())//对所有最近新添地图点进行迭代
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())//若地图点为坏点，则从最近新添地图点集中移除该点
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f )//若比例过低，则将地图点设为坏点并从最近新添地图点集中移除他
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        //若当前关键帧序号与地图点第一个关键帧序号差大于等于2且地图点观测次数小于一定阈值，则将地图点设置为坏点并从最近新添地图点集中移除他
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)//若当前关键帧序号与地图点第一个关键帧序号差大于等于3，从最近新添地图点集中移除他
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}


/*
1.取出与当前关键帧最佳共视的若干共视帧
2.对那些共视帧进行迭代操作
    2.1 求取当前帧与共视帧的位移，若不是单目，则如果位移小于相机基线，跳过此共视帧继续处理下一帧；若是单目，取出共视帧中所有地图点在共视帧坐标系下的深度，
        若位移与深度中值比例太小，则跳过此帧继续处理下一帧
    2.2 利用r,t 以及基础矩阵与外参关系求当前帧与共视帧之间的基础矩阵，求取满足对极约束的当前帧与共视帧之间的匹配对
    2.3 对于所有匹配对
        2.3.1 分别求当前帧与共视帧图像坐标u,v恢复出的射线在世界坐标系下的方向，然后核对两射线夹角
            2.3.1.1 若夹角较大且小于90度，用直接三角化的方法解关于三维点的方程
            2.3.1.2 如果当前帧为双目且余弦值较小，则利用当前帧双目的已知深度反求相机系下坐标进一步求世界系下坐标
            2.3.1.3 如果共视帧为双目且余弦值较小，则利用共视帧双目的已知深度反求相机系下坐标进一步求世界系下坐标
            2.3.1.4 若不存在双目且视差较小，则不处理这对匹配
        2.3.2 利用rt将三维点坐标从世界坐标分别转换到当前帧与共视帧坐标系下，若在某坐标系下深度为负，则跳过这对匹配继续处理下一对
        2.3.3 若相机1即当前帧非双目，世界坐标系下坐标重投影到相机1图像坐标系下的像素坐标，求原始像素坐标与重投影坐标的距离，若距离较大，跳过这对匹配继续处理下一对
            若相机1为双目，世界坐标系下坐标重投影到相机1左目图像坐标系下的像素坐标，利用深度与时差关系求取视差得到右目冲投影坐标，计算左右目原始像素坐标与
            重投影坐标的距离，若距离较大，跳过这对匹配继续处理下一对
            对相机2即共视帧进行相同的操作
        2.3.4 进行一些更新点之类的处理
*/
void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph   在共视图中恢复一些临近关键帧，若单目取20帧否则取10帧
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);//取出当前关键帧最佳的nn个共视关键帧

    ORBmatcher matcher(0.6,false);

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();//取出当前关键帧中世界坐标系到相机坐标系的旋转
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();//取出当前关键帧中世界坐标系到相机坐标系的平移
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));//利用旋转和平移组成适用于其次坐标的3*4转换矩阵
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();//取出相机中心在世界坐标系下的坐标

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew=0;

    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)//迭代共视的临近关键帧
    {
        if(i>0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();//取出共视帧相机中心在世界坐标系下的坐标
        cv::Mat vBaseline = Ow2-Ow1;//求取当前关键帧与当前共视帧之间的位移
        const float baseline = cv::norm(vBaseline);

        if(!mbMonocular)//若不是单目的情况
        {
            if(baseline<pKF2->mb)//若位移小于相机基线，则跳过此共视帧继续处理下一个共视帧
            continue;
        }
        else//若为单目情况
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);//获得共视帧所有地图点在相机坐标系下深度的中值
            const float ratioBaselineDepth = baseline/medianDepthKF2;//计算位移与深度中值的比例，若比例太小则跳过此帧继续处理后续共视帧

            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix  计算当前帧与共视帧的基础矩阵
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;//当前帧中匹配点的序号
            const int &idx2 = vMatchedIndices[ikp].second;//共视帧中匹配点的序号

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];//取出单目或者双目左图的u,v坐标
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];//取出右图的u坐标
            bool bStereo1 = kp1_ur>=0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays
            /*
            对于以图像中心点为原点的图像平面坐标（m,n）与空间点(X,Y,Z)的关系为
            m/X=n/Y=1/Z
            对于以图像中心点为原点的图像平面坐标（m,n)与以左上角为原点的图像坐标(u,v)关系为
            fx*m+cx=u   fy*n+cy=v
            由以上三式可得X=(Z*(u-cx))/fx    Y=(Z*(v-cy))/fy 可根据深度恢复三维点坐标
            对于归一化相机坐标系 Z=1，则 X=(u-cx)/fx  Y=(v-cy)/fy
            */
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);//匹配点在当前帧归一化相机坐标系下的坐标
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);//匹配点在共视帧归一化相机坐标系下的坐标

            cv::Mat ray1 = Rwc1*xn1;//求取从当前帧u,v恢复出的射线在世界坐标系下的方向
            cv::Mat ray2 = Rwc2*xn2;//求取从共视帧u,v恢复出的射线在世界坐标系下的方向
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));//计算两射线夹角余弦值

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)//若当前帧为双目 若匹配点在基线中垂线上，则左右目夹角为2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1])
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)//若共视帧为双目
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);//在0到180度范围内，角度越大余弦越小，取两个关键帧中左右目余弦较小者

            cv::Mat x3D;
            /*
            若满足以下三个条件：
            1.从当前帧u,v恢复出的射线在世界坐标系下的方向与从共视帧u,v恢复出的射线在世界坐标系下的方向夹角较大
            2.余弦值大于0
            3.当前帧与共视帧其中一个为单目或余弦值较小
            则利用两个相机归一化相机坐标系与世界坐标系的线性关系列出线性方程组然后用SVD解该方程组
            */
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method   Tcw1与Tcw2为3*4矩阵
                /*
                相机归一化坐标系与世界坐标下齐次坐标系关系为(a与lambda为与尺度相关的常量)
                            |X|                    |lambda*X| 
                  |m|       |Y|          |m|       |lambda*Y|    |m|
                a*|n| = Tcw*|Z|          |n| = Tcw*|lambda*Z|    |n| = Tcw*P 
                  |1|       |1|          |1|       |lambda*1|    |1|
                则
                m=Tcw.row(0)*P  (1)
                n=Tcw.row(1)*P  (2)
                1=Tcw.row(2)*P  (3)
                (3)*m-(1) 得  m*Tcw.row(2)*P-Tcw.row(0)*P = 0
                (3)*n-(2) 得  n*Tcw.row(2)*P-Tcw.row(1)*P = 0
                即 （m*Tcw.row(2)-Tcw.row(0)）*P = 0
                   （n*Tcw.row(2)-Tcw.row(1)）*P = 0
                */
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);//利用SVD分解求取线性方程组的解

                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);//除去lambda恢复尺度

            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)//如果当前帧为双目且余弦值较小，则利用当前帧双目的已知深度反求相机系下坐标进一步求世界系下坐标
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);                
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)//如果共视帧为双目且余弦值较小，则利用共视帧双目的已知深度反求相机坐标系下坐标进一步求取世界系下的坐标
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax    如果没有双目且视差较小，则不处理这对匹配

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);   //利用世界坐标系到相机坐标系的转换关系，用世界系下的坐标求当前帧相机系下深度，若深度为负则放弃这对匹配点
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);   //利用世界坐标系到相机坐标系的转换关系，用世界系下的坐标求共视帧相机系下深度，若深度为负则放弃这对匹配点
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];//octive为金字塔的层数
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;//求出世界坐标系下坐标转换得到的相机1坐标系下坐标

            if(!bStereo1)//如果相机1不是双目
            {
                float u1 = fx1*x1*invz1+cx1;//世界坐标系下坐标重投影到相机1图像坐标系下的像素坐标
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;//求重投影得到的像素坐标与原像素坐标的差
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            else
            {
                //世界坐标系下坐标重投影到左目图像坐标系下像素坐标，并利用视差与深度的关系将左目坐标传递到右目
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1//深度 z=f*baseline/d   视差 d=f*baseline/z
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];//octive为金字塔的层数
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            cv::Mat normal1 = x3D-Ow1;//三维点到相机1光心的距离
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);//三维点到相机2光心的距离

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            //const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);

            pMP->AddObservation(mpCurrentKeyFrame,idx1);//增加地图点被当前帧观测到的记录            
            pMP->AddObservation(pKF2,idx2);//增加地图点被共视帧观测到的记录

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);//将此地图点增加到当前帧的地图点点集中
            pKF2->AddMapPoint(pMP,idx2);//将此地图点增加到共视帧的地图点点集中

            pMP->ComputeDistinctiveDescriptors();
            /*对地图点对应的所有帧，在每帧中求该点描述子，然后分别求该描述子与其它所有帧描述子的距离，对于某帧，若其与其他所有帧描述子距离中值最小，则
            选择该帧描述子作为地图点的描述子*/

            pMP->UpdateNormalAndDepth();//更新地图点点到相机光心的向量及深度

            mpMap->AddMapPoint(pMP);//在地图中加入这个地图点
            mlpRecentAddedMapPoints.push_back(pMP);//在近期加入地图的点集中加入这个地图点

            nnew++;
        }
    }
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);//取出一些与当前帧最佳共视的共视帧
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)//循环这些共视帧
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)//若共视帧为坏帧或共视帧的融合目标帧为当前帧，则跳过此共视帧
            continue;
        vpTargetKFs.push_back(pKFi);//将共视帧加入目标帧集合中
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;//设置共视帧的融合目标帧为当前帧

        // Extend to some second neighbors 扩展一些二级近邻
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);//取共视帧的共视帧
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            //若共视帧的共视帧为坏帧，或共视帧的共视帧的融合目标帧为当前帧，或共视帧的共视帧为当前帧，则跳过这个共视帧的共视帧
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);//将共视帧的共视帧加入目标帧集合中
        }
    }


    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        matcher.Fuse(pKFi,vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

/*利用rt以及基础矩阵与rt关系求基础矩阵*/
cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        if(nObs>=thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }  

        if(nRedundantObservations>0.9*nMPs)
            pKF->SetBadFlag();
    }
}

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1) // wait until the mbResetRequested to turn false
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
