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

/*
1.Bow字典建立
（1）从训练图像离线抽取特征
（2）将抽取特征用kmeans++聚类，将描述子空间分为k类
（3）将划分的每个子空间继续用kmeans++聚类
（4）按上述循环将描述子建立树形结构
idf=log(N/ni) 每个叶子即每个word出现频率越高，区分度越小

2.在线更新字典树
当在字典树中插入一幅新图像It,在图像中提取的描述子仅按汉明距离从根节点逐渐向下到叶子节点，可得每个叶子节点即每个word在It中出现的频率
tf(i,It) = niIt/nIt  其中niIt为word在图像中出现的次数，nIt为图像中描述子总数
在树构建过程中，每个叶子节点存储了inverse index，存储了到达叶子节点的图像It的ID以及Word在It中描述vector中第i维的值
vt(i)=tf(i,It)*idf(i)
对一幅图像的所有描述子进行上述操作，可得每个word的值，将这些值构成图像的描述向量
对两图比较相似度时，s(v1,v2)=1 - 1/2(v1/|v1|-v2/|v2|)   相似度越高，s值越大

*/




#include "ORBmatcher.h"

#include<limits.h>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>

using namespace std;

namespace ORB_SLAM2
{

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}

/*
 * 1. set nmatches=0. if th is not 1, set bFactor tobe true; else set bFactor tobe false
 * 2. for every mappoint in the input vector vpMapPoints
 *    2.1 get the mappoint, if the mbTrackInView of the mappoint is false, continue to process next mappoint; if the mappoint is bad, continue to process next
 *        mappoint
 *    2.2 get the track scale level of the mappoint as nPredictedLevel
 *    2.3 if the cos value of the mappoint is bigger than threshold, set r=2.5; else set r=4. If bFactor is true, multiply r by th
 *    2.4 get vector of indices of the keypoints in F in the certain pixel range and certain levels of the pyramid as vIndices, if cannot get any index in the 
 *        above step,continue to process next mappoint
 *    2.5 get the descriptor of the mappoint
 *    2.6 for every candidate match in vIndices
 *        2.6.1 get its index in frame F, get the corresponding map point in frame F, if it is not a null pointer and the observations of the mapppoint is 
 *              bigger than 0, continue to process next point
 *        2.6.2 if the muvRight of F is bigger than 0, compute error as absolute mTrackProjXR of mappoint minus muvRight of F, if error is bigger than 
 *              the threshold, continue to process next candidate
 *        2.6.3 get the descriptor of the candidate, compute the distance between the mappoint descriptor and candidate descriptor, find the smallest and 
 *              second smallest distance, record the idx, level and distance of the smallest; and record distance and level for the second smallest
 *    2.7 if the smallest distance is no more than the threshold, if level of smallest distance is equal to level of second smallest and smallest diatance is
 *        bigger than a param multiply the second smallest distance, continue to process next mappoint
 *        set the mappoint of the smallest index tobe the current mappoint and add nmatches by 1 
 * 3. return how many mappoints in the inpput vector find match in frame F, i.e. nmatches
 */
/*
 * use the projected coordinate of the mappoint to find candidate for it in the input frame F, if it really matches, set the mappoint of frame F tobe the 
 * mappoint in the input vector vpMapPoints, finally after process all mapppoints in vpMapPoints, return the number of matches found
 */
int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th)//图像帧与地图点之间
{
    int nmatches=0;

    //if th is not 1, set bFactor tobe true, else set bFactor tobe false
    const bool bFactor = th!=1.0; //th不为1时，bFactor为true否则为false

    // for all map points in the input vector
    for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)//对于地图点集合中的每一个地图点
    {
        MapPoint* pMP = vpMapPoints[iMP];
        if(!pMP->mbTrackInView)// if the mbTrackInView of the mappoint is false, continue to process next mappoint
            continue;

        if(pMP->isBad())// if the mappoint is bad, continue to process next point
            continue;

        // get the track scale level of the mappoint
        const int &nPredictedLevel = pMP->mnTrackScaleLevel;

        // The size of the window will depend on the viewing direction
        // if the cos value is bigger than threshold, set r=2.5; else set r=4
        float r = RadiusByViewingCos(pMP->mTrackViewCos);//若cos较大r为2.5，较小r为4

        if(bFactor)// if bFactor is true, multiply r by th
            r*=th;
        //取出在金字塔某些层和图片像素范围内的关键点，返回的是包含关键点下标的向量
        // get vector of indices of the keypoints in the certain pixel range and certain levels in the pyramid
        const vector<size_t> vIndices =
                F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

        if(vIndices.empty())// if cannot get some candidate matches in F, continue to process next mappoint
            continue;

        // get the descriptor of the mappoint
        const cv::Mat MPdescriptor = pMP->GetDescriptor();//取出地图点描述子

        int bestDist=256;
        int bestLevel= -1;
        int bestDist2=256;
        int bestLevel2 = -1;
        int bestIdx =-1 ;

        // Get best and second matches with near keypoints
        // 找到地图点重投影点附近的图片像素最佳与次佳匹配
        // for every candidate match in vIndices
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)//对于所有与地图点投影接近的图片像素点
        {
            const size_t idx = *vit;// get its index in frame F

            // get the corresponding map point in frame F, if it is not a null pointer and the observations of the mapppointis bigger than 0, continue
            // to process next point
            if(F.mvpMapPoints[idx])//取出图片下标对应的地图点，若该地图点的观测次数大于0，则继续处理下一个图片点
                if(F.mvpMapPoints[idx]->Observations()>0)
                    continue;

            // if the muvRight of F is bigger than 0, compute error as absolute mTrackProjXR of mappoint minus muvRight of F, if error is bigger than 
            // the threshold, continue to process next candidate
            if(F.mvuRight[idx]>0)//若帧中存在右视图，若重投影右视图横坐标与右目横坐标差的绝对值过大，则继续处理下一个图像点
            {
                const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
                if(er>r*F.mvScaleFactors[nPredictedLevel])
                    continue;
            }

            // get the descriptor of the candidate
            const cv::Mat &d = F.mDescriptors.row(idx);//取出图像点描述子

            //compute the distance between the mappoint descriptor and candidate descriptor
            const int dist = DescriptorDistance(MPdescriptor,d);//计算图像点描述子与地图点描述子的汉明距离

            // find the smallest and second smallest distance, record the idx, level and distance of the smallest; and record distance and level for 
            // the second smallest
            if(dist<bestDist)//若距离小于当前最小距离
            {
                //将当前最小距离的相关参数赋值给次小距离，将当前点相关参数赋值给最小距离
                bestDist2=bestDist;
                bestDist=dist;
                bestLevel2 = bestLevel;
                bestLevel = F.mvKeysUn[idx].octave;
                bestIdx=idx;
            }
            else if(dist<bestDist2)//若距离大于当前最小距离但是小于当前次小距离，将当前点相关参数赋值给次小距离
            {
                bestLevel2 = F.mvKeysUn[idx].octave;
                bestDist2=dist;
            }
        }

        /* if the smallest distance is no more than the threshold, if level of smallest distance is equal to level of second smallest and smallest diatance is
           bigger than a param multiply the second smallest distance, continue to process next mappoint
           set the mappoint of the smallest index tobe the current mappoint and add nmatches by 1 */
        // Apply ratio to second match (only if best and second are in the same scale level)
        if(bestDist<=TH_HIGH)//若最佳距离小于一定阈值
        {
            if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)//若最佳距离与次佳距离的像素点在金字塔同一层，且最佳距离大于一定比例的次佳距离,则继续循环下一个地图点
                continue;

            F.mvpMapPoints[bestIdx]=pMP;//将帧中该像素对应的地图点记为当前地图点
            nmatches++;//增加匹配次数
        }
    }

    return nmatches;//返回地图点向量中在图片找到匹配的地图点个数  return how many mappoints in the inpput vector find match in frame F
}

float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998)
        return 2.5;
    else
        return 4.0;
}


bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const KeyFrame* pKF2)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    /*
              |F00 F01 F02|
    |x y 1| * |F10 F11 F12| = |x*F00+y*F10+1*F20  x*F01+y*F11+1*F21  x*F02+y*F12+1*F22|
              |F20 F21 F22|
    */
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

    const float num = a*kp2.pt.x+b*kp2.pt.y+c; //求kp2到kp1投影到2图像平面的极线的距离  |ax+by+c|/sqrt(a*a+b*b)

    const float den = a*a+b*b;

    if(den==0)
        return false;

    const float dsqr = num*num/den;

    return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];//若距离较小则返回真，否则假
}

/*
 * 1. get all map points in the key frame
 * 2. initialize a vector of null map point pointers whose size is the number of keypoints in the  named vpMapPointMatches
 * 3. for the same treenodes in keyframes and frames,get their corresponding indexes of features in the image as vIndicesKF and vIndicesF
 *    3.1 for every index in vIndicesKF
 *        3.1.1 get the corresponding map point,if it is a null pointer or bad point, continue toprocess the next feature
 *        3.1.2 get the descriptor of that feature point, compute the smallest and second smallest feature point among all feature points in vIndicesF
 *        3.1.3 if the smallest distance is smaller than a certain threshold and is smaller than a certain ratio multiply the second smallest distance, set
 *              the mappoint with the index correspond to the smallest distance in vpMapPointMatches tobe the mapppoint in keyframe
 *        3.1.4 devide the 360 degree into 30 parts,each one represents 12 degree, compute the angle distance of the feature in keyframe and frame, push the index
 *              corresponding to the smallest distance into the histogram's  
 *  4. if the element in vpMapPointMatches do not belong to the first,second and third number part in the histogram, set the pointer to null
 */
int ORBmatcher::SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)//关键帧与图像帧之间
{
    const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();//取出关键帧中所有地图点

    vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));//将向量初始化为大小为帧中关键点个数，元素为空指针的向量

    // class FeatureVector:public std::map<NodeId, std::vector<unsigned int> > NodeId is the index of the node in the 
    // tree, std::vector<unsigned int> retains the indexes of the features
    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

    int nmatches=0;

    vector<int> rotHist[HISTO_LENGTH];//const int ORBmatcher::HISTO_LENGTH = 30;       //指的是有HISTO_LENGTH个vector<int>
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();//关键帧的特征向量
    DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();//图像帧的特征向量
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
    DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

    while(KFit != KFend && Fit != Fend)
    {
        /*class FeatureVector: public std::map<NodeId, std::vector<unsigned int> >  NodeId为词汇树中节点编号  std::vector<unsigned int>存的是本地特征的下标
        */
        if(KFit->first == Fit->first)
        {
            const vector<unsigned int> vIndicesKF = KFit->second;//取出相同词汇在关键帧中的下标
            const vector<unsigned int> vIndicesF = Fit->second;//取出相同词汇在图像帧中的下标

            for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)//遍历关键帧中的地图点
            {
                const unsigned int realIdxKF = vIndicesKF[iKF];

                MapPoint* pMP = vpMapPointsKF[realIdxKF];

                if(!pMP) //若地图点指针为空或地图点为坏点，则继续处理下一个地图点
                    continue;

                if(pMP->isBad())
                    continue;                

                const cv::Mat &dKF= pKF->mDescriptors.row(realIdxKF);//取出地图点对应的描述子

                int bestDist1=256;
                int bestIdxF =-1 ;
                int bestDist2=256;

                for(size_t iF=0; iF<vIndicesF.size(); iF++) //遍历图像帧中地图点
                {
                    const unsigned int realIdxF = vIndicesF[iF];

                    if(vpMapPointMatches[realIdxF])//若图像帧此点已找到匹配地图点，则跳过这个图像帧的点继续处理下一点
                        continue;

                    const cv::Mat &dF = F.mDescriptors.row(realIdxF);//取出图像点对应的描述子

                    const int dist =  DescriptorDistance(dKF,dF); //求两个描述子的距离

                    if(dist<bestDist1)//若距离小于当前最小距离，则将当前次小距离更新为当前最小距离，随后更新最小距离
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdxF=realIdxF;
                    }
                    else if(dist<bestDist2)//若距离大于当前最小距离，小于当前次小距离，则用距离更新次小距离值
                    {
                        bestDist2=dist;
                    }
                }//结束对图像帧地图点的遍历

                if(bestDist1<=TH_LOW)//若最小距离小于阈值
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))//若最小距离小于一定比例的次小距离
                    {
                        vpMapPointMatches[bestIdxF]=pMP;//则图像帧此下标对应的最佳匹配地图点为关键帧中取出的地图点

                        const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];//记从关键帧中取出的点为关键点

                        if(mbCheckOrientation)
                        {
                            float rot = kp.angle-F.mvKeys[bestIdxF].angle;//记旋转角度为关键帧中取出的关键点角度减图像帧中取出的关键点角度
                            if(rot<0.0)//若角度小于0，则角度增大360度
                                rot+=360.0f;
                            int bin = round(rot*factor);//const float factor = 1.0f/HISTO_LENGTH;  //const int ORBmatcher::HISTO_LENGTH = 30;
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdxF);//在旋转直方图中对应此旋转角度的那栏加入这个图像帧关键点下标
                        }
                        nmatches++;
                    }
                }

            }//结束对关键帧地图点的遍历

            KFit++;
            Fit++;
        }
        else if(KFit->first < Fit->first) //若关键帧节点序号小于图像帧节点序号
        {
            KFit = vFeatVecKF.lower_bound(Fit->first); //lower_bound(k)返回一个迭代器，指向第一个关键字不小于k的元素
        }
        else
        {
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }//结束对featurevector的循环，这个循环主要处理两个帧含有相同word的情况


    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);//计算直方图每栏元素数的三个峰值（元素数越大说明这个角度对应的匹配越多）

        for(int i=0; i<HISTO_LENGTH; i++)//对于直方图的每一栏
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)//对于直方图每一栏中的每个元素
            {
                vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);//若这个元素对应的地图点为空指针，则减小匹配数
                nmatches--;
            }
        }
    }

    return nmatches;//返回图片帧在关键帧中找到的匹配点个数
}

/*关键帧与地图点之间， 用于回环*/
int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const vector<MapPoint*> &vpPoints, vector<MapPoint*> &vpMatched, int th)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);//为什么计算除法，个人认为正交矩阵的sRcw这个值本来就是1
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;//相机光心在世界坐标系下的坐标

    // Set of MapPoints already found in the KeyFrame  已经在关键帧中找到的地图点的集合
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

    int nmatches=0;

    // For each Candidate MapPoint Project and Match   对于每一个候选地图点进行投影并匹配
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)//对候选地图点进行循环
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP)) //如果候选点是坏点或已经被找到，则继续处理下一个候选点
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();//取出候选点的三维坐标

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;//利用世界坐标系与相机坐标系的rt关系求候选点在相机坐标系下的三维坐标

        // Depth must be positive 若相机坐标系下深度为负，则继续处理下一个候选点
        if(p3Dc.at<float>(2)<0.0)
            continue;

        // Project into Image
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy; //利用相机内参与深度，将候选点从相机坐标系转到图像坐标系

        // Point must be inside the image 如果投影后候选点处于图像范围外，则继续处理下一点
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;//世界坐标系下候选点坐标与相机光心坐标的差
        const float dist = cv::norm(PO);//世界坐标系下候选点与相机光心的距离

        if(dist<minDistance || dist>maxDistance)//若距离过大或过小，继续处理下一个候选点
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();//GetNormal得到的是所有观测到该点的相机中心与该点在世界坐标系下连线的单位向量均值

        if(PO.dot(Pn)<0.5*dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);//找到重投影坐标一定范围内的关键点

        if(vIndices.empty())//若关键点集合为空，则继续处理下一个候选地图点
            continue;

        // Match to the most similar keypoint in the radius 取出地图点的描述子
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)//对于重投影点附近一定范围的关键点
        {
            //vpMatched中下标为关键帧中关键点的下标
            const size_t idx = *vit;
            if(vpMatched[idx])//若该关键点已经有对应的匹配地图点，则继续处理下一个关键点
                continue;

            const int &kpLevel= pKF->mvKeysUn[idx].octave;//关键帧关键点所处的金字塔层数

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)//若关键帧关键点金字塔层数与地图点金字塔层数差距较大，则跳过此关键点继续处理下一关键点
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);//取出关键帧该关键点的描述子

            const int dist = DescriptorDistance(dMP,dKF);//计算关键帧与地图点的关键点描述子

            if(dist<bestDist)//求取所有近邻关键点中最小的描述子距离
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_LOW)//若最小距离小于阈值，则记该关键点的匹配地图点为当前候选地图点，且增加一个匹配次数
        {
            vpMatched[bestIdx]=pMP;
            nmatches++;
        }

    }//结束对候选地图点的循环

    return nmatches;//返回匹配点个数
}

/*单目情况下地图初始化的匹配*/
int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
{
    int nmatches=0;
    vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);//大小为帧1关键点个数，初值为-1

    vector<int> rotHist[HISTO_LENGTH];//直方图具有HISTO_LENGTH栏，每栏是一个整型的向量
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);//大小为帧2关键点个数，初值为最大整型值
    vector<int> vnMatches21(F2.mvKeysUn.size(),-1);//大小为帧2关键点个数，初值为-1

    for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++) //对帧1中所有关键点
    {
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave;//取出关键点坐在金字塔层数
        if(level1>0) //若层数大于0，则继续处理下一个帧1中的关键点
            continue;

        vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);//取出帧2中的近邻关键点

        if(vIndices2.empty())//若无近邻关键点，则继续处理下一个帧1中的关键点
            continue;

        cv::Mat d1 = F1.mDescriptors.row(i1);//取出帧1中当前关键点的描述子

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)//对于帧2中的近邻关键点
        {
            size_t i2 = *vit;

            cv::Mat d2 = F2.mDescriptors.row(i2);//取出近邻关键点的描述子

            int dist = DescriptorDistance(d1,d2);//求帧1当前关键点与帧2近邻关键点描述子距离

            if(vMatchedDistance[i2]<=dist)//若已有帧2此近邻点描述子距离小于当前距离，则继续处理下一个近邻点
                continue;

            if(dist<bestDist)//若距离小于最佳距离，将最佳距离赋值给次佳距离，将距离赋值给最佳距离
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            }
            else if(dist<bestDist2)//若距离大于最佳距离小于次佳距离，将距离赋值给次佳距离
            {
                bestDist2=dist;
            }
        }

        if(bestDist<=TH_LOW)//若最佳距离小于一定阈值
        {
            if(bestDist<(float)bestDist2*mfNNratio)//若最佳距离小于一定比例的次佳距离
            {
                if(vnMatches21[bestIdx2]>=0)
                //若先前帧2中的这个点已经进行过匹配，则说明这个帧2近邻关键点bestIdx2原先进行过成功匹配过帧1中关键点i1pre,但是此时bestIdx2匹配的是i1，所以将i1pre对应点擦除
                {
                    vnMatches12[vnMatches21[bestIdx2]]=-1;
                    nmatches--;//当前匹配代替了原先匹配，所以匹配次数-1
                }
                vnMatches12[i1]=bestIdx2;//帧1的关键点i1对应的是帧2中的关键点bestIdx2
                vnMatches21[bestIdx2]=i1;//帧2的关键点bestIdx2对应的是帧1中的关键点i1
                vMatchedDistance[bestIdx2]=bestDist;//bestIdx2的最佳匹配距离为bestDist
                nmatches++;//增加匹配次数

                if(mbCheckOrientation)
                {
                    float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;//帧1关键点角度-帧2关键点角度
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = round(rot*factor);
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(i1);//将帧1中的点下标存放在直方图对应的栏中
                }
            }
        }

    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)//若i是直方图三个峰值中的一个，则跳过继续处理下一栏
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)//对于直方图第i栏中所有帧1中的关键点下标
            {
                int idx1 = rotHist[i][j];
                if(vnMatches12[idx1]>=0)//若该点在帧2中存在对应点，则标记无这组对应关系且减小匹配次数
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }

    }

    //Update prev matched
    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)//对于帧1中所有关键点
        if(vnMatches12[i1]>=0)//若在帧2中存在对应关键点
            vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;//更新帧1中i1的预先匹配为最新计算的匹配点

    return nmatches;
}

int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
{
    const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;//取出去畸变的关键点
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;//取出Bow的特征向量
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();//取出对应匹配的地图点
    const cv::Mat &Descriptors1 = pKF1->mDescriptors;//取出描述子

    const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const cv::Mat &Descriptors2 = pKF2->mDescriptors;

    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));//大小为关键帧1中地图点个数，初始化为空指针
    vector<bool> vbMatched2(vpMapPoints2.size(),false);//大小为关键帧2中地图点个数，初始化为false

    vector<int> rotHist[HISTO_LENGTH];//声明一个有HISTO_LENGTH栏的直方图，直方图每一栏为一个整型的向量
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    int nmatches = 0;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it != f1end && f2it != f2end)//若关键帧1和关键帧2的featurevector迭代器都没到尾后迭代器
    {
        if(f1it->first == f2it->first)//若两帧中包含树中的同一个节点
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)//对于对应树中该节点的关键帧1的所有关键点下标
            {
                const size_t idx1 = f1it->second[i1];

                MapPoint* pMP1 = vpMapPoints1[idx1];//取出对应地图点的指针，若指针为空或地图点是坏点，则继续处理对应树中该节点的关键帧1中的其他关键点
                if(!pMP1)
                    continue;
                if(pMP1->isBad())
                    continue;

                const cv::Mat &d1 = Descriptors1.row(idx1);//取出关键点对应的描述子

                int bestDist1=256;
                int bestIdx2 =-1 ;
                int bestDist2=256;

                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)//对于对应树中该节点的关键帧2的所有关键点下标
                {
                    const size_t idx2 = f2it->second[i2];

                    MapPoint* pMP2 = vpMapPoints2[idx2];

                    if(vbMatched2[idx2] || !pMP2)//若关键帧2中该关键点已经匹配或取出的地图点指针为空或地图点为坏点，则跳过此点继续处理关键帧2中下一个关键点
                        continue;

                    if(pMP2->isBad())
                        continue;

                    const cv::Mat &d2 = Descriptors2.row(idx2); //取出关键帧2中关键点对应的描述子

                    int dist = DescriptorDistance(d1,d2); //计算关键帧1与关键帧2中描述子的距离

                    if(dist<bestDist1) //计算最小距离与次小距离双重阈值
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdx2=idx2;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<TH_LOW)//若最小距离小于阈值
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))//最小距离小于一定比例的次小距离
                    {
                        vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                        vbMatched2[bestIdx2]=true;

                        if(mbCheckOrientation)
                        {
                            float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;//求对应的关键点在两视图中视角的差
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);//计算直方图的三个峰值

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)//若对应直方图三个峰值中的一栏
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)//若匹配关系不在三个峰值中的任意一个，则清除该匹配关系并减小匹配次数
            {
                vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

/*匹配以三角化新的点，检查对极约束*/
int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                       vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo)
{    
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;//分别取出两个关键帧的FeatureVector
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    //Compute epipole in second image
    cv::Mat Cw = pKF1->GetCameraCenter();//关键帧1相机光心在世界坐标系下的坐标
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    cv::Mat C2 = R2w*Cw+t2w;//关键帧1相机光心在关键帧2坐标系下的坐标
    /*
    关键帧1在关键帧2归一化坐标系下的坐标为 （ C2(0)/C2(2) C2(1)/C2(2) 1 ）
    在关键帧2图像坐标系下的坐标为  (fx*C2(0)/C2(2)+cx  fy*C2(1)/C2(2)+cy)
    */
    const float invz = 1.0f/C2.at<float>(2);
    const float ex =pKF2->fx*C2.at<float>(0)*invz+pKF2->cx;//关键帧1光心在关键帧2图像坐标系下的坐标
    const float ey =pKF2->fy*C2.at<float>(1)*invz+pKF2->cy;

    // Find matches between not tracked keypoints  寻找未追踪关键点的匹配
    // Matching speed-up by ORB Vocabulary         用orb词汇表加速匹配
    // Compare only ORB that share the same node   只比较在树中相同节点的orb

    int nmatches=0;
    vector<bool> vbMatched2(pKF2->N,false);//大小为关键帧2中关键点个数，初值为false
    vector<int> vMatches12(pKF1->N,-1);//大小为关键帧1中关键点个数，初值为-1

    vector<int> rotHist[HISTO_LENGTH];//含有HISTO_LENGTH栏的直方图，每栏为一个整型的向量
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();//对于FeatureVector中的每个节点
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it!=f1end && f2it!=f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)//对于该树节点对应的关键帧1中的所有关键点下标
            {
                const size_t idx1 = f1it->second[i1];
                
                MapPoint* pMP1 = pKF1->GetMapPoint(idx1);//取出关键帧1关键点对应的地图点
                
                // If there is already a MapPoint skip
                if(pMP1)//如果关键帧1的关键点已经有对应的地图点
                    continue;

                const bool bStereo1 = pKF1->mvuRight[idx1]>=0;//查看关键帧1该关键点是否为双目

                if(bOnlyStereo)//若只处理双目但关键帧1此关键点不为双目，则跳过这个关键点继续处理下一个
                    if(!bStereo1)
                        continue;
                
                const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];//取出关键帧1的关键点
                
                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);//取出关键帧1关键点的描述子
                
                int bestDist = TH_LOW;
                int bestIdx2 = -1;
                
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)//对于该树节点对应的关键帧2中的所有关键点下标
                {
                    size_t idx2 = f2it->second[i2];
                    
                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2);//取出该下标对应的地图点指针
                    
                    // If we have already matched or there is a MapPoint skip如果该关键点已经有匹配或已经有对应地图点，跳过本点继续处理关键帧2中下一个关键点
                    if(vbMatched2[idx2] || pMP2)
                        continue;

                    const bool bStereo2 = pKF2->mvuRight[idx2]>=0;//查看关键帧2该关键点是否为双目

                    if(bOnlyStereo)
                        if(!bStereo2)
                            continue;
                    
                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);//取出该关键点对应的描述子
                    
                    const int dist = DescriptorDistance(d1,d2);//计算关键帧1与关键帧2关键点描述子距离
                    
                    if(dist>TH_LOW || dist>bestDist)//若距离较大则继续处理关键帧2下一关键点
                        continue;

                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];//取出关键帧2中的关键点

                    if(!bStereo1 && !bStereo2)//如果关键帧1与关键帧2均非双目，若对极点距离帧2中关键点过近，则继续处理关键帧2中下一个关键点
                    {
                        const float distex = ex-kp2.pt.x;
                        const float distey = ey-kp2.pt.y;
                        if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])//？？？？是代表三维点到成像平面距离近吗
                            continue;
                    }

                    if(CheckDistEpipolarLine(kp1,kp2,F12,pKF2))//求关键点到对极线的点到直线距离
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }//结束对该节点对应的关键帧2中关键点的循环
                
                if(bestIdx2>=0)
                {
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];
                    vMatches12[idx1]=bestIdx2;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = kp1.angle-kp2.angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }//结束对该树节点在关键帧1中关键点下标的循环

            f1it++;
            f2it++;
        }//endif f1it->first == f2it->first
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vMatches12[rotHist[i][j]]=-1;
                nmatches--;
            }
        }

    }

    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);//将求出的符合所有要求的匹配给入向量中

    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0)
            continue;
        vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
    }

    return nmatches;
}

//将地图点投影到关键帧
int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th)
{
    cv::Mat Rcw = pKF->GetRotation();
    cv::Mat tcw = pKF->GetTranslation();

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    cv::Mat Ow = pKF->GetCameraCenter();

    int nFused=0;

    const int nMPs = vpMapPoints.size();

    for(int i=0; i<nMPs; i++)//对于向量中的所有地图点
    {
        MapPoint* pMP = vpMapPoints[i];

        if(!pMP)//如果对应指针为空，则继续处理下一个地图点
            continue;

        if(pMP->isBad() || pMP->IsInKeyFrame(pKF))//如果地图点为坏点，或在本函数参数的关键帧中，则跳过此点继续处理下一个地图点
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();//取出地图点在世界坐标系下的三维坐标
        cv::Mat p3Dc = Rcw*p3Dw + tcw;//利用关键帧的rt关系将坐标从世界坐标系转到相机坐标系

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f) //若在相机坐标系下深度为负，则跳过这个地图点继续处理下一个地图点
            continue;
        //将点坐标从相机坐标系转换到归一化相机坐标系
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;
        //将点坐标从归一化相机坐标系的三维坐标转到图像平面的二维坐标
        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))//若投影点不在关键帧图像幅面内，则跳过此点继续处理下一个地图点
            continue;

        const float ur = u-bf*invz;//在右图中的横坐标为左图横坐标减视差

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;//世界坐标系下特征点到相机光心的距离
        const float dist3D = cv::norm(PO);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);//在关键帧中投影点附近查找关键点下标

        if(vIndices.empty())//若关键帧中投影点附近没有关键点，则跳过这个地图点继续处理下一个
            continue;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();//取出地图点的描述子

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)//对于地图点投影点附近的所有关键点
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF->mvKeysUn[idx];

            const int &kpLevel= kp.octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)//如果不在地图点所在金字塔的临近层，则继续处理下一个关键点
                continue;

            if(pKF->mvuRight[idx]>=0)//若关键帧为双目，求左目横纵坐标与右目横坐标三个坐标的误差
            {
                // Check reprojection error in stereo
                //u,v为重投影计算出的像素坐标、ur为重投影减视差影响得到的像素坐标
                //kpx\kpy\kpr为从关键点读出的像素坐标
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float &kpr = pKF->mvuRight[idx];
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float er = ur-kpr;
                const float e2 = ex*ex+ey*ey+er*er;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                    continue;
            }
            else
            {
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float e2 = ex*ex+ey*ey;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
                    continue;
            }

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);//取出关键帧中关键点的描述子

            const int dist = DescriptorDistance(dMP,dKF);//求关键帧关键点与地图点的描述子距离

            if(dist<bestDist)//求出与地图点描述子最相近的关键帧关键点
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }//结束对地图点投影点附近所有关键点的循环

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)//若最近距离小于阈值
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);//取出最佳距离的关键点对应的地图点
            if(pMPinKF)//如果关键点对应的地图点存在
            {
                if(!pMPinKF->isBad())//若地图点不是坏点
                {
                    if(pMPinKF->Observations()>pMP->Observations())//如果关键点对应的地图点被观测次数多于向量中地图点被观测次数，则用关键帧地图点代替向量中地图点
                        pMP->Replace(pMPinKF);
                    else//若关键点对应的地图点被观测次数少于向量中地图点被观测次数，则用向量中地图点代替关键帧地图点
                        pMPinKF->Replace(pMP);
                }
            }
            else//如果关键点对应的地图点不存在
            {
                pMP->AddObservation(pKF,bestIdx);//将关键帧及其点下标给到向量中地图点的对应
                pKF->AddMapPoint(pMP,bestIdx);//对关键帧添加这个向量中的地图点
            }
            nFused++;//融合次数加1
        }
    }//结束对地图点的循环

    return nFused;
}

int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;//关键帧相机光心在世界坐标系下的坐标

    // Set of MapPoints already found in the KeyFrame
    const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();//读出关键帧中已经包含的地图点

    int nFused=0;//初始化融合次数为0

    const int nPoints = vpPoints.size();

    // For each candidate MapPoint project and match 对于输入向量中的每个地图点
    for(int iMP=0; iMP<nPoints; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found 若地图点为坏点或者关键帧中已经包含这个地图点，则跳过这个点继续处理下一个地图点
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords. 获取地图点在世界坐标系下的三维坐标
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords. 将点从世界坐标系转到相机坐标系
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive 若在相机坐标系下深度为负，则继续处理下一个地图点
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        // Project into Image  将点投影到图像坐标系下
        const float invz = 1.0/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image   若投影点不在图像幅面内，则继续处理下一个地图点
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;//地图点到相机光心的距离
        const float dist3D = cv::norm(PO);

        if(dist3D<minDistance || dist3D>maxDistance)//若地图点到相机光心距离过大或过小，则跳过此点继续处理下一个地图点
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)//若角度过大，则跳过此点继续处理下一个地图点
            continue;

        // Compute predicted scale level
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);//在关键帧内地图点投影点附近寻找关键点

        if(vIndices.empty())//若在地图点投影点附近找不到关键点，则跳过此点继续处理下一个关键点
            continue;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)//对于地图点投影点附近的所有关键点
        {
            const size_t idx = *vit;
            const int &kpLevel = pKF->mvKeysUn[idx].octave;//关键帧中关键点所在金字塔层数

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)//若关键点金字塔层数与地图点相差较多，则继续处理下一个地图点
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);//取出关键点描述子

            int dist = DescriptorDistance(dMP,dKF);//计算关键点描述子与地图点描述子距离

            if(dist<bestDist)//取出与地图点描述子距离最小的关键点
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint， replace ；otherwise add new measurement
        if(bestDist<=TH_LOW)//若最小描述子距离小于阈值，
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);//取出关键帧关键点对应的地图点
            if(pMPinKF)//若关键点对应的地图点非空，
            {
                if(!pMPinKF->isBad())//若关键点对应的地图点不是坏点
                    vpReplacePoint[iMP] = pMPinKF;//用关键点对应的地图点替代向量中的地图点
            }
            else//若关键点对应的地图点为空
            {
                pMP->AddObservation(pKF,bestIdx);//将关键帧及下标添加到地图点的对应关系中
                pKF->AddMapPoint(pMP,bestIdx);//向关键帧的该下标添加地图点
            }
            nFused++;//融合次数加1
        }
    }//结束对向量中地图点的循环

    return nFused;
}

// Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3 [s12*R12|t12]
    // In the stereo and RGB-D case, s12=1
/*求两个关键帧之间用Sim3 [s12*R12|t12]转换的地图点的匹配，在双目与rgbd情况下s12为1
先将关键帧1地图点投影到关键帧2中，在投影点附近查找关键点，找出与地图点金字塔层数接近，描述子距离相近的关键点
再将关键帧2地图点投影到关键帧1中，在投影点附近查找关键点，找出与地图点金字塔层数接近，描述子距离相近的关键点
对于上述两次匹配关系的寻找，若找到的关系相互符合，则将这对匹配加入vpMatches12中*/
int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12,
                             const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th)
{
    const float &fx = pKF1->fx;//关键帧1的内参
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // Camera 1 from world 世界坐标系到关键帧1的变换关系
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    //Camera 2 from world 世界坐标系到关键帧2的变换关系
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    //Transformation between cameras
    cv::Mat sR12 = s12*R12;
    cv::Mat sR21 = (1.0/s12)*R12.t();
    cv::Mat t21 = -sR21*t12;

    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();//关键帧1对应的地图点集合
    const int N1 = vpMapPoints1.size();

    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();//关键帧2对应的地图点集合
    const int N2 = vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1,false);//大小为关键帧1中地图点的数量，初始化为false
    vector<bool> vbAlreadyMatched2(N2,false);//大小为关键帧2中地图点的数量，初始化为false

    for(int i=0; i<N1; i++)
    {
        MapPoint* pMP = vpMatches12[i];//对于关键帧1和关键帧2匹配的地图点指针
        if(pMP)//若指针非空
        {
            vbAlreadyMatched1[i]=true;//记关键帧1中该特征点匹配情况为已匹配
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);//取出该点在关键帧2中的下标
            if(idx2>=0 && idx2<N2)//若下标在合理范围内
                vbAlreadyMatched2[idx2]=true;//记关键帧2中该特征点匹配情况为已匹配
        }
    }

    vector<int> vnMatch1(N1,-1);
    vector<int> vnMatch2(N2,-1);

    // Transform from KF1 to KF2 and search
    for(int i1=0; i1<N1; i1++)//对于关键帧1中所有地图点
    {
        MapPoint* pMP = vpMapPoints1[i1];

        if(!pMP || vbAlreadyMatched1[i1])//如果地图点为空或者已经匹配，则继续处理关键帧1中下一个地图点
            continue;

        if(pMP->isBad())//若地图点为坏点，则继续处理关键帧1中下一个地图点
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();//取出地图点在世界坐标系下的坐标
        cv::Mat p3Dc1 = R1w*p3Dw + t1w;//将地图点从世界坐标系转换到相机1坐标系
        cv::Mat p3Dc2 = sR21*p3Dc1 + t21;

        // Depth must be positive
        if(p3Dc2.at<float>(2)<0.0)//若在相机2坐标系中深度为负，则跳过此点继续处理关键帧1中下一个地图点
            continue;

        const float invz = 1.0/p3Dc2.at<float>(2);
        const float x = p3Dc2.at<float>(0)*invz;
        const float y = p3Dc2.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;//将地图点从相机坐标系转换到图片坐标系下

        // Point must be inside the image 若投影的像素不在图片幅面内，则跳过此点继续处理下一个地图点
        if(!pKF2->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc2);//求地图点距离相机2光心的距离

        // Depth must be inside the scale invariance region  若地图点距离相机光心2距离太大或者太小，则跳过此点继续处理下一个地图点
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Compute predicted octave 计算地图点在关键帧2中的金字塔层数
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

        // Search in a radius 计算特征点的搜索范围
        const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);//取出关键帧2中投影点附近的关键点

        if(vIndices.empty())//如果投影点附近没有关键点，则跳过此地图点继续处理下一个地图点
            continue;

        // Match to the most similar keypoint in the radius 取出地图点的描述子
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)//对于投影点附近的所有关键点
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];//取出下标对应的关键点

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)//若关键点所在金字塔层数不在预测层数附近，则跳过此关键点继续处理下一个关键点
                continue;

            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);//取出关键点对应的描述子

            const int dist = DescriptorDistance(dMP,dKF);//计算关键点描述子与地图点描述子距离

            if(dist<bestDist)//取出最佳距离与最佳距离对应的下标
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }//结束对投影点附近关键点的遍历

        if(bestDist<=TH_HIGH)//若最佳距离小于阈值，记关键帧1中该点对应的匹配为关键帧2中最佳距离对应的下标
        {
            vnMatch1[i1]=bestIdx;
        }
    }//结束对关键帧1中所有地图点的遍历

    // Transform from KF2 to KF2 and search
    for(int i2=0; i2<N2; i2++)//对于关键帧2中的所有地图点
    {
        MapPoint* pMP = vpMapPoints2[i2];

        if(!pMP || vbAlreadyMatched2[i2])//若地图点为空或该点已经具有匹配关系，则跳过此点继续处理关键帧2中下一个地图点
            continue;

        if(pMP->isBad())//若地图点为坏点，则跳过这个地图点继续处理关键帧2中下一个地图点
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc2 = R2w*p3Dw + t2w;
        cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

        // Depth must be positive
        if(p3Dc1.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc1.at<float>(2);
        const float x = p3Dc1.at<float>(0)*invz;
        const float y = p3Dc1.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF1->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc1);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

    for(int i1=0; i1<N1; i1++)//若在两次来回寻找的过程中，关键帧1与关键帧2得到的匹配关系相符合
    {
        int idx2 = vnMatch1[i1];

        if(idx2>=0)
        {
            int idx1 = vnMatch2[idx2];
            if(idx1==i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];//将匹配关系添加到向量中
                nFound++;//将找到匹配的次数加1
            }
        }
    }

    return nFound;
}

/*两个帧之间的关系*/
/*
 * 1. compute the tlc using the last frame pose and the current frame pose. If the current camera pose in the last frame is bigger than the baseline
 *    and this is not the monocular case, set bForward tobe true; if the opposite of current camera in the last frame is bigger than the baseline and 
 *    this is not the monocular case, set bBackward tobe true
 * 2. for all keypoints in last frame, get the map points related with the keypoints in the last frame, if the map point is not a null pointer and the mappoint 
 *    in last frame is not marked as outlier
 *    2.1 get the 3D position in the world coordinate, use the pose of current frame in the world coordinate to compute the 3D position in the current frame, if
 *        the depth in current frame is negative, continue to process the next point
 *    2.2 use the 3D position in the camera coordinate to compute the coordinate in the current frame image, if the projection of the mappoint in current frame is
 *        out of the image bound, continue to process next mappoint
 *    2.3 get the octave of the keypoint in the last frame pyramid, compute the window radius according to the scale factor in current frame of octave in last 
 *        frame, later, we will search corresponding keypoints in current frame in the window 
 *    2.4 for different cases, search the corresponding keypoint in different levels in the pyramid
 *        2.4.1 if bForward is true, it means the camera is getting closer to the keypoints, the keypoints get bigger in the image, so we will
 *              search the corresponding keypoints in higher level of the pyramid
 *        2.4.2 if bBackward is true, it means the camera is getting farer to the keypoints, the keypoints get smaller in the image, so we will
 *              search the corresponding keypoints in lower level of the pyramid
 *        2.4.3 else, search the corresponding keypoints in the neighbor level
 *        we can get a vector of corresponding indexes in the above mentioned way, if we can not find any corresponding keypoints, skip this point and process 
 *        the next mappoint in last frame
 *    2.5 get the descriptor of the map point in the last frame, for all the candidates computed in 2.4
 *        2.5.1 get the map point corresponding to this keypoint, if the observations of this point is bigger than 0, continue to process next keypoint
 *        2.5.2 if this is a stereo case, use the depth computed above and the f*baseline to predict the u coordinate in the right image, compute the
 *              error between the predicted coordinate and the one in the muvRight vector, if it is bigger than the threshold, continue to compute 
 *              next candidate point
 *        2.5.3 compute the distance of descriptor between the last frame mappoint and the current frame candidate, get the best distance and the
 *              corresponding index
 *    2.6 if the best distance is less than the threshold, set the element whose index is the best distance index in the vector mvpMapPoint tobe
 *        the mappoint and add the nmatches by 1, put the index into the corresponding vector in rotHist
 * 3. if mbCheckOrientation is true
 *    3.1 compute the three index with biggest vector size in rotHist, if the second_biggest_size<0.1*biggest_size, set ind2 and ind3 to -1; if the 
 *        third_biggest_size<0.1*biggest_size, set ind3 tobe -1
 *    3.2 for every vector in the rotHist with index not belong to {ind1, ind2, ind3}, set the mappoint pointers in it tobe null
 *        pointers, and for every map point decrease nmatches by 1
 * 4. return the number of valid matches
 */
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
{
    int nmatches = 0;

    // Rotation Histogram (to check rotation consistency)检查旋转一致性
    // create 30 vector of int type
    vector<int> rotHist[HISTO_LENGTH];//构建具有HISTO_LENGTH栏的直方图，每栏中是一个int型的向量
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    /*
    Pc=Rcw*Pw+tcw    Pw=Rcw.inverse()*Pc-Rcw.inverse*tcw=Rcw.transpose()*Pc-Rcw.transpose()*tcw
    对于相机光心，Pc为0，Pw=-Rcw.transpose()*tcw
    */
    // compute the position of the current camera in the world coordinate
    const cv::Mat twc = -Rcw.t()*tcw;//当前帧光心在世界坐标系下的坐标

    const cv::Mat Rlw = LastFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tlw = LastFrame.mTcw.rowRange(0,3).col(3);
    /*Pl = Rlw*Pw + tlw   将当前帧光心在世界坐标系下的坐标转为当前帧光心在上一帧坐标系下的坐标*/
    // use the current position in the world coordinate and the pose of last frame in the world coordinate to compute the camera position in the last frame
    const cv::Mat tlc = Rlw*twc+tlw;

    // if the current camera pose in the last frame is bigger than the baseline of current frame and it's not the monocular case, set bForward tobe true
    // if the opposite of current camera pose in the last frame is bigger than the baseline of current frame and it's not the monocular case, set
    // bBackward tobe true
    const bool bForward = tlc.at<float>(2)>CurrentFrame.mb && !bMono;//若当前帧光心在上一帧相机坐标系下深度大于基线长度且非单目，则置bForward为真
    const bool bBackward = -tlc.at<float>(2)>CurrentFrame.mb && !bMono;//若负深度大于基线长度且非单目，则置bBackward为真

    // for all keypoints in the last frame
    for(int i=0; i<LastFrame.N; i++)//对于上一帧中所有关键点
    {
        // get the map points related with the keypoints in the last frame
        MapPoint* pMP = LastFrame.mvpMapPoints[i];//取出关键点对应的地图点

        // if the mappoint pointer is not none
        if(pMP)//若地图点非空
        {
            // if the map point is not marked as an outlier in the last frame
            if(!LastFrame.mvbOutlier[i])
            {
                // Project
                // get the 3D position in the world coordinate, use the pose of current frame in the world coordinate to compute the 3D position in the current frame
                cv::Mat x3Dw = pMP->GetWorldPos();//地图点世界坐标系下的坐标
                cv::Mat x3Dc = Rcw*x3Dw+tcw;//地图点在当前帧相机坐标系下的坐标

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);

                // if the depth is negative, skip this keypoint and continue to process the next one
                if(invzc<0)//若深度为负，则跳过这个上一帧中关键点继续处理下一个关键点
                    continue;

                // use the 3D position in the camera coordinate to compute the coordinate in the current frame image
                float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;//从相机坐标系变换到当前帧图像坐标系
                float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;

                // if the projection of the mappoint in current frame is out of the image bound, continue to process next mappoint
                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)//若投影不在图片幅面内，则跳过这个地图点处理下一个地图点
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

                // get the octave of the keypoint in the last frame pyramid
                int nLastOctave = LastFrame.mvKeys[i].octave;//取出这个关键点在上一帧中对应的金字塔层数

                // Search in a window. Size depends on scale
                // compute the window radius according to the scale factor in current frame of octave in last frame, later, we will search 
                // corresponding keypoints in current frame in the window 
                float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];//计算一个半径，在一定区域内查找关键点

                vector<size_t> vIndices2;
                //对于不同情况，在不同金字塔层中搜索关键点
                /* 
                1. if bForward is true, it means the camera is getting closer to the keypoints, the keypoints get bigger in the image, so we will
                   search the corresponding keypoints in higher level of the pyramid
                2. if bBackward is true, it means the camera is getting farer to the keypoints, the keypoints get smaller in the image, so we will
                   search the corresponding keypoints in lower level of the pyramid
                3. else, search the corresponding keypoints in the neighbor levels
                */
                if(bForward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave);
                else if(bBackward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, 0, nLastOctave);
                else
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave-1, nLastOctave+1);

                // if we can not find any corresponding keypoints, skip this point and process the next mappoint in last frame
                if(vIndices2.empty())//若当前帧投影点附近没有关键点，跳过这个上一帧关键点继续处理下一个上一帧关键点
                    continue;

                // get the descriptor of the map point in the last frame
                const cv::Mat dMP = pMP->GetDescriptor();//取出地图点的描述子

                int bestDist = 256;
                int bestIdx2 = -1;

                /*
                 * for all candidate keypoints in the vector
                 *    1. get the map point corresponding to this keypoint, if the observations of this point is bigger than 0, continue to process next keypoint
                 *    2. if this is a stereo case, use the depth computed above and the f*baseline to predict the u coordinate in the right image, compute the
                 *       error between the predicted coordinate and the one in the muvRight vector, if it is bigger than the threshold, continue to compute 
                 *       next candidate point
                 *    3. compute the distance of descriptor between the last frame mappoint and the current frame candidate, get the best distance and the
                 *       corresponding index
                 */
                for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)//对于投影点附近所有关键点
                {
                    const size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2])//取出关键点对应的地图点
                        if(CurrentFrame.mvpMapPoints[i2]->Observations()>0)//若该点被观测次数大于0，则跳过该点继续处理下一个投影点附近关键点
                            continue;

                    if(CurrentFrame.mvuRight[i2]>0)//若当前帧为双目
                    {
                        /*disparity=f*baseline/depth*/
                        const float ur = u - CurrentFrame.mbf*invzc;//利用视差求出投影点在右目中对应的横坐标
                        const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                        if(er>radius)//若上一帧投影点视差对应的右目横坐标与与当前帧关键点右目横坐标差别较大，则跳过这个关键点继续处理下一个投影点附近关键点
                            continue;
                    }

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);//取出关键点对应的描述子

                    const int dist = DescriptorDistance(dMP,d);//求出关键点描述子与地图点描述子的距离

                    if(dist<bestDist)//取出与地图点描述子距离最近的关键点
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }//结束对投影点附近关键点的遍历

                // if the best distance is less than the threshold, set the element whose index is the best distance index in the vector mvpMapPoint tobe
                // the mappoint and add the nmatches by 1, put the index into the corresponding vector in rotHist
                if(bestDist<=TH_HIGH)//若最佳关键点与地图点描述子距离小于阈值
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;//记当前帧中最佳关键点为上一帧中取出的地图点
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = LastFrame.mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);//将当前帧最佳关键点的下标给入到直方图对应的栏中
                    }
                }
            }
        }
    }//结束对上一帧所有关键点的遍历

    //Apply rotation consistency
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        /*
        compute the three index with biggest vector size in rotHist, if the second_biggest_size<0.1*biggest_size, set ind2 and ind3 to -1; if the 
        third_biggest_size<0.1*biggest_size, set ind3 tobe -1
        */
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        /*
        for every vector in the rotHist with index not belong to {ind1, ind2, ind3}, set the mappoint pointers in it tobe null
        pointers, and for every map point decrease nmatches by 1
        */
        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

//将关键帧中地图点投影到帧中寻找匹配
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist)
{
    int nmatches = 0;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    const cv::Mat Ow = -Rcw.t()*tcw;//当前帧相机光心在世界坐标系下的坐标

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];//定义一个有HISTO_LENGTH栏的直方图，直方图每一栏是一个int型向量，可以存储角度差在该区间的点下标
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)//对于关键帧中所有地图点
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP)//若存在该地图点
        {
            if(!pMP->isBad() && !sAlreadyFound.count(pMP))//若地图点不是坏点且不在已经找到的地图点集合中
            {
                //Project
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);

                const float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                const float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;//将地图点投影到当前帧图像坐标系中

                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)//若投影点不在当前帧幅面内，则跳过这个地图点继续处理下一个关键帧中地图点
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

                // Compute predicted scale level
                cv::Mat PO = x3Dw-Ow;//计算地图点到当前帧相机光心的距离
                float dist3D = cv::norm(PO);

                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();

                // Depth must be inside the scale pyramid of the image
                if(dist3D<minDistance || dist3D>maxDistance)//若距离不在一定范围内，则继续处理下一个地图点
                    continue;

                int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);//预测地图点在当前帧中的金字塔层数

                // Search in a window
                const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];//利用金字塔层数计算对应的半径

                const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);//在金字塔附近层投影点邻域取出关键点

                if(vIndices2.empty())//若投影点附近无关键点，则继续处理下一个地图点
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();//取出地图点的描述子

                int bestDist = 256;
                int bestIdx2 = -1;

                for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)//对于投影点附近的关键点
                {
                    const size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2])//若该关键点已有对应的地图点，则跳过这个关键点不再处理
                        continue;

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);//取出关键点对应的描述子

                    const int dist = DescriptorDistance(dMP,d);//计算关键点描述子与地图点描述子的距离

                    if(dist<bestDist)//统计最小的描述子距离
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=ORBdist)//若最佳距离小于阈值
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;//记当前帧中最佳距离对应下标的地图点为关键帧中取出的地图点
                    nmatches++;//匹配次数加1

                    if(mbCheckOrientation)//统计最佳下标在直方图的哪栏
                    {
                        float rot = pKF->mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }

            }
        }
    }//结束对关键帧中地图点的遍历

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

//五个参数为1.直方图（直方图每一栏为一个int型向量），2.直方图的栏数，3、4、5.三个输出的栏下标，代表向量中元素最多的三个栏
void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)//对于直方图中所有栏
    {
        const int s = histo[i].size();
        if(s>max1)//若新值大于最大值，则重置所有的三个值
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)//若新值大于次大值，则重置次大值与第三大值
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)//若新值大于第三大值，只重置第三大值
        {
            max3=s;
            ind3=i;
        }
    }//结束对直方图中栏的遍历

    if(max2<0.1f*(float)max1)//若次大值小于十分之一的最大值，则置次大值与第三大值为无效
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)//若第三大值小于十分之一最大值，则置第三大值为无效
    {
        ind3=-1;
    }
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)//循环8次
    {//每次处理8个16进制，即32位二进制  ^位异或运算符，若两个运算对象对应位置有且只有一个为1则结果中该位为1否则为0
        unsigned  int v = *pa ^ *pb;//做过异或运算后，其实要做的就是统计v中1的个数
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}
/*
对于v = v - ((v >> 1) & 0x55555555);
记v为                                    a1  a2  a3  a4    a5  a6  a7  a8    a9  a10 a11 a12   a13 a14 a15 a16   a17 a18 a19 a20   a21 a22 a23 a24   a25 a26 a27 a28   a29 a30 a31 a32
则对v右移又与上8位的16进制5,即8位的0101为   0  a1   0  a3     0  a5   0  a7     0   a9  0  a11    0  a13  0  a15    0  a17  0  a19    0  a21  0  a23    0  a25  0  a27    0  a29  0  a31
对于相邻的两位 a1a2-0a1的结果为   
1）若a1=1,a2=1,则0a1为01,11-01为10即代表a1a2有两个1
2）若a1=1,a2=0,则0a1为01,10-01为01即代表a1a2有一个1
3）若a1=0,a2=1,则0a1为00,01-00为01即代表a1a2有一个1
4）若a1=0,a2=0,则0a1为00,00-00为00即代表a1a2没有1
这样进行减法后，相邻两位的数值即为相邻两位中1的个数
a1  a2  a3  a4    a5  a6  a7  a8    a9  a10 a11 a12   a13 a14 a15 a16   a17 a18 a19 a20   a21 a22 a23 a24   a25 a26 a27 a28   a29 a30 a31 a32
 0  a1   0  a3     0  a5   0  a7     0   a9  0  a11    0  a13  0  a15    0  a17  0  a19    0  a21  0  a23    0  a25  0  a27    0  a29  0  a31
 \  /    \  /      \  /    \  /      \   /   \   /     \   /   \   /     \   /   \   /     \   /   \   /     \   /   \   /     \   /   \   /
a1a2    a3a4       a5a6    a7a8       a9a10   a11a12    .....
1的个数 1的个数    1的个数  1的个数     1的个数  1的个数   .....
进行上面这步以后，二进制串中的值已经从描述子不同的位数变成了相邻位中1的个数
记suma_b表示a_b两位中1的个数，则最新的v变为
sum1_2  sum3_4  sum5_6  sum7_8  sum9_10  sum11_12  sum13_14  sum15_16  sum17_18  sum19_20  sum21_22  sum23_24  sum25_26  sum27_28  summ29_30  sum31_32
v = (v & 0x33333333) + ((v >> 2) & 0x33333333)分析如下
v按位与8个16进制3即0011为
  0  0  sum3_4    0  0  sum7_8    0  0   sum11_12     0   0  sum15_16    0   0   sum19_20    0   0   sum23_24    0   0   sum27_28     0   0   sum31_32  
v右移两位再按位与8个16进制3即0011为
  0  0  sum1_2    0  0  sum5_6    0  0   sum9_10      0   0  sum13_14    0   0   sum17_18    0   0   sum21_22    0   0   sum25_26     0   0   sum29_30
二者再相加以后，v的值已经变成
sum1_2_3_4   sum5_6_7_8   sum9_10_11_12   sum13_14_15_16   sum17_18_19_20   sum21_22_23_24   sum25_26_27_28   sum29_30_31_32
8组二进制中最大的值为4 ，两组相加不超过8，还是可以存在4位二进制数中不用进位
(((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24分析如下
v右移四位为
0  0  0  0    sum1_2_3_4   sum5_6_7_8   sum9_10_11_12   sum13_14_15_16   sum17_18_19_20   sum21_22_23_24   sum25_26_27_28  
v加上v右移四位为
sum1_2_3_4   sum1_2_3_4_5_6_7_8   sum5_6_7_8_9_10_11_12   sum9_10_11_12_13_14_15_16   sum13_14_15_16_17_18_19_20   sum17_18_19_20_21_22_23_24   sum21_22_23_24_25_26_27_28   sum25_26_27_28_29_30_31_32
这个值再与上16进制F0F0F0F即为   0000 1111 0000 1111 0000 1111 0000 1111为
0000   sum1_2_3_4_5_6_7_8   0000   sum9_10_11_12_13_14_15_16   0000   sum17_18_19_20_21_22_23_24   0000   sum25_26_27_28_29_30_31_32
再乘以1010101即为
0000             sum1_2_3_4_5_6_7_8            0000          sum9_10_11_12_13_14_15_16       0000        sum17_18_19_20_21_22_23_24      0000         sum25_26_27_28_29_30_31_32
                                               0000             sum1_2_3_4_5_6_7_8            0000          sum9_10_11_12_13_14_15_16       0000        sum17_18_19_20_21_22_23_24      0000         sum25_26_27_28_29_30_31_32
                                                                                              0000             sum1_2_3_4_5_6_7_8            0000          sum9_10_11_12_13_14_15_16       0000        sum17_18_19_20_21_22_23_24      0000         sum25_26_27_28_29_30_31_32
                                                                                                                                            0000             sum1_2_3_4_5_6_7_8            0000          sum9_10_11_12_13_14_15_16       0000        sum17_18_19_20_21_22_23_24      0000         sum25_26_27_28_29_30_31_32
dist为32位2进制，1位16进制为4位2进制，32位2进制代表8位16进制,故而上面结果中前6位16进制溢出了,剩下的结果为
 0000         sum25_26_27_28_29_30_31_32
    0000        sum17_18_19_20_21_22_23_24      0000         sum25_26_27_28_29_30_31_32
     0000          sum9_10_11_12_13_14_15_16       0000        sum17_18_19_20_21_22_23_24      0000         sum25_26_27_28_29_30_31_32
    0000             sum1_2_3_4_5_6_7_8            0000          sum9_10_11_12_13_14_15_16       0000        sum17_18_19_20_21_22_23_24      0000         sum25_26_27_28_29_30_31_32
此时最大可能值为32,需要6位二进制两位16进制表示，故而下列结果将两位16进制写在一起，加法结果为
sum1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32   sum9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32   sum17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32   sum25_26_27_28_29_30_31_32
向右移24位可得sum1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31_32 即为二进制串中1的个数
*/

} //namespace ORB_SLAM
