/**
* This file is part of ORB-SLAM2.
* This file is a modified version of EPnP <http://cvlab.epfl.ch/EPnP/index.php>, see FreeBSD license below.
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

/**
* Copyright (c) 2009, V. Lepetit, EPFL
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
*    list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The views and conclusions contained in the software and documentation are those
* of the authors and should not be interpreted as representing official policies,
*   either expressed or implied, of the FreeBSD Project
*/

/*
一个算子A的零空间是方程Av=0的所有解v的集合，也叫作A的核，核空间
*/

/*
PNP问题：已知空间中存在地图点，世界坐标系下的坐标为Pw1，Pw2,...,Pwn以及对应于在该帧中的匹配点像素坐标u1,u2,u3,...,un，求解此时相机的位姿Tcw
epnp: 将n个三维点表示为4个虚拟控制点的加权和，将问题简化为估计控制点在相机参考系下的坐标，通过将坐标表示为12*12矩阵特征向量的加权和，然后解一定量的二次方程来得到正确权值，可以
在O(n)时间内解决问题。
大部分非迭代方法首先通过求解点深度估计相机坐标系下的点位置，然后可以通过关联点在不同坐标系下的位姿求取相机位姿
*/



#include <iostream>

#include "PnPsolver.h"

#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>
#include "Thirdparty/DBoW2/DUtils/Random.h"
#include <algorithm>

using namespace std;

namespace ORB_SLAM2
{

/*
从F中取出关键点，及其对应的金字塔层数，从vpMapPointMatches中取出地图点
*/
PnPsolver::PnPsolver(const Frame &F, const vector<MapPoint*> &vpMapPointMatches):
    pws(0), us(0), alphas(0), pcs(0), maximum_number_of_correspondences(0), number_of_correspondences(0), mnInliersi(0),
    mnIterations(0), mnBestInliers(0), N(0)
{
    mvpMapPointMatches = vpMapPointMatches;
    mvP2D.reserve(F.mvpMapPoints.size());//std::vector<MapPoint*> mvpMapPoints;向量存储帧中与关键点关联的地图点，若关键点无关联地图点，则设置为空指针
    mvSigma2.reserve(F.mvpMapPoints.size());
    mvP3Dw.reserve(F.mvpMapPoints.size());
    mvKeyPointIndices.reserve(F.mvpMapPoints.size());
    mvAllIndices.reserve(F.mvpMapPoints.size());

    int idx=0;
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)//对于vpMapPointMatches中的所有地图点
    {
        MapPoint* pMP = vpMapPointMatches[i];
        //若存在地图点且地图点不是坏点，则将对应像素二维点与世界坐标系三维点及相关参数取出
        if(pMP)
        {
            if(!pMP->isBad())
            {
                const cv::KeyPoint &kp = F.mvKeysUn[i];//从图片帧中取出关键点

                mvP2D.push_back(kp.pt);//将关键点放入容器
                mvSigma2.push_back(F.mvLevelSigma2[kp.octave]);//将关键点所在金字塔层数的参数放入容器

                cv::Mat Pos = pMP->GetWorldPos();
                mvP3Dw.push_back(cv::Point3f(Pos.at<float>(0),Pos.at<float>(1), Pos.at<float>(2)));//将地图点在世界坐标系下坐标放入容器

                mvKeyPointIndices.push_back(i);//将地图点下标放入容器
                mvAllIndices.push_back(idx);//将计数值放入容器，存的只是简单的计数值，1、2、3...依次上升，并不是关键点在关键帧中的原始下标               

                idx++;//计数值加1
            }
        }
    }//结束对于vpMapPointMatches中的所有地图点的遍历

    // Set camera calibration parameters
    fu = F.fx;
    fv = F.fy;
    uc = F.cx;
    vc = F.cy;

    SetRansacParameters();
}

PnPsolver::~PnPsolver()
{
  delete [] pws;
  delete [] us;
  delete [] alphas;
  delete [] pcs;
}

/*
double probability = 0.99, int minInliers = 8 , int maxIterations = 300, int minSet = 4, float epsilon = 0.4, float th2 = 5.991
*/
/*
设置ransac的一些参数
求sim3的问题中，单次迭代要取3对对应点
设迭代次数为iter,局内点数目至少为inlier,epsilon为(局内点数inlier/总样本数N)，希望iter次迭代中至少有一次取到的三对点均为局内点的的概率大于P
求解迭代次数步骤如下：
单次迭代取到三对局内点概率约为 pow(epsilon, 3)
单次迭代取不到三对局内点概率为 1-pow(epsilon, 3)
iter次迭代每次都取不到三对局内点概率为 pow(1-pow(epsilon, 3), iter)
iter次迭代中至少有一次取到了三对局内点概率为 1-pow(1-pow(epsilon, 3), iter)
让这个概率大于希望的概率P
1-pow(1-pow(epsilon, 3), iter) > P
则 pow(1-pow(epsilon, 3), iter) < 1-P
两边同时取对数 iter*log(1-pow(epsilon, 3)) < log(1-P)
由于1-pow(epsilon, 3)小于1，故而log(1-pow(epsilon, 3))<0
两边同时除log(1-pow(epsilon, 3))需要变符号， 可得 iter > log(1-P)/log(1-pow(epsilon, 3))
*/
void PnPsolver::SetRansacParameters(double probability, int minInliers, int maxIterations, int minSet, float epsilon, float th2)
{
    mRansacProb = probability;
    mRansacMinInliers = minInliers;
    mRansacMaxIts = maxIterations;
    mRansacEpsilon = epsilon;
    mRansacMinSet = minSet;

    N = mvP2D.size(); // number of correspondences 对应二维三维点的个数即ransac样本数

    mvbInliersi.resize(N);

    // Adjust Parameters according to number of correspondences
    int nMinInliers = N*mRansacEpsilon;//利用总样本数乘以局内点占样本数比例求局内点个数
    if(nMinInliers<mRansacMinInliers)//若按比例求出的局内点个数小于预先设定的最少局内点个数，则保持个数为预先设定最少局内点个数不变
        nMinInliers=mRansacMinInliers;
    if(nMinInliers<minSet)//若按比例求出的局内点个数小于ransac单次采样求rt的点数，则最少局内点数为单次采样点数
        nMinInliers=minSet;
    mRansacMinInliers = nMinInliers;

    if(mRansacEpsilon<(float)mRansacMinInliers/N)
        mRansacEpsilon=(float)mRansacMinInliers/N;

    // Set RANSAC iterations according to probability, epsilon, and max iterations
    int nIterations;

    if(mRansacMinInliers==N)//如果设置的最小局内点数即为总样本数，则迭代一次即可
        nIterations=1;
    else
        nIterations = ceil(log(1-mRansacProb)/log(1-pow(mRansacEpsilon,3)));

    mRansacMaxIts = max(1,min(nIterations,mRansacMaxIts));//迭代次数大于1小于mRansacMaxIts

    mvMaxError.resize(mvSigma2.size());
    for(size_t i=0; i<mvSigma2.size(); i++)
        mvMaxError[i] = mvSigma2[i]*th2;//利用金字塔层数对应的参数求最大误差阈值
}

cv::Mat PnPsolver::find(vector<bool> &vbInliers, int &nInliers)
{
    bool bFlag;
    return iterate(mRansacMaxIts,bFlag,vbInliers,nInliers);    
}

cv::Mat PnPsolver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
{
    bNoMore = false;
    vbInliers.clear();
    nInliers=0;

    set_maximum_number_of_correspondences(mRansacMinSet);

    if(N<mRansacMinInliers)//若对应点对数小于ransac的最小局内点数，则返回空矩阵
    {
        bNoMore = true;
        return cv::Mat();
    }

    vector<size_t> vAvailableIndices;

    int nCurrentIterations = 0;
    while(mnIterations<mRansacMaxIts || nCurrentIterations<nIterations)
    {
        nCurrentIterations++;
        mnIterations++;
        reset_correspondences();//将三维点与二维点对应次数置0

        vAvailableIndices = mvAllIndices;//两个存储整型的向量

        // Get min set of points 取一个最小的点集
        for(short i = 0; i < mRansacMinSet; ++i)//一次迭代内，需要mRansacMinSet对对应来求取位姿
        {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);//取出一个随机下标

            int idx = vAvailableIndices[randi];

            add_correspondence(mvP3Dw[idx].x,mvP3Dw[idx].y,mvP3Dw[idx].z,mvP2D[idx].x,mvP2D[idx].y);//增加一对三维点与二维点的对应  将点加入pws与us中

            vAvailableIndices[randi] = vAvailableIndices.back();//将向量中最后一个元素赋给刚刚取出的元素，然后将最后一个元素丢弃，相当于将刚才取出的那个元素丢弃掉
            vAvailableIndices.pop_back();
        }

        // Compute camera pose  double PnPsolver::compute_pose(double R[3][3], double t[3])
        // 根据取出的对应点采用epnp技术计算Rt
        compute_pose(mRi, mti);

        // Check inliers
        /*
        利用求出的Rt关系求世界坐标系下三维点向图像投影的像素坐标，计算投影点与原始像素坐标欧式距离的平方，若距离平方在一定范围内，则记该点为局内点的标记mvbInliersi为真，且
        将局内点数加1，否则记局内点标记为假
        */
        CheckInliers();
        
        /*
        若局内点数大于ransac要求的最小局内点数
        */
        if(mnInliersi>=mRansacMinInliers)
        {
            // If it is the best solution so far, save it
            /*
            若这组局内点数是迄今为止最好的局内点数，则将这组Rt放入4*4矩阵中
            */
            if(mnInliersi>mnBestInliers)
            {
                mvbBestInliers = mvbInliersi;
                mnBestInliers = mnInliersi;

                cv::Mat Rcw(3,3,CV_64F,mRi);
                cv::Mat tcw(3,1,CV_64F,mti);
                Rcw.convertTo(Rcw,CV_32F);
                tcw.convertTo(tcw,CV_32F);
                mBestTcw = cv::Mat::eye(4,4,CV_32F);
                Rcw.copyTo(mBestTcw.rowRange(0,3).colRange(0,3));
                tcw.copyTo(mBestTcw.rowRange(0,3).col(3));
            }
            /*用刚才的所有局内点求rt并统计新的局内点数，若局内点数大于要求的最小局内点数则Refine()返回真，否则Refine()返回假*/
            if(Refine())
            {
                nInliers = mnRefinedInliers;
                vbInliers = vector<bool>(mvpMapPointMatches.size(),false);
                for(int i=0; i<N; i++)//对于所有的对应点，若其在refine算法中为局内点，则标记其为局内点
                {
                    if(mvbRefinedInliers[i])
                        vbInliers[mvKeyPointIndices[i]] = true;
                }
                //返回refine算法计算的变换矩阵
                return mRefinedTcw.clone();
            }

        }
    }//结束要求的最大迭代次数
    //若没有从上个循环中返回，即迭代完了足够次数没有触发返回条件，则取出刚才最佳的那次结果,若局内点大于阈值则返回那组结果的局内点标记及转换矩阵
    if(mnIterations>=mRansacMaxIts)
    {
        bNoMore=true;
        if(mnBestInliers>=mRansacMinInliers)
        {
            nInliers=mnBestInliers;
            vbInliers = vector<bool>(mvpMapPointMatches.size(),false);
            for(int i=0; i<N; i++)
            {
                if(mvbBestInliers[i])
                    vbInliers[mvKeyPointIndices[i]] = true;
            }
            return mBestTcw.clone();
        }
    }
    //若上述退出条件均不满足，则返回空矩阵
    return cv::Mat();
}

bool PnPsolver::Refine()
{
    /*
    取出刚才计算出的局内点
    */
    vector<int> vIndices;
    vIndices.reserve(mvbBestInliers.size());

    for(size_t i=0; i<mvbBestInliers.size(); i++)
    {
        if(mvbBestInliers[i])
        {
            vIndices.push_back(i);
        }
    }
    /*为相关量（点坐标、alpha等）的指针申请空间*/
    set_maximum_number_of_correspondences(vIndices.size());
    /*将对应点数重置为0*/
    reset_correspondences();

    /*将所有局内点加入对应来求取rt*/
    for(size_t i=0; i<vIndices.size(); i++)
    {
        int idx = vIndices[i];
        add_correspondence(mvP3Dw[idx].x,mvP3Dw[idx].y,mvP3Dw[idx].z,mvP2D[idx].x,mvP2D[idx].y);
    }

    // Compute camera pose
    compute_pose(mRi, mti);

    // Check inliers 统计局内点个数并对局内点进行标记
    CheckInliers();

    mnRefinedInliers =mnInliersi;
    mvbRefinedInliers = mvbInliersi;
    
    /*若局内点数大于要求的最小局内点数，则将Rt放入4*4矩阵中并返回真，否则返回假*/
    if(mnInliersi>mRansacMinInliers)
    {
        cv::Mat Rcw(3,3,CV_64F,mRi);
        cv::Mat tcw(3,1,CV_64F,mti);
        Rcw.convertTo(Rcw,CV_32F);
        tcw.convertTo(tcw,CV_32F);
        mRefinedTcw = cv::Mat::eye(4,4,CV_32F);
        Rcw.copyTo(mRefinedTcw.rowRange(0,3).colRange(0,3));
        tcw.copyTo(mRefinedTcw.rowRange(0,3).col(3));
        return true;
    }

    return false;
}

/*
利用求出的Rt关系求世界坐标系下三维点向图像投影的像素坐标，计算投影点与原始像素坐标欧式距离的平方，若距离平方在一定范围内，则记该点为局内点的标记mvbInliersi为真，且
将局内点数加1，否则记局内点标记为假
*/
void PnPsolver::CheckInliers()
{
    mnInliersi=0;

    for(int i=0; i<N; i++)
    {
        cv::Point3f P3Dw = mvP3Dw[i];
        cv::Point2f P2D = mvP2D[i];

        float Xc = mRi[0][0]*P3Dw.x+mRi[0][1]*P3Dw.y+mRi[0][2]*P3Dw.z+mti[0];
        float Yc = mRi[1][0]*P3Dw.x+mRi[1][1]*P3Dw.y+mRi[1][2]*P3Dw.z+mti[1];
        float invZc = 1/(mRi[2][0]*P3Dw.x+mRi[2][1]*P3Dw.y+mRi[2][2]*P3Dw.z+mti[2]);

        double ue = uc + fu * Xc * invZc;
        double ve = vc + fv * Yc * invZc;

        float distX = P2D.x-ue;
        float distY = P2D.y-ve;

        float error2 = distX*distX+distY*distY;

        if(error2<mvMaxError[i])
        {
            mvbInliersi[i]=true;
            mnInliersi++;
        }
        else
        {
            mvbInliersi[i]=false;
        }
    }
}

/*
为指针申请空间,若maximum_number_of_correspondences小于函数输入n，则将maximum_number_of_correspondences设置为n
pws中存储的是世界坐标系下三维坐标，每对点包含3个double, us中存储的是像素坐标，每对点包含两个double,
alphas存储控制点的权重，每对点包含4个double,pcs每对点包含3个double
*/
void PnPsolver::set_maximum_number_of_correspondences(int n)
{
  if (maximum_number_of_correspondences < n) {
    if (pws != 0) delete [] pws;
    if (us != 0) delete [] us;
    if (alphas != 0) delete [] alphas;
    if (pcs != 0) delete [] pcs;

    maximum_number_of_correspondences = n;
    pws = new double[3 * maximum_number_of_correspondences];
    us = new double[2 * maximum_number_of_correspondences];
    alphas = new double[4 * maximum_number_of_correspondences];
    pcs = new double[3 * maximum_number_of_correspondences];
  }
}

/*
将对应点数即number_of_correspondences重置为0
*/
void PnPsolver::reset_correspondences(void)
{
  number_of_correspondences = 0;
}

/*
五个参数均为输入，分别为三维坐标与像素坐标，将点坐标加入pws与us中
*/
void PnPsolver::add_correspondence(double X, double Y, double Z, double u, double v)
{
  pws[3 * number_of_correspondences    ] = X;
  pws[3 * number_of_correspondences + 1] = Y;
  pws[3 * number_of_correspondences + 2] = Z;

  us[2 * number_of_correspondences    ] = u;
  us[2 * number_of_correspondences + 1] = v;

  number_of_correspondences++;
}

//求取所有点的平均坐标作为第一个参考点，分别将平均坐标加上协方差矩阵的三个特征向量作为其余三个特征点
void PnPsolver::choose_control_points(void)
{
  // Take C0 as the reference points centroid:
  cws[0][0] = cws[0][1] = cws[0][2] = 0;
  for(int i = 0; i < number_of_correspondences; i++)//将三维点与二维点的对应点对中所有三维点坐标加在一起
    for(int j = 0; j < 3; j++)
      cws[0][j] += pws[3 * i + j];

  for(int j = 0; j < 3; j++)
    cws[0][j] /= number_of_correspondences;//求所有三维点的坐标平均值


  // Take C1, C2, and C3 from PCA on the reference points:
  CvMat * PW0 = cvCreateMat(number_of_correspondences, 3, CV_64F);//存储每个点坐标减去点集平均坐标

  double pw0tpw0[3 * 3], dc[3], uct[3 * 3];
  CvMat PW0tPW0 = cvMat(3, 3, CV_64F, pw0tpw0);
  CvMat DC      = cvMat(3, 1, CV_64F, dc);
  CvMat UCt     = cvMat(3, 3, CV_64F, uct);

  for(int i = 0; i < number_of_correspondences; i++)
    for(int j = 0; j < 3; j++)
      PW0->data.db[3 * i + j] = pws[3 * i + j] - cws[0][j];//将每个点坐标减去平均点坐标存入向量中

  cvMulTransposed(PW0, &PW0tPW0, 1);//求点坐标减均值点坐标矩阵与其转置的积，即协方差阵
  cvSVD(&PW0tPW0, &DC, &UCt, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);//求协方差阵的分解
  //void cvSVD(CvArr* A, CvArr* W, CvArr* U=NULL, CvArr* V=NULL, int flags=0)

  cvReleaseMat(&PW0);//释放存储着相对点坐标的内存
  //cws[0]中存储的是三维点的平均坐标
  for(int i = 1; i < 4; i++) {
    double k = sqrt(dc[i - 1] / number_of_correspondences);
    for(int j = 0; j < 3; j++)
      cws[i][j] = cws[0][j] + k * uct[3 * (i - 1) + j];//将特征向量方向作为坐标轴方向，特征值大小作为沿轴方向的尺度，确定除中心点外其他三个参考点
  }
}

//利用当前点与控制点坐标求取控制点权重即4个alpha
void PnPsolver::compute_barycentric_coordinates(void)//barycentric 质心
{
  double cc[3 * 3], cc_inv[3 * 3];
  CvMat CC     = cvMat(3, 3, CV_64F, cc);
  CvMat CC_inv = cvMat(3, 3, CV_64F, cc_inv);

/*
  cws[0]中存储的是三维点的平均坐标
  for(int i = 1; i < 4; i++) {
    double k = sqrt(dc[i - 1] / number_of_correspondences);
    for(int j = 0; j < 3; j++)
      cws[i][j] = cws[0][j] + k * uct[3 * (i - 1) + j];
  }
*/
/*
点集中的每个点可用四个参考点加权表示 p = a0*c0+a1*c1+a2*c2+a3*c3  其中 a0+a1+a2+a3=1
pix = a0*c0x + a1*c1x + a2*c2x + a3*c3x   (1)
piy = a0*c0y + a1*c1y + a2*c2y + a3*c3y   (2)
piz = a0*c0z + a1*c1z + a2*c2z + a3*c3z   (3)
这里有三个方程四个未知数，欠定，所以可以引入另一个约束a0+a1+a2+a3=1，并不是说存在神奇的控制点权重和为1这个定理
(1)式两边同时减去c0x, (2) 式两端同时减去c0y, (3)式两端同时减去c0z可得
pix-c0x = a0*c0x + a1*c1x + a2*c2x + a3*c3x - c0x  (4)
piy-c0y = a0*c0y + a1*c1y + a2*c2y + a3*c3y - c0y  (5)
piz-c0z = a0*c0z + a1*c1z + a2*c2z + a3*c3z - c0z  (6)
即
pix-c0x = a0*c0x + a1*c1x + a2*c2x + a3*c3x - (a0+a1+a2+a3)*c0x  (7)
piy-c0y = a0*c0y + a1*c1y + a2*c2y + a3*c3y - (a0+a1+a2+a3)*c0y  (8)
piz-c0z = a0*c0z + a1*c1z + a2*c2z + a3*c3z - (a0+a1+a2+a3)*c0z  (9)
即
pix-c0x = a1*(c1x - c0x) + a2*(c2x - c0x) + a3*(c3x - c0x)    (10)
piy-c0y = a1*(c1y - c0y) + a2*(c2y - c0y) + a3*(c3y - c0y)    (11)
piz-c0z = a1*(c1z - c0z) + a2*(c2z - c0z) + a3*(c3z - c0z)    (12)
写作矩阵形式为
| pix-c0x |   | c1x - c0x  c2x - c0x  c3x - c0x | | a1 |
| piy-c0y | = | c1y - c0y  c2y - c0y  c3y - c0y |*| a2 |
| piz-c0z |   | c1z - c0z  c2z - c0z  c3z - c0z | | a3 |
p_minus_c = deltac * a
a = deltac.inverse()*p_minus_c
*/
  for(int i = 0; i < 3; i++)
    for(int j = 1; j < 4; j++)
      cc[3 * i + j - 1] = cws[j][i] - cws[0][i];//相当于又把刚才的特征向量取出来，取出被特征值加权过的特征向量，即参考点相对参考点中心点的坐标

  cvInvert(&CC, &CC_inv, CV_SVD);//cvInvert求原矩阵的逆矩阵
  double * ci = cc_inv;
  for(int i = 0; i < number_of_correspondences; i++) {//对于所有的对应点
    double * pi = pws + 3 * i;//将指针指向下一个三维点坐标的起始
    double * a = alphas + 4 * i;

    for(int j = 0; j < 3; j++)    //pi[0] - cws[0][0]为点坐标减去坐标均值
      a[1 + j] =
	ci[3 * j    ] * (pi[0] - cws[0][0]) +
	ci[3 * j + 1] * (pi[1] - cws[0][1]) +
	ci[3 * j + 2] * (pi[2] - cws[0][2]);
    a[0] = 1.0f - a[1] - a[2] - a[3];
  }
}


/*
根据相机投影方程有
   | u |   | fu 0  uc|   | X |
wi*| v | = | 0  fv vc| * | Y |
   | 1 |   | 0  0  1 |   | Z |
wi*u = fu*X + uc*Z  (1)
wi*v = fv*Y + vc*Z  (2) 
wi   = Z            (3)
将(3)中表示的wi带入（1）（2）可得
fu*X+(uc-u)*Z = 0   (4)
fv*Y+(vc-v)*Z = 0   (5)
其中
X = a0*c0x + a1*c1x + a2*c2x + a3*c3x   (6)  
Y = a0*c0y + a1*c1y + a2*c2y + a3*c3y   (7)
Z = a0*c0z + a1*c1z + a2*c2z + a3*c3z   (8) 
由此可得（4）可以表示为向量相乘形式
| a0*fu 0 a0*(uc-u) a1*fu 0 a1*(uc-u) a2*fu 0 a2*(uc-u) a3*fu 0 a3*(uc-u) | * | c0x c0y c0z c1x c1y c1z c2x c2y c2z c3x c3y c3z |.transpose() = 0    (9)
将（5）表示为向量相乘形式为 
| 0 a0*fv a0*(uc-u) 0 a1*fv a1*(uc-u) 0 a2*fv a2*(uc-u) 0 a3*fv a3*(uc-u) | * | c0x c0y c0z c1x c1y c1z c2x c2y c2z c3x c3y c3z |.transpose() = 0    (10)
即为下列函数中的 M1与M2
*/
/*
u为像素横坐标，v为像素纵坐标，as为参考点的权重，row为对应点对的序号，每对点可以产生两个方程
函数作用为利用对应点的像素坐标与其三维坐标求出的相对参考点的权重来计算参考点约束关系的系数
上述系数由 1.相机内参 2.参考点权重 3.对应点像素坐标决定
*/
void PnPsolver::fill_M(CvMat * M,
		  const int row, const double * as, const double u, const double v)
{
  double * M1 = M->data.db + row * 12;
  double * M2 = M1 + 12;

  for(int i = 0; i < 4; i++) {
    M1[3 * i    ] = as[i] * fu;
    M1[3 * i + 1] = 0.0;
    M1[3 * i + 2] = as[i] * (uc - u);

    M2[3 * i    ] = 0.0;
    M2[3 * i + 1] = as[i] * fv;
    M2[3 * i + 2] = as[i] * (vc - v);
  }
}

void PnPsolver::compute_ccs(const double * betas, const double * ut)
{
  for(int i = 0; i < 4; i++)
    ccs[i][0] = ccs[i][1] = ccs[i][2] = 0.0f;

  for(int i = 0; i < 4; i++) {
    const double * v = ut + 12 * (11 - i);
    for(int j = 0; j < 4; j++)
      for(int k = 0; k < 3; k++)
	ccs[j][k] += betas[i] * v[3 * j + k];
  }
}

void PnPsolver::compute_pcs(void)
{
  /*
  对于number_of_correspondences对对应点中的第i对对应点，其坐标可以表示为四个控制点坐标的加权和，alpha1、alpha2、alpha3、alpha4可以由对应点在世界坐标系下三维坐标以及
  控制点在世界坐标系下三维坐标求取，这里根据alpha以及刚才求出的四个控制点在相机坐标系下坐标求对应点在相机坐标系下三维坐标
  */
  for(int i = 0; i < number_of_correspondences; i++) {
    double * a = alphas + 4 * i;
    double * pc = pcs + 3 * i;

    for(int j = 0; j < 3; j++)
      pc[j] = a[0] * ccs[0][j] + a[1] * ccs[1][j] + a[2] * ccs[2][j] + a[3] * ccs[3][j];
  }
}

double PnPsolver::compute_pose(double R[3][3], double t[3])
{
  choose_control_points();//计算4个控制点的坐标
  compute_barycentric_coordinates();//计算三维点对控制点的权重

  //每一对对应点可产生2个方程，待求取的向量包含4个控制点三维坐标即12个参数，所以参数矩阵M包含 2*对应点行，12列
  CvMat * M = cvCreateMat(2 * number_of_correspondences, 12, CV_64F);

  for(int i = 0; i < number_of_correspondences; i++)
    fill_M(M, 2 * i, alphas + 4 * i, us[2 * i], us[2 * i + 1]);

  double mtm[12 * 12], d[12], ut[12 * 12];
  CvMat MtM = cvMat(12, 12, CV_64F, mtm);
  CvMat D   = cvMat(12,  1, CV_64F, d);
  CvMat Ut  = cvMat(12, 12, CV_64F, ut);

  /*
  M矩阵大小为 (2*number_of_correspondences)*12   MtM大小应为 12*(2*number_of_correspondences)*(2*number_of_correspondences)*12即12*12
  Mx=0 其中x为四个参考点的坐标，可以被表示为
  sumof_i_from_1_to_N(beta(i)*v(i)),其中v(i)为对应于M的N个零奇异值的M的右奇异向量，可以通过求MtM的零特征向量的方式有效求取（MtM矩阵较小，只有12*12）
  计算MtM时间时间复杂度为O(n)
  解可以视为MtM矩阵零特征向量的加权和
  对于较小焦距，MtM只有一个零特征值，随着焦距增加会有4个零特征值
  MtM的零空间会从1到4变化，去计算到底有几个零特征值在特征值大体相同时可能会产生错误，计算所有的4个特征值，然后选一个使重投影误差最小的
  res = sum_of_i(square_dist(A*(R|t)*|pw(i)|,u(i))   suqare_dist(m_estimate,n)为两点二维距离的平方
                                     |  1  |
  以下按照零特征值的个数对问题分类
  case 1：  N = 1
  则x形式为x=beta*v,控制点在相机坐标系下的距离等于控制点在世界坐标系下的距离，可以利用这点求beta的值
  将控制点cc(i)对应的v中子向量记为v(i),有
  ||beta*v(i) - beta*v(j)||.square() = ||cw(i)-cw(j)||.square()    (1)
  则可以计算beta为
  beta = sumof_i_j_both_from_1_to_4(||v(i)-v(j)||*||cw(i)-cw(j)||)/sumof_i_j_both_from_1_to_4(||v(i)-v(j)||.square())    (2)
  case 2:  N = 2
  此时 x=beta1*v1+beta2*v2  ,此时距离约束为
  ||（beta1*v1(i) + beta2*v2(i)）- (beta1*v1(j) + beta2*v2(j))||.square() = ||cw(i)-cw(j)||.square()               (3)
  可通过解一个关于[beta11,beta12, beta22].transpose()的线性系统来求解，其中beta11=beta1*beta1, beta12=beta1*beta2, beta22=beta2*beta2
  由于有4对点产生一个有c42=(4*3)/(2*1)=6个方程关于betamn的线性系统，记为
  L*beta=raw
  L为一个由v1与v2构成的6*3矩阵，raw是一个存储距离平方||cw(i)-cw(j)||.square()的6维向量，beta=[beta11, beta12, beta22].transpose()为未知向量
  利用L的广义逆进行求解，且选择使pc(i)有正深度的betam
  可以再利用方程（2）进一步调节beta值，即令 cc(i)=beta*(beta1*v1(i)+beta2*v2(i))                   (4)
  case 3:  N = 3
  类似N=2的情况，用式（3）表示的六个约束，产生一个维度更大的线性系统 L*beta=raw
  L是一个由v1 v2 v3构成的6*6矩阵，beta是一个六维向量[beta11,beta12,beta13,beta22,beta23,beta33].transpose()
  L是方阵故而可以用L的逆矩阵而不是广义逆解这个线性系统
  case 4:  N = 4
  当前有4个未知的betam,理论上，使用的6个距离约束仍然足够
  然而线性化过程产生10个积 betamn=betam*betan作为未知量，故而此时不再有足够的约束，利用重线性化的技术解决这个问题，理论基础和决定控制点坐标的理论相同
  引入新的二次方程并且重新利用线性化手段求解的方案称为称为重线性化
  利用乘法的交换律  betaab*betacd=betaa*betab*betac*betad=betaa'b'*betac'd'
  其中{a',b',c',d'}是{a,b,c,d}的任意排序


  平面情况
  在平面情况下，即参考点的moment matrix有一个很小的特征值，仅需三个控制点，M的维度2n*9,其特征向量v(i)为9维，距离二次约束的个数从6个减少到3个，需要利用重线性化计数解决这个问题


  上述方法是一种非迭代方法，在迭代方法拥有良好初值时精度不如迭代方法，所以再用高斯牛顿法进行优化调整，选取使控制点距离变化最小的四个值  beta=[beta1, beta2, beta3, beta4].transpose()
  用高斯牛顿法来进行最小化   Error(beta) = sum_of_i_j_st_i<j(||cc(i)-cc(j)||.square()-||cw(i)-cw(j)||.square())
  ||cw(i)-cw(j)||.square()是已知的，控制点在相机参考系下的坐标可表示为关于beta的方程  cc(i)=sum_of_j_from_1_to_4(betaj*vj(i))
  优化在4个参数上进行，计算复杂度独立于三维二维点对应的个数，一般十次以内迭代即可收敛

  */
  cvMulTransposed(M, &MtM, 1);
  cvSVD(&MtM, &D, &Ut, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);
  cvReleaseMat(&M);

  double l_6x10[6 * 10], rho[6];
  CvMat L_6x10 = cvMat(6, 10, CV_64F, l_6x10);
  CvMat Rho    = cvMat(6,  1, CV_64F, rho);

  compute_L_6x10(ut, l_6x10);//x=beta1*v1+beta2*v2+beta3*v3+beta4*v4 这个函数表示出了几个beta满足的线性约束
  compute_rho(rho);   //求取控制点在世界坐标系下的坐标差平方，几对差分别为 0-1、0-2、0-3、1-2、1-3、2-3

  double Betas[4][4], rep_errors[4];
  double Rs[4][3][3], ts[4][3];

  /*
  对于MtM矩阵有N=1,N=2,N=3个零特征值的三种情况（不知为何这里没有计算N=4,Epnp论文里并未排除N=4的情况）
  1.先利用相机坐标系下的坐标差与世界坐标系下坐标差相等计算N=1,N=2,N=3这三种情况下的beta，
  2.然后再用高斯牛顿法修正求出的beta
  3.利用求出的beta计算控制点在相机坐标系下的坐标，然后可以根据alpha与控制点在相机坐标系下的坐标求取对应点在相机坐标系下的坐标，根据对应点在相机坐标系与世界坐标系下的坐标对求出rt，然后再利用旋转平移以及
  相机内参求出重投影误差，选取三种情况下重投影误差最小的那组。
  */
  find_betas_approx_1(&L_6x10, &Rho, Betas[1]);
  gauss_newton(&L_6x10, &Rho, Betas[1]);
  rep_errors[1] = compute_R_and_t(ut, Betas[1], Rs[1], ts[1]);

  find_betas_approx_2(&L_6x10, &Rho, Betas[2]);
  gauss_newton(&L_6x10, &Rho, Betas[2]);
  rep_errors[2] = compute_R_and_t(ut, Betas[2], Rs[2], ts[2]);

  find_betas_approx_3(&L_6x10, &Rho, Betas[3]);
  gauss_newton(&L_6x10, &Rho, Betas[3]);
  rep_errors[3] = compute_R_and_t(ut, Betas[3], Rs[3], ts[3]);
  /*取出重投影误差最小的那组rt*/
  int N = 1;
  if (rep_errors[2] < rep_errors[1]) N = 2;
  if (rep_errors[3] < rep_errors[N]) N = 3;
  /*将Rs[N],ts[N]的值拷贝到R,t中*/
  copy_R_and_t(Rs[N], ts[N], R, t);
  /*返回最终的重投影误差*/
  return rep_errors[N];
}

void PnPsolver::copy_R_and_t(const double R_src[3][3], const double t_src[3],
			double R_dst[3][3], double t_dst[3])
{
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      R_dst[i][j] = R_src[i][j];
    t_dst[i] = t_src[i];
  }
}

/*两点三维距离的平方*/
double PnPsolver::dist2(const double * p1, const double * p2)
{
  return
    (p1[0] - p2[0]) * (p1[0] - p2[0]) +
    (p1[1] - p2[1]) * (p1[1] - p2[1]) +
    (p1[2] - p2[2]) * (p1[2] - p2[2]);
}

double PnPsolver::dot(const double * v1, const double * v2)//求两个三维向量的点积
{
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

/*
R,t均为输入，是世界坐标系向相机坐标系的转换关系，利用rt与内参将对应点三维坐标从相机坐标系转换到图像坐标系，然后再求投影点与像素点的距离平方，最后取距离平方的平均
*/
double PnPsolver::reprojection_error(const double R[3][3], const double t[3])
{
  double sum2 = 0.0;

  /*
  |Xc|   |R[0]|        |t[0]|
  |Yc| = |R[1]| * pw + |t[1]|  Xc,Yc,Zc为从世界坐标系转换到相机坐标系的坐标
  |Zc|   |R[2]|        |t[2]|
  
  |Xc/Zc|
  |Yc/Zc|为归一化相机坐标系的坐标
  |  1  |

  |ue|   |fu  0  cu|   |Xc/Zc|   | fu*Xc/Zc + cu |
  |ve| = | 0 fv  cv| * |Yc/Zc| = | fv*Yc/Zc + cv |
  | 1|   | 0  0   1|   |  1  |   |       1       |
  */
  for(int i = 0; i < number_of_correspondences; i++) 
  {
    double * pw = pws + 3 * i;//指向对应点中的下一个三维点
    double Xc = dot(R[0], pw) + t[0];
    double Yc = dot(R[1], pw) + t[1];
    double inv_Zc = 1.0 / (dot(R[2], pw) + t[2]);//将对应三维点坐标从世界坐标系变换到相机坐标系
    double ue = uc + fu * Xc * inv_Zc;
    double ve = vc + fv * Yc * inv_Zc;
    double u = us[2 * i], v = us[2 * i + 1];//取出对应点的图像坐标

    sum2 += sqrt( (u - ue) * (u - ue) + (v - ve) * (v - ve) );
  }

  return sum2 / number_of_correspondences;//平均的坐标差平方
}


/*
对于两个三维空间中的点集合P={p1,p2,...,pn}与Q={q1,q2,...,qn},欲求取他们之间的刚体变换即R与t,可以建模为
(R,t)=argmin(sumof_i_from_1_to_n(wi*||(R*pi+t)-qi||.square()))   （1）     wi表示每个点对之间的权重
对于sumof_i_from_1_to_n(wi*||(R*pi+t)-qi||.square())对t求导，令导数为0可得
0 = (sumof_i_from_1_to_n(wi*||(R*pi+t)-qi||.square()))'=sumof_i_from_1_to_n(2*wi*(R*pi+t-qi))
= 2*t*sumof_i_from_1_to_n(wi)+2*R*sumof_i_from_1_to_n(wi*pi)-2*sumof_i_from_1_to_n(wi*qi)            (2)
引入点集P与Q的中心点 p_ave与q_ave  
p_ave=sumof_i_from_1_to_n(wi*pi)/sumof_i_from_1_to_n(wi)               (3)
q_ave=sumof_i_from_1_to_n(wi*qi)/sumof_i_from_1_to_n(wi)               (4)
公式（2）两边同时除以sumof_i_from_1_to_n(wi)可得
0/sumof_i_from_1_to_n(wi) = 2*t*sumof_i_from_1_to_n(wi)/sumof_i_from_1_to_n(wi)+2*R*sumof_i_from_1_to_n(wi*pi)/sumof_i_from_1_to_n(wi)-2*sumof_i_from_1_to_n(wi*qi)/sumof_i_from_1_to_n(wi)
0=2*t+2*R*p_ave-2*q_ave
q_ave-R*p_ave=t     (5)
将（5）带入（1）,则t被消去，可得
sumof_i_from_1_to_n(wi*||(R*pi+t)-qi||.square()) = sumof_i_from_1_to_n(wi*||R*pi+q_ave-R*p_ave-qi||.square()) = sumof_i_from_1_to_n(wi*||R*(pi-p_ave)-(qi-q_ave)||.square())        (6)
记 xi:=pi-p_ave     (7)
   yi:=qi-q_ave     (8)
此时（1）可以等价于
R = argmin(sumof_i_from_1_to_n(wi*||R*xi-yi||.square()))            (9)
将||R*xi-yi||.square()展开可得
||R*xi-yi||.square() = (R*xi-yi).transpose()*(R*xi-yi)
=(xi.transpose()*R.transpose()-yi.transpose())*(R*xi-yi) = xi.transpose()*R.transpose()*R*xi - yi.transpose()*R*xi-xi.transpose()*R.transpose()*yi+yi.transpose()*yi
由于R.transpose()*R=I    = xi.transpose()*xi-yi.transpose()*R*xi-xi.transpose()*R.transpose()*yi+yi.transpose()*yi                   (10)
(10)中最终等号后面的4项均为标量，一个标量与其转置相等，故而
xi.transpose()*R.transpose()*yi = (xi.transpose()*R.transpose()*yi).transpose() = yi.transpose()*R*xi           (11)
则（10）可写为     ||R*xi-yi||.square()=xi.transpose()*xi-2*yi.transpose()*R*xi+yi.transpose()*yi           (12)
则（9）式可写为
argmin(sumof_i_from_1_to_n(wi*||R*xi-yi||.square())) = argmin(sumof_i_from_1_to_n(wi*(xi.transpose()*xi-2*yi.transpose()*R*xi+yi.transpose()*yi)))
=argmin(sumof_i_from_1_to_n(wi*xi.transpose()*xi) - 2*sumof_i_from_1_to_n(wi*yi.transpose()*R*xi) + sumof_i_from_1_to_n(wi*yi.transpose()*yi))                    (13)
由于sumof_i_from_1_to_n(wi*xi.transpose()*xi)以及sumof_i_from_1_to_n(wi*yi.transpose()*yi)与R无关，因此
argmin(sumof_i_from_1_to_n(wi*||R*xi-yi||.square()))=argmin(-2*sumof_i_from_1_to_n(wi*yi.transpose()*R*xi))=argmax(sumof_i_from_1_to_n(wi*yi.transpose()*R*xi))              (14)
写成矩阵形式
                      |w1   0  0  ...  0 |   |y1.transpose()|
                      | 0  w2  0  ...  0 |   |y2.transpose()|
W*Y.transpose()*R*X = | 0 ... ... ... ...| * |     ...      |* R * |x1  x2  ...  xn|
                      | 0 ... .... 0   wn|   |yn.transpose()|
  |w1*y1.transpose()|                             |w1*y1.transpose()*R*x1  w1*y1.transpose()*R*x2  ...  w1*y1.transpose()*R*xn|
  |w2*y2.transpose()|                             |w2*y2.transpose()*R*x1  w2*y2.transpose()*R*x2  ...  w2*y2.transpose()*R*xn|
= |      ...        | * |R*x1  R*x2  ...  R*xn| = |        ...                     ...             ...          ...           |
  |wn*yn.transpose()|                             |wn*yn.transpose()*R*x1  wn*yn.transpose()*R*x2  ...  wn*yn.transpose()*R*xn|
可得 sumof_i_from_1_to_n(wi*yi.transpose()*R*xi) = tr(W*Y.transpose()*R*X)        (15)
如上所述，其中 W=diag(w1, w2,..., wn)  Y=|y1  y2  ...  yn|  X=|x1  x2  ...  xn|
矩阵的迹有  tr(A*B)=tr(B*A)         (16)
则                                                                          （n*n*n*3）*(3*3*3*n)        (3*3*3*n)*(n*n*n*3)
sumof_i_from_1_to_n(wi*yi.transpose()*R*xi) = tr(W*Y.transpose()*R*X) = tr((W*Y.transpose())*(R*X)) = tr((R*X)*(W*Y.transpose())) = tr(R*X*W*Y.transpose())                  (17)
记S=X*W*Y.transpose()  S的SVD分解记为S=U*D*V.transpose()
sumof_i_from_1_to_n(wi*yi.transpose()*R*xi) = tr(R*X*W*Y.transpose()) = tr(R*S) = tr(R*U*D*V.transpose()) = tr((D*V.transpose())*(R*U)) = tr(D*V.transpose()*R*U)            (18)
由于 V,R,U为正交阵，所以M=V.transpose()*R*U同样为正交阵，即对于M矩阵中每一列mj有mj.transpose()*mj=1
因此  
          | D1   0  ...  ... |   | m11  m12  ...  m1d |
          |  0  D2  ...  ... |   | m21  m22  ...  m2d |
tr(D*M) = |  0   0  ...  ... | * | ...  ...  ...  ... | = sumof_i_from_1_to_n(Di*mii) <= sumof_i_from_1_to_n(Di)                            （19）
          |  0   0  ...   Dn |   | mn1  mn2  ...  mdd |
由mj.transpose()*mj=1的约束可得当m为单位阵时，sumof_i_from_1_to_n(wi*yi.transpose()*R*xi)最大
I=M=V.transpose()*R*U          V=R*U         R=V*U.transpose()   计算出R之后，平移矩阵 t = q_ave-R*p_ave


省去上述推到过程，对于编码来说需要以下几步
1.对于每一对对应点，求取 xi:=pi-p_ave  yi:=qi-q_ave   其中p_ave与q_ave分别为点集P与Q中的点的平均坐标
2.求S=X*Y.transpose()   其中Y=|y1  y2  ...  yn|  X=|x1  x2  ...  xn|
3.求S的SVD分解 记为S=U*D*V.transpose()
4.R=V*U.transpose()
5.t = q_ave-R*p_ave
*/
/*
这个函数求的是pc=R*pw+t
*/
void PnPsolver::estimate_R_and_t(double R[3][3], double t[3])
{
  double pc0[3], pw0[3];

  pc0[0] = pc0[1] = pc0[2] = 0.0;
  pw0[0] = pw0[1] = pw0[2] = 0.0;

  /*
  以下两个for循环求取所有对应点在世界坐标系以及相机坐标系下的平均坐标
  */
  for(int i = 0; i < number_of_correspondences; i++) 
  {
    const double * pc = pcs + 3 * i;
    const double * pw = pws + 3 * i;

    for(int j = 0; j < 3; j++) 
    {
      pc0[j] += pc[j];
      pw0[j] += pw[j];
    }
  }
  for(int j = 0; j < 3; j++) {
    pc0[j] /= number_of_correspondences;
    pw0[j] /= number_of_correspondences;
  }

  double abt[3 * 3], abt_d[3], abt_u[3 * 3], abt_v[3 * 3];
  CvMat ABt   = cvMat(3, 3, CV_64F, abt);
  CvMat ABt_D = cvMat(3, 1, CV_64F, abt_d);
  CvMat ABt_U = cvMat(3, 3, CV_64F, abt_u);
  CvMat ABt_V = cvMat(3, 3, CV_64F, abt_v);

  cvSetZero(&ABt);
  /*
  对于每对对应点，分别取出其在相机坐标系以及世界坐标系下的坐标，此处减去平均坐标的目的是进行坐标中心化，可以减少有效位数，保证计算精度

  */
  for(int i = 0; i < number_of_correspondences; i++) {
    double * pc = pcs + 3 * i;
    double * pw = pws + 3 * i;
    /*
    求(pw-pw0)*(pc-pc0.transpose())
    | pw(0,0)  pw(1,0)  ...  pw(n,0) |   | pc(0,0)  pc(0,1)  pc(0,2) | 
    | pw(0,1)  pw(1,0)  ...  pw(n,0) | * |   ...      ...      ...   |
    | pw(0,2)  pw(1,0)  ...  pw(n,0) |   | pc(n,0)  pc(n,1)  pc(n,2) |
    对于单次循环，向其中加的是
    | pwi(0) |                                | pwi(0)*pci(0)  pwi(0)*pci(1)  pwi(0)*pci(2) |
    | pwi(1) | * | pci(0)  pci(1)  pci(2) | = | pwi(1)*pci(0)  pwi(1)*pci(1)  pwi(1)*pci(2) |
    | pwi(2) |                                | pwi(2)*pci(0)  pwi(2)*pci(1)  pwi(2)*pci(2) |
    对于函数名字前的注释中算法的5步  第二步S=X*Y.transpose()  这里代码循环中求的是Y*X.transpose()=S.transpose()=S_new
    第三步 S=U*D*V.transpose()  S_new = S.transpose() = V*D.transpose()*U.transpose() 由于D为对角阵 = V*D*U.transpose() = U_new*D*V_new.transpose()
    第四步 R=V*U.transpose() = U_new*V_new.transpose()
    这里从一开始就求了转置，避免了SVD后再转置
    */
    for(int j = 0; j < 3; j++) {
      abt[3 * j    ] += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);
      abt[3 * j + 1] += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
      abt[3 * j + 2] += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
    }
  }

  cvSVD(&ABt, &ABt_D, &ABt_U, &ABt_V, CV_SVD_MODIFY_A);
  //U*V(i,j)中的元素即为U矩阵第i行与V矩阵第j列的点积，即为U矩阵第i行与V转置第j行的点积
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
      R[i][j] = dot(abt_u + 3 * i, abt_v + 3 * j);//不用对v转置，直接取V中j行就相当于取V.transpose()中的j列了

  const double det =
    R[0][0] * R[1][1] * R[2][2] + R[0][1] * R[1][2] * R[2][0] + R[0][2] * R[1][0] * R[2][1] -
    R[0][2] * R[1][1] * R[2][0] - R[0][1] * R[1][0] * R[2][2] - R[0][0] * R[1][2] * R[2][1];

  if (det < 0) 
  {
    R[2][0] = -R[2][0];
    R[2][1] = -R[2][1];
    R[2][2] = -R[2][2];
  }

  t[0] = pc0[0] - dot(R[0], pw0);
  t[1] = pc0[1] - dot(R[1], pw0);
  t[2] = pc0[2] - dot(R[2], pw0);
}

/*
输出[r|t]矩阵（3*4）
*/
void PnPsolver::print_pose(const double R[3][3], const double t[3])
{
  cout << R[0][0] << " " << R[0][1] << " " << R[0][2] << " " << t[0] << endl;
  cout << R[1][0] << " " << R[1][1] << " " << R[1][2] << " " << t[1] << endl;
  cout << R[2][0] << " " << R[2][1] << " " << R[2][2] << " " << t[2] << endl;
}

/*
若求出的第一个对应点在相机坐标系下深度为负，则将所有控制点在相机坐标系下坐标以及对应点在相机坐标系下坐标全部取反
*/
void PnPsolver::solve_for_sign(void)
{
  if (pcs[2] < 0.0) {
    for(int i = 0; i < 4; i++)
      for(int j = 0; j < 3; j++)
	ccs[i][j] = -ccs[i][j];

    for(int i = 0; i < number_of_correspondences; i++) {
      pcs[3 * i    ] = -pcs[3 * i];
      pcs[3 * i + 1] = -pcs[3 * i + 1];
      pcs[3 * i + 2] = -pcs[3 * i + 2];
    }
  }
}

double PnPsolver::compute_R_and_t(const double * ut, const double * betas,
			     double R[3][3], double t[3])
{
  /*取出M矩阵的最后4个特征向量，利用高斯牛顿法求出的beta对四个特征值加权对于i=0,1,2,3，ccs[i][0]、ccs[i][1]、ccs[i][2]存的是第i点的x,y,z坐标*/
  compute_ccs(betas, ut);
  /*利用alpha以及compute_ccs求出的控制点在相机坐标系下的坐标求对应点在相机坐标系下的坐标
  compute_pcs()求出了所有对应点在相机坐标系下的三维坐标*/
  compute_pcs();

  solve_for_sign();

  estimate_R_and_t(R, t);

  return reprojection_error(R, t);
  /*
  R,t均为输入，是世界坐标系向相机坐标系的转换关系，利用rt与内参将对应点三维坐标从相机坐标系转换到图像坐标系，然后再求投影点与像素点的距离平方，最后取距离平方的平均
  */
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]
//这里表示的是N=的情况，N指的是M矩阵零空间的维度，根据论文M矩阵零空间维度可能是1,2,3,4
void PnPsolver::find_betas_approx_1(const CvMat * L_6x10, const CvMat * Rho,
			       double * betas)
{
  double l_6x4[6 * 4], b4[4];
  CvMat L_6x4 = cvMat(6, 4, CV_64F, l_6x4);
  CvMat B4    = cvMat(4, 1, CV_64F, b4);

  for(int i = 0; i < 6; i++) {
    cvmSet(&L_6x4, i, 0, cvmGet(L_6x10, i, 0));
    cvmSet(&L_6x4, i, 1, cvmGet(L_6x10, i, 1));
    cvmSet(&L_6x4, i, 2, cvmGet(L_6x10, i, 3));
    cvmSet(&L_6x4, i, 3, cvmGet(L_6x10, i, 6));
  }

  cvSolve(&L_6x4, Rho, &B4, CV_SVD);//L_6x4*B4=Rho 

  if (b4[0] < 0) {
    betas[0] = sqrt(-b4[0]);
    betas[1] = -b4[1] / betas[0];
    betas[2] = -b4[2] / betas[0];
    betas[3] = -b4[3] / betas[0];
  } else {
    betas[0] = sqrt(b4[0]);
    betas[1] = b4[1] / betas[0];
    betas[2] = b4[2] / betas[0];
    betas[3] = b4[3] / betas[0];
  }
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]

void PnPsolver::find_betas_approx_2(const CvMat * L_6x10, const CvMat * Rho,
			       double * betas)
{
  double l_6x3[6 * 3], b3[3];
  CvMat L_6x3  = cvMat(6, 3, CV_64F, l_6x3);
  CvMat B3     = cvMat(3, 1, CV_64F, b3);

  for(int i = 0; i < 6; i++) {
    cvmSet(&L_6x3, i, 0, cvmGet(L_6x10, i, 0));
    cvmSet(&L_6x3, i, 1, cvmGet(L_6x10, i, 1));
    cvmSet(&L_6x3, i, 2, cvmGet(L_6x10, i, 2));
  }

  cvSolve(&L_6x3, Rho, &B3, CV_SVD);
  //若B11与B22同号，则betas[1]为sqrt(fabs(B22)),否则为0
  if (b3[0] < 0) {
    betas[0] = sqrt(-b3[0]);
    betas[1] = (b3[2] < 0) ? sqrt(-b3[2]) : 0.0;
  } else {
    betas[0] = sqrt(b3[0]);
    betas[1] = (b3[2] > 0) ? sqrt(b3[2]) : 0.0;
  }

  if (b3[1] < 0) betas[0] = -betas[0];

  betas[2] = 0.0;
  betas[3] = 0.0;
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]

void PnPsolver::find_betas_approx_3(const CvMat * L_6x10, const CvMat * Rho,
			       double * betas)
{
  double l_6x5[6 * 5], b5[5];
  CvMat L_6x5 = cvMat(6, 5, CV_64F, l_6x5);
  CvMat B5    = cvMat(5, 1, CV_64F, b5);

  for(int i = 0; i < 6; i++) {
    cvmSet(&L_6x5, i, 0, cvmGet(L_6x10, i, 0));
    cvmSet(&L_6x5, i, 1, cvmGet(L_6x10, i, 1));
    cvmSet(&L_6x5, i, 2, cvmGet(L_6x10, i, 2));
    cvmSet(&L_6x5, i, 3, cvmGet(L_6x10, i, 3));
    cvmSet(&L_6x5, i, 4, cvmGet(L_6x10, i, 4));
  }

  cvSolve(&L_6x5, Rho, &B5, CV_SVD);

  if (b5[0] < 0) {
    betas[0] = sqrt(-b5[0]);
    betas[1] = (b5[2] < 0) ? sqrt(-b5[2]) : 0.0;
  } else {
    betas[0] = sqrt(b5[0]);
    betas[1] = (b5[2] > 0) ? sqrt(b5[2]) : 0.0;
  }//经过上述过程betas[0]与betas[1]均为正，若b5[1]<0即betas[0]*betas[1]<0，则将betas[0]改为负数
  if (b5[1] < 0) betas[0] = -betas[0];
  betas[2] = b5[3] / betas[0];
  betas[3] = 0.0;
}

/*
利用世界坐标系距离等于相机坐标系下距离的原理得出 L*beta=raw
此处6指的是1-2,1-3,1-4,2-3,2-4,3-4六对点的坐标差
x=beta1*v1+beta2*v2+beta3*v3+beta4*v4  这里此函数表示出了beta1、beta2、beta3、beta4满足的线性约束
*/
void PnPsolver::compute_L_6x10(const double * ut, double * l_6x10)
{
  const double * v[4];

  v[0] = ut + 12 * 11;//对应最小的4个特征值的特征向量
  v[1] = ut + 12 * 10;
  v[2] = ut + 12 *  9;
  v[3] = ut + 12 *  8;

  double dv[4][6][3];

  /*
  j=0时a=0 b=1  dv[i][0][0] = v[i][0]-v[i][3]   dv[i][0][1] = v[i][1]-v[i][4]   dv[i][0][2] = v[i][2]-v[i][5]
  j=1时a=0 b=2  dv[i][1][0] = v[i][0]-v[i][6]   dv[i][1][1] = v[i][1]-v[i][7]   dv[i][1][2] = v[i][2]-v[i][8]
  j=2时a=0 b=3  dv[i][2][0] = v[i][0]-v[i][9]   dv[i][2][1] = v[i][1]-v[i][10]  dv[i][2][2] = v[i][2]-v[i][11]
  j=3时a=1 b=2  dv[i][3][0] = v[i][3]-v[i][6]   dv[i][3][1] = v[i][4]-v[i][7]   dv[i][3][2] = v[i][5]-v[i][8]
  j=4时a=1 b=3  dv[i][4][0] = v[i][3]-v[i][9]   dv[i][4][1] = v[i][4]-v[i][10]  dv[i][4][2] = v[i][5]-v[i][11]
  j=5时a=2 b=3  dv[i][5][0] = v[i][6]-v[i][9]   dv[i][5][1] = v[i][7]-v[i][10]  dv[i][5][2] = v[i][8]-v[i][11]
  */
  for(int i = 0; i < 4; i++) {  //此处4指的是四个特征值为0的特征向量
    int a = 0, b = 1;
    for(int j = 0; j < 6; j++) {  //此处6指的是1-2,1-3,1-4,2-3,2-4,3-4六对点的坐标差
      dv[i][j][0] = v[i][3 * a    ] - v[i][3 * b];
      dv[i][j][1] = v[i][3 * a + 1] - v[i][3 * b + 1];
      dv[i][j][2] = v[i][3 * a + 2] - v[i][3 * b + 2];

      b++;
      if (b > 3) {
        a++;
        b = a + 1;
      }
    }
  }
  /*
  由控制点在相机坐标系下距离等于在世界坐标系下距离相等可知
  ||(beta1*v1(i)+beta2*v2(i)+beta3*v3(i)+beta4*v4(i)) - (beta1*v1(j)+beta2*v2(j))+beta3*v3(j)+beta4*v4(j))||.square() = ||cw(i)-cw(j))||.square()
  等式左边可写为以下形式
  = ||(v1(i)-v1(j))*beta1 + (v2(i)-v2(j)))*beta2 + (v3(i)-v3(j))*beta3 + (v4(i)-v4(j))*beta4||.square()  其中v1、v2、v3、v4指4个特征向量 i,j在[1,4]区间内指其中第i个与第j个控制点
  = ||delta1*beta1 + delta2*beta2 + delta3*beta3 + delta4*beta4||.square() 其中delta1、delta2、delta3、delta4为向量，beta1、beta2、beta3、beta4为数值
  = dot(delta1,delta1)*beta11 + 2*dot(delta1,delta2)*beta12 + 2*dot(delta1,delta3)*beta13 + 2*dot(delta1, delta4)*beta14 + dot(delta2, delta2)*beta22
   + 2*dot(delta2,delta3)*beta23 + 2*dot(delta2, delta4)*beta24 + dot(delta3, delta3)*beta33 + 2*dot(delta3,delta4)*beta34 + dot(delta4, delta4)*beta44
  对应下方式子相当于
  row[0]*beta11 + row[1]*beta12 + row[2]*beta22 + row[3]*beta13 + row[4]*beta23 + row[5]*beta33 + row[6]*beta14 + row[7]*beta24 + row[8]*beta34 + row[9]*beta44
  */
  for(int i = 0; i < 6; i++) {
    double * row = l_6x10 + 10 * i;

    row[0] =        dot(dv[0][i], dv[0][i]);
    row[1] = 2.0f * dot(dv[0][i], dv[1][i]);
    row[2] =        dot(dv[1][i], dv[1][i]);
    row[3] = 2.0f * dot(dv[0][i], dv[2][i]);
    row[4] = 2.0f * dot(dv[1][i], dv[2][i]);
    row[5] =        dot(dv[2][i], dv[2][i]);
    row[6] = 2.0f * dot(dv[0][i], dv[3][i]);
    row[7] = 2.0f * dot(dv[1][i], dv[3][i]);
    row[8] = 2.0f * dot(dv[2][i], dv[3][i]);
    row[9] =        dot(dv[3][i], dv[3][i]);
  }
}

//求取控制点在世界坐标系下的坐标差平方，几对差分别为 0-1、0-2、0-3、1-2、1-3、2-3
void PnPsolver::compute_rho(double * rho)
{
  rho[0] = dist2(cws[0], cws[1]);
  rho[1] = dist2(cws[0], cws[2]);
  rho[2] = dist2(cws[0], cws[3]);
  rho[3] = dist2(cws[1], cws[2]);
  rho[4] = dist2(cws[1], cws[3]);
  rho[5] = dist2(cws[2], cws[3]);
}

/*
l_6x10尺寸为6*10,rho尺寸为6*1，A尺寸为6*4,b尺寸为6*1
rowL[0]*beta11 + rowL[1]*beta12 + rowL[2]*beta22 + rowL[3]*beta13 + rowL[4]*beta23 + rowL[5]*beta33 + rowL[6]*beta14 + rowL[7]*beta24 + rowL[8]*beta34 + rowL[9]*beta44 = ||cw(i)-cw(j)||.square()
rho每一行中存储的是||cw(i)-cw(j)||.square(),所以b中存储的即为世界坐标系坐标差与相机坐标系坐标差的平方
rowA[0]、rowA[1]、rowA[2]、rowA[3]中存储的是控制点在相机坐标系下的距离关于beta1、beta2、beta3、beta4的导数
这个函数是由l_6x10、rho、betas作为输入，A与b作为输出   功能是根据(1)控制点在相机坐标系下距离关于beta的关系（l_6x10）(2)控制点在世界坐标系下的距离rho (3)各个beta
求出(1)控制点在相机坐标系下距离关于beta的关系A (2)控制点在相机坐标系下距离与在世界坐标系下距离的残差
*/
void PnPsolver::compute_A_and_b_gauss_newton(const double * l_6x10, const double * rho,
					double betas[4], CvMat * A, CvMat * b)
{
  for(int i = 0; i < 6; i++) {
    const double * rowL = l_6x10 + i * 10;
    double * rowA = A->data.db + i * 4;

    rowA[0] = 2 * rowL[0] * betas[0] +     rowL[1] * betas[1] +     rowL[3] * betas[2] +     rowL[6] * betas[3];
    rowA[1] =     rowL[1] * betas[0] + 2 * rowL[2] * betas[1] +     rowL[4] * betas[2] +     rowL[7] * betas[3];
    rowA[2] =     rowL[3] * betas[0] +     rowL[4] * betas[1] + 2 * rowL[5] * betas[2] +     rowL[8] * betas[3];
    rowA[3] =     rowL[6] * betas[0] +     rowL[7] * betas[1] +     rowL[8] * betas[2] + 2 * rowL[9] * betas[3];

    cvmSet(b, i, 0, rho[i] -
	   (
	    rowL[0] * betas[0] * betas[0] +
	    rowL[1] * betas[0] * betas[1] +
	    rowL[2] * betas[1] * betas[1] +
	    rowL[3] * betas[0] * betas[2] +
	    rowL[4] * betas[1] * betas[2] +
	    rowL[5] * betas[2] * betas[2] +
	    rowL[6] * betas[0] * betas[3] +
	    rowL[7] * betas[1] * betas[3] +
	    rowL[8] * betas[2] * betas[3] +
	    rowL[9] * betas[3] * betas[3]
	    ));
  }
}


//高斯牛顿法 x(k+1)=x(k)+delta  delta=(Jr.transpose()*Jr).inverse()*Jr.transpose()*r
void PnPsolver::gauss_newton(const CvMat * L_6x10, const CvMat * Rho,
			double betas[4])
{
  const int iterations_number = 5;

  double a[6*4], b[6], x[4];//6指六对控制点距离差，4指beta1,beta2,beta3,beta4
  CvMat A = cvMat(6, 4, CV_64F, a);
  CvMat B = cvMat(6, 1, CV_64F, b);
  CvMat X = cvMat(4, 1, CV_64F, x);

  for(int k = 0; k < iterations_number; k++) {
    //求出导数A以及残差b方便计算高斯牛顿
    compute_A_and_b_gauss_newton(L_6x10->data.db, Rho->data.db,//db指double类型的数据
				 betas, &A, &B);
    qr_solve(&A, &B, &X);
    /*将上方求出的增量X加到此次迭代开始的初值上*/
    for(int i = 0; i < 4; i++)
      betas[i] += x[i];
  }
}


/*
若n阶复方阵U的n个列向量是U空间的一个标准正交基，则U为酉矩阵。酉矩阵是正交矩阵向复数域上的推广

若非奇异矩阵A能够转化为正交（酉）矩阵Q与非奇异上三角矩阵R的乘积，即A=QR，则称其为A的QR分解  Am*n = Qm*nRn*n  其中m>n

<householder变换>将一个向量变换为由一个超平面反射的镜像，是一种线性变换，超平面的法向量被称为householder向量
对于两个不相等的n维向量x,y；x!=y但是x与y模长相等，则存在householder矩阵 H=I-2*U*U.transpose/||U||.square() 使得Hx=y，其中U=x-y

<施密特正交化>是求欧式空间正交基的一种方法，从欧式空间任意线性无关的向量组alpha(1),alpha(2),...,alpha(m)出发，求得正交向量组beta(1),beta(2),...,beta(n),使alpha(1),alpha(2),...,alpha(m)与
beta(1),beta(2),...,beta(n)等价，再将正交向量组中每个向量单位化，即得一个标准正交向量组，这种方法称为施密特正交化
求解方法为  beta(1)=alpha(1)
beta(2) = alpha(2) - <alpha(2),beta(1)>/<beta(1),beta(1)>*beta(1)
beta(m) = alpha(m) - <alpha(m),beta(1)>/<beta(1),beta(1)>*beta(1) - <alpha(m),beta(2)>/<beta(2),beta(2)>*beta(2) - ... - <alpha(m),beta(m-1)>/<beta(m-1),beta(m-1)>*beta(m-1) 
则beta(1), beta(2),..., beta(m)是一个正交向量组，再令 e(i)=beta(i)/||beta(i)||(i=1,2,...,m)
则得到标准正交向量组 e(1),e(2),...,e(m),且该向量组与alpha(1),alpha(2),...,alpha(m)等价

对于超定问题Ax=b,A为n*k维，x为k*1维，b为n*1维， A.transpose()*A*x = A.transpose()*b  x=(A.transpose()*A).inverse()*A.transpose()*b
若对矩阵A进行QR分解，Q为n*k维，R为k*k维, Q*R*x=b  Q.transpose*Q*R*x=Q.transpose()*b，由于Q为正交阵，即为 R*x=Q.transpose()*b   得到x=R.inverse()*Q.transpose()*b

高斯牛顿法 x(k+1)=x(k)+delta  delta=(Jr.transpose()*Jr).inverse()*Jr.transpose()*r   

*/
/*
关于矩阵初等变换的知识
对于m*n维矩阵A,向量x=[x1,x2,...,xn].transpose(),y=[y1,y2,...,ym]
公式1： 记A=[a1, a2, .., an],其中a1,..,an为m*1维的列向量，A*x=sumof_i_from_1_to_n(ai*xi)
公式2： 记A=[a1, a2, .., am].transpose(), y*A = sumof_i_from_1_to_m(yi*ai),此处ai代表A矩阵第i行
上面两个公式表明，矩阵乘以列向量等于向量中每个元素乘以对应列再相加，行向量乘以矩阵等于行向量每个元素乘以矩阵对应行然后再相加

初等矩阵左乘，相当于行变换；初等矩阵右乘，相当于列变换

初等行（列）变换包含以下三种操作：
操作1： 以一个非0的数乘以矩阵的某一行（列）
操作2： 把矩阵某一行（列）的c倍加到另一行（列）
操作3： 互换矩阵中两行（列）的位置

初等变换的五条相关性质：
性质1： 矩阵转置，行列式不变
性质2： 某行（列）乘以k,则行列式变为原始行列式的k倍
性质3： 若两行（列）相同或成比例，则行列式为0
性质4： 将某行（列）倍数加到另一行（列），行列式不变
性质5： 互换行列式中两行（列）的位置，行列式反号


*/

/*
下三角矩阵的逆仍为下三角矩阵，上三角矩阵的逆仍为上三角矩阵
若m*n矩阵A的列线性无关，则可以被分解为
                     |R11  R12  ...  R1n|  
                     | 0   R22  ...  R2n|
A = [q1 q2 ... qn] * |...  ...  ...  ...|
                     | 0    0   ...  Rnn|
q1,...,qn为m阶正交向量， ||qi||=1   当i!=j时qi.transpose()*qj=0
对角元素 Rii非0
若Rii<0,可改变Rii,...,Rin以及qi的符号 注：矩阵相乘时对于A(m,n) 是 q1(m)*R1n + q2(m)*R2n + ... + qn(m)*Rnn 故而qi中的元素会被R的第i行影响
大部分定义要求Rii>0,这个要求会令Q与R唯一
A = QR
Q为包含正交列的m*n阶矩阵 Q.transpose()*Q = I
若A为方阵则Q为正交阵 Q.transpose()*Q = Q*Q.transpose() = I
R为n*n上三角矩阵，对角元素非零
R不是奇异阵（对角元素非零）

给定一个矩阵A，A的值域（range）通常定义为由A的列向量张成的线性空间

QR分解可以解决（1）线性方程  （2）最小二乘问题   （3）带约束的最小二乘问题
可以为下列问题提供简单形式 （1）含有线性不想关列向量的矩阵广义逆  （2）非奇异阵的逆 (3)线性无关列的矩阵的值域

对于广义逆问题  A_pseudo_inverse = ((QR).transpose()*(QR)).inverse()*(QR).transpose()
= (R.transpose()*Q.transpose()*Q*R).inverse()*R.transpose()*Q.transpose() 由于Q为正交阵 Q.transpose()*Q=I
= (R.transpose()*R).inverse()*R.transpose()*Q.transpose() = R.inverse()*(R.transpose()).inverse()*R.transpose()*Q.transpose() = R.inverse()*Q.transpose()

A矩阵的逆为 A.inverse()=(QR).inverse()=R.inverse()*Q.transpose()

Q与A有相同的range
y在A的值域中 <==> y=Ax <==> y=QRX <==> y=Qz <==> y在Q的值域中

A*A_pseudo_inverse = Q*R*R.inverse()*Q.transpose() = Q*Q.transpose()                     A_pseodo_inverse*A = I


求QR分解的方法
一、Gram-Schmidt算法
    一列一列地计算Q与R
    在k步之后有一个局部QR分解
                         |R11  R12  ...  R1k|  
                         | 0   R22  ...  R2k|
    A = [q1 q2 ... qk] * |...  ...  ...  ...|  对于R矩阵中的每一列i，R(i,i)对应的第i列下方的i之后所有行值均为0,所以可以直接对q截取在k列之前并只取R的左上角
                         | 0    0   ...  Rkk|  
    q1,...,qk这些列是正交的
    R11,R22,...,Rkk这些对角元素均为正  q1,...,qk与a1,...,ak具有同样的span

    假设已经计算了前k-1列的分解
    A=QR的第k列为   ak=R1k*q1+R2k*q2+...+Rk-1,k*qk-1+Rkk*qk
    忽略选取R1k,...,Rk-1,k的方法 qk_estimate = ak-R1k*q1-R2k*q2-...-Rk-1,k*qk-1
    由于a1,a2,...,ak线性无关，因此ak不在{a1,...,ak-1}的值域中，由于{q1,...,qk}与{a1,...,ak}具有同样的值域，所以ak也不在{q1,...,qk-1}的值域中
    qk为qk_estimate的归一化，令Rkk=||qk_estimate||,qk=qk_estimate/Rkk
    若选取R1k,...,Rk-1,k 为 R1k=q1.transpose()*ak  R2k=q2.transpose()*ak   Rk-1,k=qk-1.transpose()*ak 则qk、qk_estimate与q1,...,qk-1正交

    Gram-Schmidt算法
    输入： m*n阶拥有线性无关列向量a1,a2,...,an的矩阵A
    算法：
        对于k=1到n
            R1k = q1.transpose()*ak
            R2k = q2.transpose()*ak
            ...
            Rk-1,k = qk-1.transpose()*ak
            qk_estimate = ak-(R1k*q1+R2k*q2+...+Rk-1,k*qk-1)
            Rkk = ||qk_estimate||
            qk = qk_estimate/Rkk


二、householder算法
    是目前应用最广泛的算法
    可以计算一个全的qr分解

    A=[Q Q_es]*|R|          [Q Q_es]为正交阵
               |0|
    完整的Q由正交矩阵的积组成    [Q Q_es]=H1*H2*...*Hn  每一个Hi均为一个m*m的对称正交  H=I-2*v*v.transpose() 其中||v||=1
    H*x是x关于超平面的对称 v.transpose()*z=0
    H对称且H正交
    H*x = (I - 2*v*v.transpose())*x = x - 2*v*v.transpose()*x = x - 2*v*(v.transpose()*x)         其中v.transpose()*x为一个常数
        = x - 2*(v.transpose()*x)*v
    
    给定一个非零p维向量 y=(y1, y2,...,yp),定义
        | y1+sign(y1)*||y|| |
    w = |        y2         |     v=w/||w||         定义 sign(0)=1
        |       ...         |
        |        yp         |
    由于||y||的系数表明此项与y1同号，所以w1=+-(|y1|+||y||)
    ||w||.square() = w.transpose()*w = w1*w1＋w2*w2+...+wp*wp = (|y1|+||y||)*(|y1|+||y||)+y2*y2+...+yp*yp = ||y||*||y||+2*||y||*|y1|+y1*y1+y2*y2+...+yp*yp
                   = ||y||*||y||+2*||y||*|y1|+||y||*||y|| = 2*||y||*(||y||+|y1|) = 2*(w.transpose()*y)
    H = I-2*v*v.transpose()可以将y映射为与e1=(1, 0, ..., 0).transpose()的乘积
    H*y = y-2*(w.transpose()*y)/||w||.square()*w = y-||w||.square()/||w||.square()*w = y-w = -sign(y1)*||y||*e1
    H将向量y映射到向量-sign(y1)*||y||*e1
    计算将A矩阵减少到三角阵形式的映射矩阵H1,...,Hn
                       | R |
    Hn*Hn-1*...*H1*A = | 0 |
    在k步之后，矩阵Hk*Hk-1*...*H1*A有下列形式
    |x x x x|
    |0 x x x|
    |0 0 y y|  x表示存在普通的元素           在A逐渐变为上三角阵的过程中,在第k次迭代时只处理还不是上三角部分的子矩阵，对应左图即为元素是y的那些值
    |0 0 y y|                              按照上方思路即为慢慢将1到n列对角线以下的元素变为只有第一个元素有值的列向量，此时研究的列向量不再是1*m维,而是1*(m-k+1)维
    |0 0 y y|即前k行k列子阵已变为对角阵，k列以后均正常有值

    Householder算法
        将A矩阵映射为| R|
                    | 0|
        算法：
            对于k=1到n
            1.定义 y=Ak:m,k然后计算(m-k+1)维向量vk:  w=y+sign(y1)*||y||*e1   vk=1/||w||*w
            2.计算Ak:m,k:n被映射I-2*vk*vk.transpose()映射的结果   Ak:m,k:n := Ak:m,k:n-2*vk*(vk.transpose()*Ak:m,k:n)
            第二步中对A的子阵 Ak:m,k:n进行该操作相当于A矩阵整体左乘Hk=|  I((k-1)*(k-1)维)             0((k-1)*(m-k+1)维)             | = I-2*| 0 |*| 0 |.transpose()
                                                                  | 0((m-k+1)*(k-1)维)  I-2*vk*vk.transpose()((m-k+1)*(m-k+1)维) |       | vk| | vk|
            上述算法逐渐将A变换为|R|,返回一系列向量v1,...,vn，其中vk的长度为m-k+1
                               |0|
            返回向量v1,...,vn的householder算法对应的q为  [ Q Q_es] = H1*H2*...*Hn
            通常没有必要明确计算[ Q Q_es ]
            向量v1,...,vn是[ Q Q_es ]的一种表达
            [ Q Q_es]以及其转置与向量的积可以以下列形式计算
            [ Q Q_es]*x = H1*H2*...*Hn*x
            [ Q Q_es].transpose()*y = Hn*Hn-1*...*H1*y
*/

/*
高斯牛顿法中增量为（Jr.transpose()*Jr）.inverse()*Jr.transpose()*r 即为R.inverse()*Q.transpose()*r  
对于这个函数，若A=QA*RA 欲求取的是  (A.transpose()*A).inverse()*A.transpose()*b=RA.inverse()*QA.transpose()*r
*/
void PnPsolver::qr_solve(CvMat * A, CvMat * b, CvMat * X)//A为6*4、b为6*1，A为相机坐标系下两点距离平方关于4个beta的导数，b为相机坐标系下两点距离平方与世界坐标系下两点距离平方的差
{
  static int max_nr = 0;
  static double * A1, * A2;

  const int nr = A->rows;
  const int nc = A->cols;

  if (max_nr != 0 && max_nr < nr) 
  {
    delete [] A1;
    delete [] A2;
  }
  if (max_nr < nr) 
  {
    max_nr = nr;
    A1 = new double[nr];//A1与A2均指向nr个double
    A2 = new double[nr];
  }

  double * pA = A->data.db, * ppAkk = pA;//pA表示指向A矩阵的指针、ppAkk表示指向A矩阵的指针
  for(int k = 0; k < nc; k++) 
  {
    double * ppAik = ppAkk, eta = fabs(*ppAik);//ppAkk在k次迭代中指向A矩阵的k行k列
    for(int i = k + 1; i < nr; i++) //让eta保存k行到nr-2行第k列的最大绝对值的那个
    {
      double elt = fabs(*ppAik);//ppAik此时指向A(i-1,k)
      if (eta < elt) eta = elt;//eta为k行到nr-2行最大的那个绝对值
      ppAik += nc;
    }//出了这个循环后ppAik指向A(nr-1,k)

    if (eta == 0) //若第k列所有行中最大绝对值为0，则表明A为奇异阵，不合理，退出本函数
    {
      A1[k] = A2[k] = 0.0;
      cerr << "God damnit, A is singular, this shouldn't happen." << endl;
      return;
    } 
    /*
    householder算法的两步：
    对于k=1到n
    1.定义y=Ak:m,k并计算(m-k+1)维向量vk            w=y+sign(y1)*||y||*e1    vk=w/||w||
    2.用(m-k+1)*(m-k+1)维向量I-2*vk*vk.transpose()来乘以Ak:m,k:n      Ak:m,k:n := Ak:m,k:n - 2*vk*(vk.transpose()*Ak:m,k:n)  
    */
    else //若第k列所有行中最大绝对值不为0
    {
      double * ppAik = ppAkk, sum = 0.0, inv_eta = 1. / eta;//让ppAik重新指向第k行第k列的元素
      for(int i = k; i < nr; i++) //对于k行到nr-1行
      {
        *ppAik *= inv_eta;//对于i>=k A(i,k) = A(i,k)/eta  eta为第k列k到nr-2行的最大绝对值  此处更改了A矩阵的值,对于i从k到nr-1行的A(i,k),除以i从k到nr-2行中最大的一个A(i,k)的绝对值
        sum += *ppAik * *ppAik;
        ppAik += nc;//将指针指向第k列第i行的元素
      }//最终，sum为 sumof_i_from_k_to_nr-1(A(i,nc)/eta*A(i,nc)/eta)
      double sigma = sqrt(sum);//求向量Ak:m,k的模长，此时的向量已变为原来的1/eta倍
      if (*ppAkk < 0)//若A(k,k)<0,则sigma=-sigma即令sigma为负数，对应于householder变换中取 w=[ y1+sign(y1)*||y|| y2 ...  yn].transpose()
	      sigma = -sigma; //sigma即为sign(y1)*||y||，由于每个元素均被变为原来的1/eta倍，故而sigma=sign(y1)*||y||=sign(y1)*||Ak:m,k||/eta
      /*
      让ppAkk的值加上sigma，此时Ak:m,k向量已经变为w=[ y1+sign(y1)*||y|| y2 ...  yn].transpose()  此时W模长的平方为 ||w||.square=2*||y||*(||y||+|y1|) = 2*(w.transpose()*y)
      A1数组对应的第k列的值为sigma*A(k,k)  A1[k]=sigma*A(k,k) = sign(y1)*||y||*(y1+sign(y1)*||y||)= |y1|*||y||+||y||.square()=||w||.square()/2
      A2数组对应的第k列的值为-eta*sigma A2[k] = -sign(y1)*||Ak:m,k||
      经过变换后 第k列k到nr-1行应该变为   -sign(y1)*||y||*e1即为-sigma*e1
      */
      *ppAkk += sigma;
      A1[k] = sigma * *ppAkk;
      A2[k] = -eta * sigma;
      for(int j = k + 1; j < nc; j++) //对于k+1列到nc-1列 j=k+1:nc-1
      {
        /*
        在这个循环中，A(k:m,k)中存储的是wk,A1[k]中存储的是||wk||.square()/2
        */
        double * ppAik = ppAkk, sum = 0;
        /*
        出循环后ppAik指向的是A(nr-1,k) 出这个循环后，sum=Ak:m,k.transpose()*Ak:m,j
        */
        for(int i = k; i < nr; i++) //对于k行到nr-1行
        {
          sum += *ppAik * ppAik[j - k];//将最新的A(i,k)与A(i,j)相乘，并将i=k到nr-1行的这些值相加   ppAik指向的是A(i,k),所以ppAik[j - k]指向的是A(i,j)
          ppAik += nc;
        }
        /*
        此处sum=Ak:m,k.transpose()*Ak:m,j 即为k列k到m行组成的列向量的转置乘以j列k到m行组成的列向量，相当于householder算法第二步中的vk.transpose()*Ak:m,k:n不过此处k列中存储的不是vk而是wk
        由上方相关注释，A1[k]=||w||.square()/2
        tau = wk.transpose()*Ak:m,j/(||wk||.square()/2) = 2*wk.transpose()*Ak:m,j/||wk||.square()
        下方循环中，i从k到nr-1行 A(i,j) = A(i,j)-tau*A(i,k)
                                     =A(i,j)-2*wk.transpose()*Ak:m,j/||wk||.square()*A(i,k)         A(i,k)此时已经是wk了
                                     = A(i,j)-2*wk.transpose()*Ak:m,j*wk/||wk||.square()
                                     = A(i,j)-2*vk.transpose()*Ak:m,j*vk
                                     = A(i,j)-2*vk*(vk.transpose()*Ak:m,j)   即为householder算法第二步
        由于这里j=k+1:nc,故而下方的循环只改变了A(k:nr-1,k+1:nc-1)的值
        
        */
        double tau = sum / A1[k];
        ppAik = ppAkk;//将ppAik重新指向A(k, k)
        for(int i = k; i < nr; i++) //对于k行到nr-1行
        {
          ppAik[j - k] -= tau * *ppAik;//让A(i,j)=A(i,j)-tau*A(i,k)
          ppAik += nc;
        }
      }
    }
    ppAkk += nc + 1;//令ppAkk指向下一行下一列
  }
  //到这里已经完成了雅克比矩阵的QR分解
  /*
  对于Ax=b 其中A为m*n维，x为n*1维，b为m*1维
  A.transpose()*A*x=A.transpose()*b
  x=(A.transpose()*A).inverse()*A.transpose()*b 将A=QR代入 =(R.transpose()*Q.transpose()*Q*R).inverse()*R.transpose()*Q.transpose()*x
   = (R.transpose()*R).inverse()*R.transpose()*Q.transpose()*x = R.inverse()*Q.transpose()*x
  */
  // b <- Qt b
  double * ppAjj = pA, * pb = b->data.db;
  /*
  这个循环每循环一次代表H矩阵对b向量作用一次，根据上面注释中提到的原理， [ Q Q_es] = H1*H2*...*Hn 
  所以 [ Q Q_es].transpose()*b=Hn.transpose()*...*H2.transpose()*H1.transpose()*b
  v为单位向量   根据householder矩阵定义 H=I-2*v*v.transpose()  
  H.transpose()=(I-2*v*v.transpose()).transpose() = I.transpose()-2*(v*v.transpose()).transpose()
               =I-2*(v.transpose()).transpose()*v.transpose()
               =I-2*v*v.transpose() = H
  H.transpose()*H=H*H.transpose()=(I-2*v*v.transpose())*(I-2*v*v.transpose())
                 = I-2*v*v.transpose()-2*v*v.transpose()+4*v*v.transpose()*v*v.transpose()
                 = I-4*v*v.transpose()+4*v*(v.transpose()*v)*v.transpose()
                 = I-4*v*v.transpose()+4*v*1*v.transpose()
                 = I-4*v*v.transpose()+4*v*v.transpose()
                 = I
  */
  /*
  这个关于j的循环中，行数大于列数，相当于对对角线下方的元素进行操作
  */
  for(int j = 0; j < nc; j++) //ppAjj从A(0,0)指向A(nc-1, nc-1)
  {
    double * ppAij = ppAjj, tau = 0;
    /*
    此时取行坐标大于列坐标，由于在上面计算QR的过程中，每次处理第k行时，相当于对Ak:m,k求取了||wk||放入Ak:m,k位置，在householder矩阵作用时，只作用了Ak:m,k+1:n部分的分块矩阵，并没有作用
    在Ak:m,k向量上，所以此时Ak:m,k即A矩阵行坐标大于列坐标位置的元素中保存的仍为wk
    Ak-1,k:n中保存的即为上三角阵的非对角元素，上三角阵的对角元素为y向量（即原始的Ak:m,k）经过householder矩阵映射得到的列向量的第一个元素
    结论：
        Ak:m,k即A矩阵行坐标大于等于列坐标位置的元素中保存的仍为wk
        Ak-1,k:n中保存的即为上三角阵的非对角元素，上三角阵的对角元素为y向量（即原始的Ak:m,k）经过householder矩阵映射得到的列向量的第一个元素
    */
    /*
    下面两个循环求的是 H*b = (I-2*v*v.transpose())*b = b-2*v*v.transpose()*b  (v.transpose()*b是一个常数不是矩阵，所以可以提到前面)
                         = b-2*v.transpose()*b*v
    
    */
    for(int i = j; i < nr; i++)	
    {
      tau += *ppAij * pb[i];
      ppAij += nc;
    }
    //此时tau=wj.transpose()*b
    tau /= A1[j];//A1[j]=||wj||.square()/2  此时tau=2*vj.transpose()*b/||wj||
    ppAij = ppAjj;
    /*
    经过这次循环，对所有i相加的tau*A(i,j)=2*vj.transpose()*b/||wj||*wj=2*vj.transpose()*b*vj
    pb减这些项即为 b-2*vj.transpose()*b*vj 为H矩阵作用在b向量的结果 
    */
    for(int i = j; i < nr; i++) 
    {
      pb[i] -= tau * *ppAij;
      ppAij += nc;
    }
    ppAjj += nc + 1;
  }


  /*
  在进行QR分解的过程中，将每个列向量A(k:m,k)进行householder前保留了下来方便求Q.transpose()的时候根据Hk矩阵解Q矩阵，而y(对应此问题为A(k:m,k))进行householder变换后形式非常简单，只有第一个元素
  非0为-sign(y(0))*||y||*e1,其中e1为(m-k+1)维，
  在上面求QR计算的过程中，对于第k列，对i>=k的那些行进行 A(i,k)=A(i,k)/eta的缩放，记经过此步骤的A(k:m,k)为yk, 上面的sigma=sign(yk(0))*||yk||
  A2[k]=-eta*sigma=-eta*sign(yk(0))*||yk||
  对于A矩阵，其对角线上方元素经过一系列变换已经变为R矩阵元素，但是对角线上元素仍未变换，对角线元素即为每个yk经过householder变换的向量的第一维即为-sign(yk(0))*||yk||与A2[k]只差一个eta
  R矩阵费对角线上的元素，在对角线下方由于是上三角矩阵为0，对角线上方的A(j,k)(j<k)并没有受到除以eta的影响，对角线上的元素需要乘以eta进行恢复，即对角线上的元素为A2[k]
  */
  /*
  这里需要求X=R.inverse()*b
  先考虑对于A*x=b这个式子，若A为上三角矩阵
  | A(1,1)  A(1,2)  ...  A(1,n) |   |x(1)|   |b(1)|
  |   0     A(2,2)  ...  A(2,n) |   |x(2)|   |b(2)|
  |   0       0     ...   ...   | * |... | = |... |　
  |   0       0          A(n,n) |   |x(n)|   |b(n)|
  写为方程形式为
  A(1,1)*x(1) + A(1,2)*x(2) + ... + A(1,n)*x(n) = b(1)      (1)
  A(2,2)*x(2) + ... + A(2,n)*x(n) = b(2)                    (2)
  .............................................
  A(n,n)*x(n) = b(n)                                        (n)
  对于第k行方程为   A(k,k)*x(k) + A(k,k+1)*x(k+1) + ... + A(k,n)*x(n) = b(n)
  观察上述方程可得（n）式只有一个未知数x(n),利用该式本身即可求取x(n),(n-1)式具有未知数x(n-1)与x(n),此时已知从(n)式中求取的x(n),故而仅凭此式即可求得x(n-1),由此可知，在对上述方程组从下到上传输的
  过程中，(k)式有（n+1-k）个未知数x(k),x(k+1),...,x(n)，其中x(k+1),...,x(n)已由下方的各个方程求出，故而对于A(k,k)*x(k) + A(k,k+1)*x(k+1) + ... + A(k,n)*x(n) = b(n)可
  得到 x(k) = (b(n)-A(k,k+1)*x(k+1)-...-A(k,n)*x(n))/A(k,k) 
  对于此处的问题 X=R.inverse()*b,不妨将其转换为R*X=b的问题，这样就可以用上面的方程从下往上求取的过程来求X,避免了繁琐的求逆，有效简化了计算
  */
  // X = R-1 b
  double * pX = X->data.db;
  pX[nc - 1] = pb[nc - 1] / A2[nc - 1];//求出X的nc-1列的值
  /*
  在下面对于i的循环中，列数大于行数，相当于对对角线上方的元素进行操作，R为方阵，下方i指行，j指列
  */
  for(int i = nc - 2; i >= 0; i--) //求出X向量nc-2列到0列的值
  {
    double * ppAij = pA + i * nc + (i + 1), sum = 0;//此时ppAij指向的是A(i, i+1)
    /*
    这个for循环计算的是A(k,k+1)*x(k+1)+...+A(k,nc-1)*x(nc-1)
    */
    for(int j = i + 1; j < nc; j++) 
    {
      sum += *ppAij * pX[j];
      ppAij++;
    }
    pX[i] = (pb[i] - sum) / A2[i];//x(k) = (b(n)-A(k,k+1)*x(k+1)-...-A(k,n)*x(n))/A(k,k)
  }
}


//分别求四元数真值与四元数估计值以及旋转真值与旋转估计值的欧氏距离除以真值向量的模
void PnPsolver::relative_error(double & rot_err, double & transl_err,
			  const double Rtrue[3][3], const double ttrue[3],
			  const double Rest[3][3],  const double test[3])
{
  double qtrue[4], qest[4];

  mat_to_quat(Rtrue, qtrue);
  mat_to_quat(Rest, qest);
  /*可能是考虑到负的四元数与原四元数表示相同的元转，所以二者可能是加也可能是减  求的是两个q的欧式距离*/
  double rot_err1 = sqrt((qtrue[0] - qest[0]) * (qtrue[0] - qest[0]) +
			 (qtrue[1] - qest[1]) * (qtrue[1] - qest[1]) +
			 (qtrue[2] - qest[2]) * (qtrue[2] - qest[2]) +
			 (qtrue[3] - qest[3]) * (qtrue[3] - qest[3]) ) /
    sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

  double rot_err2 = sqrt((qtrue[0] + qest[0]) * (qtrue[0] + qest[0]) +
			 (qtrue[1] + qest[1]) * (qtrue[1] + qest[1]) +
			 (qtrue[2] + qest[2]) * (qtrue[2] + qest[2]) +
			 (qtrue[3] + qest[3]) * (qtrue[3] + qest[3]) ) /
    sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

  rot_err = min(rot_err1, rot_err2);
  /*求两个平移向量的欧氏距离*/
  transl_err =
    sqrt((ttrue[0] - test[0]) * (ttrue[0] - test[0]) +
	 (ttrue[1] - test[1]) * (ttrue[1] - test[1]) +
	 (ttrue[2] - test[2]) * (ttrue[2] - test[2])) /
    sqrt(ttrue[0] * ttrue[0] + ttrue[1] * ttrue[1] + ttrue[2] * ttrue[2]);
}

/*
将旋转矩阵转换为四元数
对于四元数来说，-q与+q代表同样的变换
单位四元数 q = q0+iqx+jqy+kqz 对应的正交矩阵为
    | (q0*q0+qx*qx-qy*qy-qz*qz)       2*(qx*qy-q0*qz)             2*(qx*qz+q0*qy)      |
R = |      2*(qy*qx+q0*qz)       (q0*q0-qx*qx+qy*qy-qz*qz)        2*(qy*qz-q0*qx)      |
    |      2*(qz*qx-q0*qy)            2*(qz*qy+q0*qx)        (q0*q0-qx*qx-qy*qy+qz*qz) |
1+r11+r22+r33 = 4*q0*q0           (1)                     
1+r11-r22-r33 = 4*qx*qx           (2)
1-r11+r22-r33 = 4*qy*qy           (3)
1-r11-r22+r33 = 4*qz*qz           (4)
对于非对角元素，有三个差的关系
r32-r23 = 4*q0*qx                 (5)  
r13-r31 = 4*q0*qy                 (6)  
r21-r12 = 4*q0*qz                 (7)  
对于非对角元素，还有三个和的关系
r21+r12 = 4*qx*qy                 (8)
r32+r23 = 4*qy*qz                 (9)
r13+r31 = 4*qz*qx                 (10)
利用（1）-（4）求出q0,qx,qy,qz中最大的那个q_max，再利用（5）-（10）求出q_max与其他三个值的积，获取其他三个值
这样先求出最大元素再利用最大元素求出剩下三个元素的方式可以有效地避免奇异值的出现
*/
void PnPsolver::mat_to_quat(const double R[3][3], double q[4])
{
  double tr = R[0][0] + R[1][1] + R[2][2];
  double n4;

  if (tr > 0.0f) 
  {
    q[0] = R[1][2] - R[2][1];//r23-r32 = -4*q0*qx
    q[1] = R[2][0] - R[0][2];//r31-r13 = -4*q0*qy
    q[2] = R[0][1] - R[1][0];//r12-r21 = -4*q0*qz
    q[3] = tr + 1.0f;//1+r11+r22+r33 = 4*q0*q0
    n4 = q[3];
  }
  /*
  r11>r22 即 q0*q0+qx*qx-qy*qy-qz*qz > q0*q0-qx*qx+qy*qy-qz*qz  得到 qx*qx > qy*qy
  r11>r33 即 q0*q0+qx*qx-qy*qy-qz*qz > q0*q0-qx*qx-qy*qy+qz*qz  得到 qx*qx > qz*qz
  */ 
  else if ( (R[0][0] > R[1][1]) && (R[0][0] > R[2][2]) ) 
  {
    q[0] = 1.0f + R[0][0] - R[1][1] - R[2][2];//1+r11-r22-r33 = 4*qx*qx
    q[1] = R[1][0] + R[0][1];// r21+r12 = 4*qx*qy
    q[2] = R[2][0] + R[0][2];// r31+r13 = 4*qx*qz
    q[3] = R[1][2] - R[2][1];// r23-r32 = -4*q0*qx
    n4 = q[0];
  } 
  //r22>r33 即 q0*q0-qx*qx+qy*qy-qz*qz > q0*q0-qx*qx-qy*qy+qz*qz 得到 qy*qy > qz*qz
  else if (R[1][1] > R[2][2]) 
  {
    q[0] = R[1][0] + R[0][1];// r21+r12 = 4*qx*qy
    q[1] = 1.0f + R[1][1] - R[0][0] - R[2][2];//1-r11+r22-r33 = 4*qy*qy
    q[2] = R[2][1] + R[1][2];// r32+r23 = 4*qy*qz
    q[3] = R[2][0] - R[0][2];// r31-r13 = -4*q0*qy
    n4 = q[1];
  } 
  else 
  {
    q[0] = R[2][0] + R[0][2];// r31+r13 = 4*qx*qz
    q[1] = R[2][1] + R[1][2];// r32+r23 = 4*qy*qz
    q[2] = 1.0f + R[2][2] - R[0][0] - R[1][1]; // 1-r11-r22+r33 = 4*qz*qz
    q[3] = R[0][1] - R[1][0];// r12-r21 = -4*q0*qz
    n4 = q[2];
  }
  double scale = 0.5f / double(sqrt(n4));//记q_max为q0,qx,qy,qz中最大的那个  n4=4*q_max*q_max  sqrt(n4) = 2*q_max  scale=0.5/2*q_max=1/4*q_max

  q[0] *= scale;
  q[1] *= scale;
  q[2] *= scale;
  q[3] *= scale;
}

} //namespace ORB_SLAM
