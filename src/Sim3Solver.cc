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


#include "Sim3Solver.h"

#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>

#include "KeyFrame.h"
#include "ORBmatcher.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

namespace ORB_SLAM2
{

/*
mvpMapPoints1与mvpMapPoints2中存储的是关键帧1与关键帧2对应的地图点
mvX3Dc1与mvX3Dc2为各自关键帧对应的地图点从世界坐标系转换到各自关键帧坐标系的坐标
*/
Sim3Solver::Sim3Solver(KeyFrame *pKF1, KeyFrame *pKF2, const vector<MapPoint *> &vpMatched12, const bool bFixScale):
    mnIterations(0), mnBestInliers(0), mbFixScale(bFixScale)
{
    mpKF1 = pKF1;
    mpKF2 = pKF2;

    vector<MapPoint*> vpKeyFrameMP1 = pKF1->GetMapPointMatches();//取出关键帧1的地图点

    mN1 = vpMatched12.size();//存储的是关键帧1中的关键点与关键帧2关键点的对应关系，即1中每个关键点对应的2中地图点的指针，mN1为关键帧1中关键点个数

    mvpMapPoints1.reserve(mN1);
    mvpMapPoints2.reserve(mN1);
    mvpMatches12 = vpMatched12;
    mvnIndices1.reserve(mN1);
    mvX3Dc1.reserve(mN1);
    mvX3Dc2.reserve(mN1);

    cv::Mat Rcw1 = pKF1->GetRotation();//世界坐标系向相机坐标系的rt关系
    cv::Mat tcw1 = pKF1->GetTranslation();
    cv::Mat Rcw2 = pKF2->GetRotation();
    cv::Mat tcw2 = pKF2->GetTranslation();

    mvAllIndices.reserve(mN1);

    size_t idx=0;
    for(int i1=0; i1<mN1; i1++)//对于关键帧1中所有关键点
    {
        if(vpMatched12[i1])//若关键帧1该下标对应有关键帧2中的地图点
        {
            MapPoint* pMP1 = vpKeyFrameMP1[i1];//关键帧1地图点
            MapPoint* pMP2 = vpMatched12[i1];//关键帧2地图点

            if(!pMP1)//若关键帧1中下标对应的地图点为空，则继续处理下一个关键点
                continue;

            if(pMP1->isBad() || pMP2->isBad())//若关键帧1或者关键帧2的地图点为坏点，则继续处理下一个关键点
                continue;

            int indexKF1 = pMP1->GetIndexInKeyFrame(pKF1);//取出1,2地图点在各自关键帧中的下标
            int indexKF2 = pMP2->GetIndexInKeyFrame(pKF2);

            if(indexKF1<0 || indexKF2<0)//若1、2中某点对应下标为负，则继续处理下一个关键点
                continue;

            const cv::KeyPoint &kp1 = pKF1->mvKeysUn[indexKF1];//取出两个关键点
            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[indexKF2];

            const float sigmaSquare1 = pKF1->mvLevelSigma2[kp1.octave];//取出两个关键点对应金字塔的参数
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];

            mvnMaxError1.push_back(9.210*sigmaSquare1);//用金字塔层对应的参数计算1,2对应的误差
            mvnMaxError2.push_back(9.210*sigmaSquare2);

            mvpMapPoints1.push_back(pMP1);//将两个地图点分别放入对应的容器中
            mvpMapPoints2.push_back(pMP2);
            mvnIndices1.push_back(i1);//将关键帧1中对应的下标放入容器

            cv::Mat X3D1w = pMP1->GetWorldPos();//取出地图点1在世界坐标系的坐标
            mvX3Dc1.push_back(Rcw1*X3D1w+tcw1);//将地图点1在关键帧1相机坐标系下的坐标放入容器

            cv::Mat X3D2w = pMP2->GetWorldPos();
            mvX3Dc2.push_back(Rcw2*X3D2w+tcw2);//将地图点2在关键帧2相机坐标系下的坐标放入容器

            mvAllIndices.push_back(idx);
            idx++;
        }
    }//结束对关键帧1中关键点的遍历

    mK1 = pKF1->mK;
    mK2 = pKF2->mK;

    FromCameraToImage(mvX3Dc1,mvP1im1,mK1);//将坐标从相机坐标系转到图像坐标系存储到mvP1im1中
    FromCameraToImage(mvX3Dc2,mvP2im2,mK2);

    SetRansacParameters();
}


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
void Sim3Solver::SetRansacParameters(double probability, int minInliers, int maxIterations)
{
    mRansacProb = probability;
    mRansacMinInliers = minInliers;
    mRansacMaxIts = maxIterations;    

    N = mvpMapPoints1.size(); // number of correspondences

    mvbInliersi.resize(N);

    // Adjust Parameters according to number of correspondences
    float epsilon = (float)mRansacMinInliers/N;

    // Set RANSAC iterations according to probability, epsilon, and max iterations
    int nIterations;

    if(mRansacMinInliers==N)
        nIterations=1;
    else
        nIterations = ceil(log(1-mRansacProb)/log(1-pow(epsilon,3)));

    mRansacMaxIts = max(1,min(nIterations,mRansacMaxIts));// 1 <= nIterations <= mRansacMaxIts

    mnIterations = 0;
}

//对对应点对使用ransac法求sim3变换，如果最佳变换局内点个数大于阈值，返回那组变换，否则返回空矩阵
cv::Mat Sim3Solver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
{
    bNoMore = false;
    vbInliers = vector<bool>(mN1,false);//大小为关键帧1中关键点的数目，初始值为false
    nInliers=0;

    if(N<mRansacMinInliers)//若对应点个数小于最小局内点数，设置bNoMore为真，返回空矩阵
    {
        bNoMore = true;
        return cv::Mat();
    }

    vector<size_t> vAvailableIndices;//

    cv::Mat P3Dc1i(3,3,CV_32F);//关键帧1相机坐标系下的地图点
    cv::Mat P3Dc2i(3,3,CV_32F);//关键帧2相机坐标系下的地图点

    int nCurrentIterations = 0;
    while(mnIterations<mRansacMaxIts && nCurrentIterations<nIterations)
    {
        nCurrentIterations++;
        mnIterations++;

        vAvailableIndices = mvAllIndices;//存储的不是关键帧1中的关键点下标，而是关键帧1与关键帧2有对应地图点的数目，从0开始计数

        // Get min set of points
        for(short i = 0; i < 3; ++i)//从点对中取三对点
        {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);//随机取出一对点

            int idx = vAvailableIndices[randi];//mvAllIndices即vAvailableIndices中的下标即为存储的值

            mvX3Dc1[idx].copyTo(P3Dc1i.col(i));//矩阵中每一列代表一个点坐标，将对应地图点在每个关键帧相机坐标系下的坐标取出
            mvX3Dc2[idx].copyTo(P3Dc2i.col(i));

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();//将最后一个下标放到刚才取出的地方，然后丢弃最后一个下标，相当于将刚才取过的下标丢弃
        }

        ComputeSim3(P3Dc1i,P3Dc2i);//求出了2到1与1到2的变换 输入是对应地图点在各自关键帧相机坐标系下的坐标，求出1相机坐标系与2相机坐标系的sim3变换

        CheckInliers();//将点从1相机坐标系三维坐标利用sim3转换到2像素坐标系下，再从2相机坐标系三维坐标利用sim3转到1像素坐标系下，查看局内点个数

        if(mnInliersi>=mnBestInliers)//取出局内点最佳的变换
        {
            mvbBestInliers = mvbInliersi;
            mnBestInliers = mnInliersi;
            mBestT12 = mT12i.clone();
            mBestRotation = mR12i.clone();
            mBestTranslation = mt12i.clone();
            mBestScale = ms12i;

            if(mnInliersi>mRansacMinInliers)//如果局内点个数大于设定的最小数目
            {
                nInliers = mnInliersi;
                for(int i=0; i<N; i++)//N为存在对应的点的个数
                    if(mvbInliersi[i])
                        vbInliers[mvnIndices1[i]] = true;//mvnIndices1存储的是关键点在关键帧1中的下标，所以这里是对关键帧关键点的原始下标进行标记，记录了关键帧1的关键点中哪些是局内点
                return mBestT12;
            }
        }
    }

    if(mnIterations>=mRansacMaxIts)
        bNoMore=true;

    return cv::Mat();
}

cv::Mat Sim3Solver::find(vector<bool> &vbInliers12, int &nInliers)
{
    bool bFlag;
    return iterate(mRansacMaxIts,bFlag,vbInliers12,nInliers);
}

void Sim3Solver::ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C)
{
    cv::reduce(P,C,1,CV_REDUCE_SUM);
    C = C/P.cols;

    for(int i=0; i<P.cols; i++)
    {
        Pr.col(i)=P.col(i)-C;
    }
}



/*四元数的一些性质
四元数可以写成复数形式，包含三个虚部和一个实部
q = qo + iqx + jqy + kqz
ii=-1 jj=-1 kk=-1 ij=k jk=i ki=j ji=-k kj=-i ik=-j
记r = r0 + irx + jry + krz

1.两个四元数的积
可得 rq = (r0q0 - rxqx - ryqy - rzqz)
          + i(r0qx + rxq0 + ryqz - rzqy)
          + j(r0qy - rxqz + ryq0 + rzqx)
          + k(r0qz + rxqy - ryqx + rzq0)
rq != qr 形式相似但是有六项符号相反
两个四元数的积可以被表示为一个正交4*4矩阵和一个四维列向量的积
     |r0  -rx  -ry  -rz|
     |rx   r0  -rz   ry|
rq = |ry   rz   r0  -rx|q = rleft_matrix*q
     |rz  -ry   rx   r0|

     |r0  -rx  -ry  -rz|
     |rx   r0   rz  -ry|
qr = |ry  -rz   r0   rx|q = rright_matrix*q 
     |rz   ry  -rx   r0|

rleft_matrix与rright_matrix中每一行与每一列的平方均为 r0*r0 + rx*rx + ry*ry + rz*rz 即rr
2.两个四元数的点积
p(dot)q = p0q0 + pxqx + pyqy + pzqz
四元数模的平方为四元数与自身的点积
模为1的四元数是单位四元数
对一个四元数取共轭即对其虚部取反
q.conjugate() = q0 - iqx - jqy - kqz
共轭四元数的4*4矩阵为原始四元数4*4矩阵的转置，由于矩阵为正交阵。所以矩阵与其共轭的积为对角阵Q(Q.transpose)=q(dot)qI 其中I为4*4单位阵
四元数与其共轭的积为四元数与其自身的点积  q(q.conjugate) = q(dot)q  则非0四元数的逆为 q.inverse = 1/(q(dot)q)q.conjugate 单位四元数的逆即为单位四元数的共轭
3.四元数的性质
(qp)(dot)(qr)=(Qp)(dot)(Qr)=(Qp).transpose(Qr) = p.transposeQ.transposeQr = p.transpose(q(dot)q)Ir = (q(dot)q)p.transposeIr=(q(dot)q)p.transposer=(q(dot)q)(p(dot)r)
q(dot)q为常数，所以可总结为 (qp)(dot)(qr)=(q(dot)q)(p(dot)r) 当q为单位四元数时即为 p(dot)r
由上面性质可得 (pq)(dot)(pq)=(p(dot)p)(q(dot)q)   积的模=模的积
(pq)(dot)r = p(dot)(rq.conjugate)  推导过程为  (pq)(dot)r = (q_right_matrixp)(dot)r = (q_right_matrixp).transpose()r
=p.transpose()q_right_matrix.transpose()r = p(dot)(q_right_matrix.transpose()r) = p(dot)(rq.conjugate)
三维向量可以被表示为纯虚四元数，若 r=(x,y,z).transpose()，记为四元数形式 r=0+ix+jy+kz
此时r_right_matrix与r_left_matrix主对角线元素为0，是反对称阵，r_right_matrix.transpose()=-r_right_matrix  r_left_matrix.transpose() = -r_left_matrix
4.单位四元数与旋转
旋转不会改变向量的长度以及向量之间的夹角，因此旋转后点积不变;反射也保持点积但是改变叉积（右手法则变左手法则），
两个四元数被单位四元数乘之后，点积结果不变  若q为单位四元数，有(qp)(dot)(qr) = p(dot)r
由于一个单位四元数乘一个纯虚四元数通常不再是一个纯虚四元数，所以不能简单地用乘法代表旋转
可以用复合乘法表示旋转
r'=qrq.conjugate()
展开可得  qrq.conjugate=(q_left_matrixr)q.conjugate = q_right_matrix.transpose()(q_left_matrixr) = (q_right_matrix.transpose()q_left_matrix)r
q_right_matrix与q_left_matrix为与四元数q对应的4*4矩阵
                                          |q(dot)q                0                           0                           0            |
                                          |   0       (q0*q0+qx*qx-qy*qy-qz*qz)         2(qxqy-q0qz)                2(qxqz+q0qy)       |
q_right_matrix.transpose()q_left_matrix = |   0             2(qyqx+q0qz)          (q0*q0-qx*qx+qy*qy-qz*qz)         2(qyqz-q0qx)       |
                                          |   0             2(qzqx-q0qy)                2(qzqy+q0qx)          (q0*q0-qx*qx-qy*qy+qz*qz)|
可见若r为虚，r'也为虚，若q为单位阵，则q_left_matrix与q_right_matrix为正交阵，q(dot)q为1，上面矩阵的右下3*3子阵也是正交的，即为熟悉的旋转矩阵r'=Rr
 qrq.conjugate同样保持叉积不变，
(-q)r(-q.conjugate) = qrq.conjugate,可见-q与q代表同样的旋转
绕单位向量w=(wx, wy, wz).transpose()转theta角表示为单位四元数为  q=cos(theta/2)+sin(theta/2)(iwx+jwy+kwz) 虚部表示了旋转轴的方向，实部和虚部的模可以用来恢复角度大小
5.旋转的分解
考虑旋转 r'=qrq.conjugate，再实行第二次旋转p, r''=pr'p.conjugate() = p(qrq.conjugate())p.conjugate()
可以简单地证明 (q.conjugate()p.conjugate())=(pq)conjugate()
故而  r''=(pq)r(pq).conjugate()
整体旋转可以被单位四元数pq表示，
两个四元数的乘法比两个3*3矩阵乘法的算术操作少，而且计算也会损失精度，多次乘正交矩阵不能再保持矩阵的正交性，单位四元数多次相乘也不会继续是单位向量，但是
找到近似单位向量比找近似正交阵容易
*/
/*
对于两对对应点集rr与rl 有rr=sR(rl)+r0
求每组点集的中心点  rl_ave=1/n*(sumof_i_from_1_to_n(rl(i)))   rr_ave=1/n*(sumof_i_from_1_to_n(rr(i)))
将点集中的点转换到以平均坐标为中心，有 rl'(i)=rl(i)-rl_ave     rr'(i)=rr(i)-rr_ave
有 sumof_i_from_1_to_n(rl'(i))=0   sumof_i_from_1_to_n(rr'(i))=0
一对对应点对的误差为 
e(i) = rr(i)-sR(rl(i))-r0 = rr'(i)+rr_ave-sR(rl'(i)+rl_ave)-r0 = rr'(i)-sR(rl'(i))-(r0-rr_ave+sR(rl_ave)) = rr'(i)-sR(rl'(i))-r0'
对于点集中所有点对的误差
e_sum = sumof_i_from_1_to_n(||rr'(i)-sR(rl'(i))-r0'||*||rr'(i)-sR(rl'(i))-r0'||)
=sumof_i_from_1_to_n(||rr'(i)-sR(rl'(i))||*||rr'(i)-sR(rl'(i))||) - 2*r0'*sumof_i_from_1_to_n(rr'(i)-sR(rl'(i))) + n*sumof_i_from_1_to_n(||r0'||*||r0'||)
由于点坐标是相对于点集平均坐标的，上面三项中第二项含rr'(i)的和以及rl'(i)的和，为0
e_sum中第三项非负，当第三项为0时e_sum最小，即 r0=rr_ave-sR(rl_ave)  ---------------------------------------------------------------------------------------平移确定

--------尺度确定方法1（对称形式）---------
对于e_sum中第一项，旋转不改变向量模长，故而有||R(rl'(i))||*||R(rl'(i))|| = ||rl'(i)||*||rl'(i)||
由于只是求第一项最小值，故而每项除以sqrt(s)不改变最小值的取值点
sumof_i_from_1_to_n(||1/sqrt(s)*rr'(i)-sqrt(s)*R(rl'(i))||*||1/sqrt(s)*rr'(i)-sqrt(s)*R(rl'(i))||)
 = 1/s*sumof_i_from_1_to_n(||rr'(i)||*||rr'(i)||) -2*sumof_i_from_1_to_n(rr'(i)*R(rl'(i))) + s*sumof_i_from_1_to_n(||R(rl'(i))||*||R(rl'(i))||)
 = 1/s*sumof_i_from_1_to_n(||rr'(i)||*||rr'(i)||) -2*sumof_i_from_1_to_n(rr'(i)*R(rl'(i))) + s*sumof_i_from_1_to_n(||rl'(i)||*||rl'(i)||)
 记为1/s*Sr*Sr-2*D+s*Sl*Sl  (sqrt(s)*Sl-1/sqrt(s)*Sr)*(sqrt(s)*Sl-1/sqrt(s)*Sr)+2*(Sl*Sr-D)此式中，第二项与s无关，当第一项为0时最小，s=Sr/Sl
 s = sqrt(sumof_i_from_1_to_n(||rr'(i)||*||rr'(i)||)/sumof_i_from_1_to_n(||rl'(i)||*||rl'(i)||))--------------------------------------------------------尺度确定方法1
 这种方法求尺度无需知道旋转
--------尺度确定方法2---------
sumof_i_from_1_to_n(||rr'(i)-sR(rl'(i))||*||rr'(i)-sR(rl'(i))||)
 = sumof_i_from_1_to_n(||rr'(i)||*||rr'(i)||) - 2*s*sumof_i_from_1_to_n(rr'(i)*R(rl'(i))) + s*s*sumof_i_from_1_to_n(||R(rl'(i))||*||R(rl'(i))||)
 记为 Sr - 2s*D - s*s*Sl
 可转换为
 (s*sqrt(Sl) - D/sqrt(Sl))*(s*sqrt(Sl) - D/sqrt(Sl))+(Sl*Sr-D*D)/Sl
 当上式第一项为0时此式最小，即 s=D/Sl = sumof_i_from_1_to_n(rr'(i)*R(rl'(i)))/sumof_i_from_1_to_n(||R(rl'(i))||*||R(rl'(i))||)----------------------------------尺度确定方法2

 然后求旋转  (sqrt(s)*Sl-1/sqrt(s)*Sr)*(sqrt(s)*Sl-1/sqrt(s)*Sr)+2*(Sl*Sr-D)或者(s*sqrt(Sl) - D/sqrt(Sl))*(s*sqrt(Sl) - D/sqrt(Sl))+(Sl*Sr-D*D)/Sl都是D越大误差越小
 用四元数寻找sumof_i_from_1_to_n(rr'(i)*R(rl'(i)))的最小值
 利用四元数表示为 sumof_i_from_1_to_n((qrl'(i)q.conjugate())(dot)rr'(i))最大   由于四元数的性质(pq)(dot)r = p(dot)(rq.conjugate) 
 有sumof_i_from_1_to_n((qrl'(i)q.conjugate())(dot)rr'(i)) = sumof_i_from_1_to_n((qrl'(i))(dot)(rr'(i)q))
 记rl'(i) = (xl'(i), yl'(i), zl'(i)).transpose()   rr'(i) = (xr'(i), yr'(i), zr'(i)).transpose()  将二者视为只有虚部的四元数
           |  0      -xl'(i)   -yl'(i)   -zl'(i)|
           |xl'(i)      0       zl'(i)   -yl'(i)|
 qrl'(i) = |yl'(i)   -zl'(i)      0       xl'(i)|q = rl'_right_matrixq 
           |zl'(i)    yl'(i)   -xl'(i)      0   |

           |  0      -xr'(i)   -yr'(i)   -zr'(i)|
           |xr'(i)     0       -zr'(i)    yr'(i)|
 rr'(i)q = |yr'(i)    zr'(i)      0      -xr'(i)|q = rr'_left_matrixq
           |zr'(i)   -yr'(i)    xr'(i)      0   |
 rl'_right_matrix与rr'_left_matrix均反对称且正交，要最大化的和变为
 sumof_i_from_1_to_n(rr'_left_matrixq)(dot)(rl'_right_matrixq)
 = sumof_i_from_1_to_n(rr'_left_matrixq).transpose()(rl'_right_matrixq)
 = sumof_i_from_1_to_n(q.transpose()rr'_left_matrix.transpose()rl'_right_matrixq)
 = q.transpose()sumof_i_from_1_to_n(rr'_left_matrix.transpose()rl'_right_matrix)q
 记为 q.transpose()sumof_i_from_1_to_n(Ni)q = q.transpose()Nq 其中每个Ni均为对称阵，N也是对称阵
 引入矩阵M = sumof_i_from_1_to_n(rl'(i)rr'(i).transpose())
     |Sxx  Sxy  Sxz|
 M = |Syx  Syy  Syz|
     |Szx  Szy  Szz|
 其中， Sxx = sumof_i_from_1_to_n(xl'(i)xr'(i))  Sxy = sumof_i_from_1_to_n(xl'(i)yr'(i))
 可写出N关于M的形式为
     |(Sxx+Syy+Szz)     Syz-Szy        Szx-Sxz         Sxy-Syx    |
     |   Syz-Szy     (Sxx-Syy-Szz)     Sxy+Syx         Szx+Sxz    |
 N = |   Szx-Sxz        Sxy+Syx     (-Sxx+Syy-Szz)     Syz+Szy    |
     |   Sxy-Syx        Szx+Sxz        Syz+Szy      (-Sxx-Syy+Szz)|
 使q.transpose()Nq最大的单位四元数是N的最大特征值对应的特征向量 -----------------------------------------------------------------------------------------------------旋转确定方法
 这个方法不涉及近似，不进行迭代修正
*/

/*
对于一个矩阵N,求其特征值lambda的方法为  det(N - lambda*I) = 0
求某一特征值lambda(i)对应的特征向量e(i)的方法为  (N - lambda(i)*I)*e(i) = 0
*/
void Sim3Solver::ComputeSim3(cv::Mat &P1, cv::Mat &P2)
{
    // Custom implementation of:
    // Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

    // Step 1: Centroid and relative coordinates

    cv::Mat Pr1(P1.size(),P1.type()); // Relative coordinates to centroid (set 1)相对于点集中心的坐标
    cv::Mat Pr2(P2.size(),P2.type()); // Relative coordinates to centroid (set 2)
    cv::Mat O1(3,1,Pr1.type()); // Centroid of P1
    cv::Mat O2(3,1,Pr2.type()); // Centroid of P2点集的中心

    ComputeCentroid(P1,Pr1,O1);//P中存储原始点集，Pr中存储每个点减去点坐标均值，O中存储点坐标均值，P与Pr均有三行，点的个数列
    ComputeCentroid(P2,Pr2,O2);

    // Step 2: Compute M matrix

    cv::Mat M = Pr2*Pr1.t();//两个点集的互协方差    M = sumof_i_from_1_to_n(rl'(i)rr'(i).transpose()) 求的是l到r的变换

    // Step 3: Compute N matrix  

    double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

    cv::Mat N(4,4,P1.type());

    N11 = M.at<float>(0,0)+M.at<float>(1,1)+M.at<float>(2,2);
    N12 = M.at<float>(1,2)-M.at<float>(2,1);
    N13 = M.at<float>(2,0)-M.at<float>(0,2);
    N14 = M.at<float>(0,1)-M.at<float>(1,0);
    N22 = M.at<float>(0,0)-M.at<float>(1,1)-M.at<float>(2,2);
    N23 = M.at<float>(0,1)+M.at<float>(1,0);
    N24 = M.at<float>(2,0)+M.at<float>(0,2);
    N33 = -M.at<float>(0,0)+M.at<float>(1,1)-M.at<float>(2,2);
    N34 = M.at<float>(1,2)+M.at<float>(2,1);
    N44 = -M.at<float>(0,0)-M.at<float>(1,1)+M.at<float>(2,2);

    N = (cv::Mat_<float>(4,4) << N11, N12, N13, N14,
                                 N12, N22, N23, N24,
                                 N13, N23, N33, N34,
                                 N14, N24, N34, N44);


    // Step 4: Eigenvector of the highest eigenvalue

    cv::Mat eval, evec;

    cv::eigen(N,eval,evec); //evec[0] is the quaternion of the desired rotation evec[0]对应的是使q.transpose()Nq最大的四元数

    cv::Mat vec(1,3,evec.type());
    (evec.row(0).colRange(1,4)).copyTo(vec); //extract imaginary part of the quaternion (sin*axis) 提取四元数虚部

    // Rotation angle. sin is the norm of the imaginary part, cos is the real part
    // q=cos(theta/2)+sin(theta/2)(iwx+jwy+kwz)  vec中保存的是四元数虚部，其模为sin值；cos为实部  第一个参数为虚部的模，即为sin(theta/2)，第二个参数为cos(theta/2)
    double ang=atan2(norm(vec),evec.at<float>(0,0));  //求出的是旋转弧度的一半

    vec = 2*ang*vec/norm(vec); //Angle-axis representation. quaternion angle is the half   2*ang为旋转的弧度 vec/norm(vec)为旋转轴方向的单位向量

    mR12i.create(3,3,P1.type());

    /*
    处理三维旋转时，常用旋转矩阵描述，还可以用旋转向量描述旋转，旋转向量的长度等于绕轴逆时针旋转的弧度。旋转向量与旋转矩阵可以通过罗德里格斯（Rodrigues）变换进行转换
    theta = norm(r)
    r = r/theta
                                                               | 0    -rz   ry| 
    R = cos(theta)*I+(1-cos(theta))*r*r.transpose()+sin(theta)*| rz    0   -rx|
                                                               |-ry    rx   0 |
    反变换为
               | 0    -rz   ry|
    sin(theta)*| rz    0   -rx| = (R - R.transpose())/2
               |-ry    rx   0 |
    */
    cv::Rodrigues(vec,mR12i); // computes the rotation matrix from angle-axis  从旋转向量求旋转矩阵

    // Step 5: Rotate set 2

    cv::Mat P3 = mR12i*Pr2;//求旋转后的点集2

    // Step 6: Scale
    //若为单目，尺度没有可信度，需要求取
    if(!mbFixScale)//s=D/Sl = sumof_i_from_1_to_n(rr'(i)*R(rl'(i)))/sumof_i_from_1_to_n(||R(rl'(i))||*||R(rl'(i))||)
    {
        double nom = Pr1.dot(P3);//opencv中的dot既可以做向量点乘，还扩展到了矩阵中表示矩阵对应位相乘再相加
        cv::Mat aux_P3(P3.size(),P3.type());
        aux_P3=P3;
        cv::pow(P3,2,aux_P3);
        double den = 0;

        for(int i=0; i<aux_P3.rows; i++)
        {
            for(int j=0; j<aux_P3.cols; j++)
            {
                den+=aux_P3.at<float>(i,j);
            }
        }

        ms12i = nom/den;
    }
    else
        ms12i = 1.0f;

    // Step 7: Translation

    mt12i.create(1,3,P1.type());
    mt12i = O1 - ms12i*mR12i*O2;//求得平移向量

    // Step 8: Transformation

    // Step 8.1 T12
    /*mT12i为
    |sR t|
    | 0 0|
    */
    mT12i = cv::Mat::eye(4,4,P1.type());

    cv::Mat sR = ms12i*mR12i;

    sR.copyTo(mT12i.rowRange(0,3).colRange(0,3));
    mt12i.copyTo(mT12i.rowRange(0,3).col(3));

    // Step 8.2 T21
    /*
    P1 = s*R*P2 + t
    1/s*P1 = R*P2 + t/s
    R*P2 = 1/s*P1 - t/s
    P2 = 1/s*R.inverse()*P1 - t/s*R.inverse
    sR' = 1/s*R.inverse() = 1/s*R.transpose()
    t' = -t/s*R.inverse() = -t/s*R.transpose() = -t*sR'
    */

    mT21i = cv::Mat::eye(4,4,P1.type());

    cv::Mat sRinv = (1.0/ms12i)*mR12i.t();

    sRinv.copyTo(mT21i.rowRange(0,3).colRange(0,3));
    cv::Mat tinv = -sRinv*mt12i;
    tinv.copyTo(mT21i.rowRange(0,3).col(3));
}


void Sim3Solver::CheckInliers()
{
    vector<cv::Mat> vP1im2, vP2im1;
    Project(mvX3Dc2,vP2im1,mT12i,mK1);//将三维点从相机2的相机坐标系投影到相机1的图像坐标系
    Project(mvX3Dc1,vP1im2,mT21i,mK2);//将三维点从相机1的相机坐标系投影到相机2的图像坐标系

    mnInliersi=0;
    //mvP1im1, mvP2im2 为将三维点从自身的相机坐标系投影到自身的图像坐标系
    for(size_t i=0; i<mvP1im1.size(); i++)
    {
        cv::Mat dist1 = mvP1im1[i]-vP2im1[i];//相机1相机坐标系投影到图像坐标系与相机2相机坐标系投影到相机1图像坐标系的差值
        cv::Mat dist2 = vP1im2[i]-mvP2im2[i];

        const float err1 = dist1.dot(dist1);//欧式距离的平方
        const float err2 = dist2.dot(dist2);

        if(err1<mvnMaxError1[i] && err2<mvnMaxError2[i])//若两个距离均在阈值内，则标记这对点为局内点，将局内点个数加1，否则标记这对点为局外点
        {
            mvbInliersi[i]=true;
            mnInliersi++;
        }
        else
            mvbInliersi[i]=false;
    }
}


cv::Mat Sim3Solver::GetEstimatedRotation()
{
    return mBestRotation.clone();
}

cv::Mat Sim3Solver::GetEstimatedTranslation()
{
    return mBestTranslation.clone();
}

float Sim3Solver::GetEstimatedScale()
{
    return mBestScale;
}

/*
vP3Dw为一系列世界坐标系下的三维点
vP2D为上述三维点投影得到的二维像素坐标
Tcw为世界坐标系向三维坐标系的转换
K为相机内参
将世界坐标系下三维点利用内外参转化为像素坐标系下的二维点
*/
void Sim3Solver::Project(const vector<cv::Mat> &vP3Dw, vector<cv::Mat> &vP2D, cv::Mat Tcw, cv::Mat K)
{
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    const float &fx = K.at<float>(0,0);
    const float &fy = K.at<float>(1,1);
    const float &cx = K.at<float>(0,2);
    const float &cy = K.at<float>(1,2);

    vP2D.clear();
    vP2D.reserve(vP3Dw.size());

    for(size_t i=0, iend=vP3Dw.size(); i<iend; i++)
    {
        cv::Mat P3Dc = Rcw*vP3Dw[i]+tcw;
        const float invz = 1/(P3Dc.at<float>(2));
        const float x = P3Dc.at<float>(0)*invz;//投影到以主点为原点的像素坐标系下
        const float y = P3Dc.at<float>(1)*invz;

        vP2D.push_back((cv::Mat_<float>(2,1) << fx*x+cx, fy*y+cy));//从以主点为原点的像素坐标转到以左上角为原点的像素坐标系
    }
}

/*
vP3Dc为一系列相机坐标系下的三维点，大小为三维点个数
vP2D为一系列相机坐标系下二维点，大小为三维点个数
K为相机内参矩阵
将三维点投影到像素坐标系下的二维点
*/
void Sim3Solver::FromCameraToImage(const vector<cv::Mat> &vP3Dc, vector<cv::Mat> &vP2D, cv::Mat K)
{
    const float &fx = K.at<float>(0,0);
    const float &fy = K.at<float>(1,1);
    const float &cx = K.at<float>(0,2);
    const float &cy = K.at<float>(1,2);

    vP2D.clear();
    vP2D.reserve(vP3Dc.size());

    for(size_t i=0, iend=vP3Dc.size(); i<iend; i++)
    {
        const float invz = 1/(vP3Dc[i].at<float>(2));
        const float x = vP3Dc[i].at<float>(0)*invz;
        const float y = vP3Dc[i].at<float>(1)*invz;

        vP2D.push_back((cv::Mat_<float>(2,1) << fx*x+cx, fy*y+cy));
    }
}

} //namespace ORB_SLAM
