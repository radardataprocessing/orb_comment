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
一个矩阵的QR分解指将矩阵A分解为乘积A=QR，其中Q为正交矩阵，R为上三角矩阵，字母R表示右，指上三角矩阵，与QR分解类似的有QL，LQ，RQ分解，其中L表示左或者下三角分解。
一个3维Givens旋转是绕三个坐标轴中的一个轴进行旋转，三个Givens旋转为
     |1       |         | c     s|         |c  -s   |
Qx = |   c  -s|    Qy = |    1   |    Qz = |s   c   |          c = cos(theta)  s = sin(theta)
     |   s   c|         |-s     c|         |       1|
一个3*3矩阵A右侧乘以Qz的效果是保证A最后一列不变

用Givens旋转将一个3*3矩阵A进行RQ分解分为3步（乘法次序选择必须使已经为0的元素不受干扰，除了下面的三步也可选取其他组合给出相同的结果）
1）乘Qx使A32为0
2）乘Qy使A31为0，这一乘法不改变A的第二列，因此A32保持为0
3）乘Qz使A21为0，这一乘法使A的前两列由它们的线性组合代替，因此A31与A32保持不变
由以上旋转可得 AQxQyQz=R,其中R为上三角矩阵，因此A=RQz.transpose()Qy.transpose()Qx.transpose()，记作A=RQ
在更大维度的问题中，可用householder矩阵进行QR分解
*/

/*
1.若A为一个实对称矩阵，则A可以分解为A=UDU.transpose()，其中U为正交矩阵，D为实对角矩阵，因此一个实对称阵有实特征值，且其特征矢量两两相交
2.若S为一个实的反对称阵，S=UBU.transpose()，其中B为形如diag(a1Z,a2Z,...,amZ,0,...,0)的分块对角矩阵，Z=|0  1|.S的特征矢量均为纯虚数且奇数阶反对称矩阵必为奇异的
                                                                                              |-1 0|
*/


/*
叉乘与反对称矩阵的关系为 axb = [a]xb = (a.transpose()[b]x).transpose()
*/








#include "Initializer.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include "Optimizer.h"
#include "ORBmatcher.h"

#include<thread>

namespace ORB_SLAM2
{

Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations)
{
    mK = ReferenceFrame.mK.clone();

    mvKeys1 = ReferenceFrame.mvKeysUn;

    mSigma = sigma;
    mSigma2 = sigma*sigma;
    mMaxIterations = iterations;
}

bool Initializer::Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                             vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
{
    // Fill structures with current keypoints and matches with reference frame
    // Reference Frame: 1, Current Frame: 2
    mvKeys2 = CurrentFrame.mvKeysUn;//取出当前帧中的关键点

    mvMatches12.clear();
    mvMatches12.reserve(mvKeys2.size());//当前帧关键点数量
    mvbMatched1.resize(mvKeys1.size());//参考帧关键点数量
    for(size_t i=0, iend=vMatches12.size();i<iend; i++)
    {
        if(vMatches12[i]>=0)
        {
            mvMatches12.push_back(make_pair(i,vMatches12[i]));
            mvbMatched1[i]=true;
        }
        else
            mvbMatched1[i]=false;
    }

    const int N = mvMatches12.size();

    // Indices for minimum set selection
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }

    // Generate sets of 8 points for each RANSAC iteration
    mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));

    DUtils::Random::SeedRandOnce(0);

    for(int it=0; it<mMaxIterations; it++)
    {
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);//在两个参数的范围内取一个随机数
            int idx = vAvailableIndices[randi];

            mvSets[it][j] = idx;//为it次迭代选取8个种子点

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }

    // Launch threads to compute in parallel a fundamental matrix and a homography
    vector<bool> vbMatchesInliersH, vbMatchesInliersF;
    float SH, SF;
    cv::Mat H, F;

    thread threadH(&Initializer::FindHomography,this,ref(vbMatchesInliersH), ref(SH), ref(H));
    thread threadF(&Initializer::FindFundamental,this,ref(vbMatchesInliersF), ref(SF), ref(F));

    // Wait until both threads have finished
    threadH.join();
    threadF.join();

    // Compute ratio of scores
    float RH = SH/(SH+SF);

    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
    if(RH>0.40)
        return ReconstructH(vbMatchesInliersH,H,mK,R21,t21,vP3D,vbTriangulated,1.0,50);
    else //if(pF_HF>0.6)
        return ReconstructF(vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,1.0,50);

    return false;
}




/*
a = (a1, a2, a3)则a的反对称矩阵为
        0    -a3   a2
[a]x =  a3    0    -a1
       -a2    a1    0
[a]x为奇异阵
叉乘运算 axb = [a]xb = (a.transpose()[b]x).transpose()

本质矩阵的特性：
1.秩为2
2.仅依赖外部参数R与T
3.使用摄像机坐标系
E = [t]xR只有五个自由度，t与R均有三自由度，但是整体有一个尺度模糊
一个矩阵为本质矩阵当且仅当两个奇异值相等，第三个奇异值为0

对于任意一个m*n维的矩阵A,存在A = UDV'，这里U是一个m*m正交矩阵，D是一个m*n对角阵，V是一个n*n正交阵
D中对角元素为A的奇异值， U中列向量为A的左奇异向量， V中列向量为A的右奇异向量

对于方程Ax = b,其中A维度为mxn
1.m<n，方程个数小于未知数个数，无唯一解
2.m=n, 若A可逆，有唯一解
3.m>n, 方程个数多于未知数个数，若rank(A)=rank(A|b),则有解
*/

void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
{
    // Number of putative matches putative假定的、推定的
    const int N = mvMatches12.size();

    // Normalize coordinates
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2inv = T2.inv();

    /*
    |Pnx|   |sX  0  -meanX*sX|   |Px|
    |Pny| = |0  sY  -meanY*sY| * |Py|
    | 1 |   |0   0       1   |   |1 |
    */

    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat H21i, H12i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            int idx = mvSets[it][j];

            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }
        /*
        Pn2 = H21Pn1
        T2P2 = H21T1P1
        P2 = T2.inverseH21T1P1 = (T2.inverseH21T1)P1
        */

        cv::Mat Hn = ComputeH21(vPn1i,vPn2i);
        H21i = T2inv*Hn*T1;
        H12i = H21i.inv();

        currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);

        if(currentScore>score)
        {
            H21 = H21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}


void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
    // Number of putative matches
    const int N = vbMatchesInliers.size();

    // Normalize coordinates
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2t = T2.t();

    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat F21i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(int j=0; j<8; j++)
        {
            int idx = mvSets[it][j];

            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }

        cv::Mat Fn = ComputeF21(vPn1i,vPn2i);

        F21i = T2t*Fn*T1;

        currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

        if(currentScore>score)
        {
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}


cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    /*
    A的SVD分解 A=USV'
    A'A = (VS'U')USV' = VS'(U'U)SV' = V(S'S)V'
    AA' = USV'(VS'U') = US(V'V)S'U' = U(SS')U'
    对于Ax = 0 , 其解为S中最小奇异值对应的V的右奇异向量

    |u2|   |h11 h12 h13|   |u1|
    |v2| = |h21 h22 h23| * |u2|
    |1 |   |h31 h32 h33|   |1 |

    u2 = h11u1 + h12v1 + h13
    v2 = h21u1 + h22v1 + h23
    1 = h31u1 + h32v1 + h33

    (h31u1 + h32v1 + h33)u2 = h11u1 + h12v1 + h13
    (h31u1 + h32v1 + h33)v2 = h21u1 + h22v1 + h23

    [u1 v1 1 0 0 0 -u1u2 -v1u2 -u2][h11 h12 h13 h21 h22 h23 h31 h32 h33]' = 0
    [0 0 0 -u1 -v1 -1 u1v2 v1v2 v2][h11 h12 h13 h21 h22 h23 h31 h32 h33]' = 0
    */
    const int N = vP1.size();

    cv::Mat A(2*N,9,CV_32F);

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -u1;
        A.at<float>(2*i,4) = -v1;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2;

        A.at<float>(2*i+1,0) = u1;
        A.at<float>(2*i+1,1) = v1;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2;

    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    return vt.row(8).reshape(0, 3);
}

cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
{
    /*
                |f11  f12  f13|   |u1|
    [u2 v2 1] * |f21  f22  f23| * |v1| = 0
                |f31  f32  f33|   |1 |
    f11*u2*u1 + f21*v2*u1 + f31*u1 + f12*u2*v1 + f22*v2*v1 + f32*v1 + f13*u2 + f23*v2 + f33 = 0
    [u2*u1 u2*v1 u2 v2*u1 v2*v1 v2 u1 v1 1] * [f11 f12 f13 f21 f22 f23 f31 f32 f33]' = 0
    P2'*F21*p1 = 0  对其取逆得  （P2'*F21*P1)' = 0          P1'*F21'*P2 = 0   故F12 = F21'
    */
    const int N = vP1.size();

    cv::Mat A(N,9,CV_32F);

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<float>(2)=0;

    return  u*cv::Mat::diag(w)*vt;
}

float Initializer::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
{   
    const int N = mvMatches12.size();

    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);

    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);

    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in first image
        // x2in1 = H12*x2
        
        //利用h12对2向1进行映射
        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);//求2向1进行映射的点与原始点1的距离平方

        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)//若距离大于阈值则不计分，否则分数增加
            bIn = false;
        else
            score += th - chiSquare1;

        // Reprojection error in second image
        // x1in2 = H21*x1

        //利用h21对1向2进行映射
        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);//求1向2进行映射的点与原始点2的距离平方

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)//若距离大于阈值则不计分，否则分数增加
            bIn = false;
        else
            score += th - chiSquare2;

        if(bIn)//若距离较小，则记这组点为inlier，否则置为false
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
{
    const int N = mvMatches12.size();

    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 3.841;
    const float thScore = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);

    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in second image
        // l2=F21xP1 = [a2 b2 c2]'
        //图像中的点P1到另一幅图像的对极线l2为   l2 = F21*P1   对应的点P2在对极线上，则P2'*F21*P1=0       
        //点线距离公式  d=(ax+by+c)/sqrt(a*a＋b*b)

        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;//计算点1投影到图片2的极线

        const float num2 = a2*u2+b2*v2+c2;

        const float squareDist1 = num2*num2/(a2*a2+b2*b2);

        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)

        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;//计算点2投影到图片1的极线

        const float num1 = a1*u1+b1*v1+c1;

        const float squareDist2 = num1*num1/(a1*a1+b1*b1);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                            cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
                            //参数中minParallax = 1      minTriangulated = 50
{
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // Compute Essential Matrix from Fundamental Matrix
    cv::Mat E21 = K.t()*F21*K;

    cv::Mat R1, R2, t;

    // Recover the 4 motion hypotheses
    /*
    E = [t]xR  可得其等价E=SR，从而用S去表征t
    若E的SVD分解为U*diag(1,1,0)*V',则E=SR可以有以下两种表示
    S=UZU'  R=UWV'或R=UW'V'   t=U3=u(0,0,1)'
    */
    DecomposeE(E21,R1,R2,t);  

    cv::Mat t1=t;
    cv::Mat t2=-t;

    // Reconstruct with the 4 hyphoteses and check
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
    float parallax1,parallax2, parallax3, parallax4;

    /*
    int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
    */
    int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);

    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));//取出四个分数中最大的一个

    R21 = cv::Mat();
    t21 = cv::Mat();

    int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);//匹配的局内点的0.9倍与minTriangulated中的最大者，相当于做这个算法所期待的高分点的个数，若个数小于这个值，则认为效果不好不再继续进行

    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)//若分值大于一定比例的最大分值，则增加相似性
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    if(maxGood<nMinGood || nsimilar>1)//若各个分值相似性较高或者最大分值也较小
    {
        return false;
    }

    // If best reconstruction has enough parallax initialize
    if(maxGood==nGood1)
    {
        if(parallax1>minParallax)//若最大分值的视差角比较大，则将那组Rt作为求解的结果
        {
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood2)
    {
        if(parallax2>minParallax)
        {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood3)
    {
        if(parallax3>minParallax)
        {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood4)
    {
        if(parallax4>minParallax)
        {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    return false;
}


/*
H矩阵分解两种常用方法： 1)Faugeras  SVD-based decomposion   2)Zhang SVD-based decomposion   Homography 

- Faugeras et al, Motion and structure from motion in a piecewise planar environment. International Journal of Pattern Recognition and Artificial Intelligence, 1988.
- Deeper understanding of the homography decomposition for vision-based control
*/


/*
householder变换：给定一个向量a,令b表示a关于平面M(以w为法向量)的像  w=(a-b)/|a-b|  H(w) = 1-2*w*w.transpose()    则H(w)a = b

定理：假设u为一个n维单位向量，对任意n维向量x，存在householder阵使得Hx=au，其中a为x的二范数
*/

/*对K2.inverse*H*K1 = R+(1/d)t*n.transpose() 进行奇异值分解。利用分解出的对角阵的特性并记一些新矩阵为原先矩阵的变换，利用对角阵性质求出新矩阵的值*/
/*
相机1的投影矩阵为P1=K1[I|0]  (1)    P2=K2[R|t]  (2)    
平面方程为 aX+bY+cZ=d   平面单位法向量为n=(a b c).transpose()    （3）
对于平面上的点X (1/d)n.transpose()X = 1   (4) 
X2=RX1+t   （5）   
X1满足平面方程，故而(1/d)n.transpose()X1 = 1      
X2=(R+(1/d)*t*n.transpose())*X1  (6)
x2 = H*x1 (7)  K2*X2 = H*K1*X1   X2 = K2.inverse()*H*K1*X1  (8)    
对比（6）与（8）可得K2.inverse*H*K1 = R+(1/d)t*n.transpose()     
记A=d*R+t*n.transpose()=K2.inverse()*H*K1
对A进行奇异值分解   A=U*D*V.transpose() （U.transpose()*U=V.transpose()*V=I）  D=diag(d1,d2,d3) (d1>=d2>=d3)
D=U.transpose()*A*V = d*U.transpose()*R*V+U.transpose*t*n.transpose*V = d*U.transpose()*R*V+U.transpose*t*(V.transpose()*n).transpose()
记 s=detUdetV  s*s = 1
R' = s*U.transpose()*R*V          (9.1)              |d1  0  0 |
t' = U.transpose()*t              (9.2)              |0   d2 0 |
n' = V.transpose()*n              (9.3)          D = |0   0  d3|=d'*R'+t'*n'.transpose()  (10)
d' = s*d                          (9.4)
s = detU*detV                     (9.5)
空间基底为e1=(1 0 0).transpose()    e2=(0 1 0).transpose()   e3=(0 0 1).transpose()
记n' = (x1 x2 x3).transpose() = x1*e1 + x2*e2 + x3*e3
由（10）式可得  D = d'*R' + t'*x1*e1.transpose() + t'*x2*e2.transpose() + t'*x3*e3.transpose()
D*I = d'*R'*I + t'*x1*e1.transpose()*I + t'*x2*e2.transpose()*I + t'*x3*e3.transpose()*I
D*[e1 e2 e3] = d'*R'*[e1 e2 e3] + t'*x1*e1.transpose()*[e1 e2 e3] + t'*x2*e2.transpose()*[e1 e2 e3] + t'*x3*e3.transpose()*[e1 e2 e3]
[d1*e1 d2*e2 d3*e3] = [d'*R'*e1  d'*R'*e2  d'*R'*e3] + [t'*x1*e1.transpose()*e1  t'*x1*e1.transpose()*e2  t'*x1*e1.transpose()*e3] +
           [t'*x2*e2.transpose()*e1  t'*x2*e2.transpose()*e2  t'*x2*e2.transpose()*e3] + [t'*x3*e3.transpose()*e1  t'*x3*e3.transpose()*e2  t'*x3*e3.transpose()*e3]
即 [d1*e1 d2*e2 d3*e3] = [d'*R'*e1  d'*R'*e2  d'*R'*e3] + [t'*x1*e1.transpose()*e1  0  0] + [0  t'*x2*e2.transpose()*e2  0] + [0  0  t'*x3*e3.transpose()*e3]
可得
d1*e1 = d'*R'*e1 + t'*x1    (11.1)
d2*e2 = d'*R'*e2 + t'*x2    (11.2)
d3*e3 = d'*R'*e3 + t'*x3    (11.3)
由于n为单位法向量，V为旋转矩阵，n'也为单位向量，即 x1*x1+x2*x2+x3*x3=1
对（11）中的式子消去t'可得
d'*R'*(x2*e1-x1*e2) = d1*x2*e1-d2*x1*e2   (12.1)
d'*R'*(x3*e2-x2*e3) = d2*x3*e2-d3*x2*e3   (12.2)
d'*R'*(x1*e3-x3*e1) = d3*x1*e3-d1*x3*e1   (12.3)
由于R'为旋转矩阵，是否乘以他范数不变
d'*d'*x2*x2+d'*d'*x1*X1 = d1*d1*x2*x2+d2*d2*x1*x1    
d'*d'*x3*x3+d'*d'*x2*x2 = d2*d2*x3*x3+d3*d3*x2*x2
d'*d'*x1*X1+d'*d'*x3*x3 = d3*d3*x1*x1+d1*d1*x3*x3
即
(d'*d'-d2*d2)*x1*x1 + (d'*d'-d1*d1)*x2*x2 = 0    (13.1)
(d'*d'-d3*d3)*x2*x2 + (d'*d'-d2*d2)*x3*x3 = 0    (13.2)
(d'*d'-d1*d1)*x3*x3 + (d'*d'-d3*d3)*x1*x1 = 0    (13.3)
记为矩阵形式为
| d'*d'-d2*d2  d'*d'-d1*d1       0      | |x1*x1|
|      0       d'*d'-d3*d3  d'*d'-d2*d2 |*|x2*x2| = 0
| d'*d'-d3*d3       0       d'*d'-d1*d1 | |x3*x3|
由于x1*x1+x2*x2+x3*x3，故解不是全0，系数矩阵不满秩即行列式为0
(d'*d'-d1*d1)*(d'*d'-d2*d2)*(d'*d'-d3*d3) = 0
由于D奇异值的大小顺序    d1>=d2>=d3  d'只能取+-d2才可满足（13）中的三个式子
从而有三种情况
1. d1!=d2!=d3
2. d1 = d2 != d3 或者 d1!=d2=d3
3. d1 = d2 = d3
若d1!=d3，由（13）以及x1*x1+x2*x2+x3*x3=1可得
x1 = alpha1*sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3))    (14.1) alpha1=+-1
x2 = 0                                           (14.2)
x3 = alpha3*sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3))    (14.3) alpha3=+-1
接下来分两种情况讨论
一、当d' = d2 >0
1.d1 != d2 != d3
由于x2 = 0,带入（11.2）可得e2 = R'*e2所以R'是沿着e2轴旋转，有
     |cos(theta)  0  -sin(theta)|
R' = |    0       1       0     |     (15)-----------------------------------------------------------------------------------------------------------------------------------1.R'
     |sin(theta)  0  cos(theta) |
将R’带回（12.3）得
     |cos(theta)  0  -sin(theta)|    |0 |   |x3|     |  0  |   |d1*x3|
d2 * |    0       1       0     | * (|0 | - |0 | ) = |  0  | - |  0  |
     |sin(theta)  0  cos(theta) |    |x1|   |0 |     |d3*x1|   |  0  |
即
     |cos(theta)  0  -sin(theta)|   |-x3|   |-d1*x3|
d2 * |    0       1       0     | * |0  | = |  0   |   (16)
     |sin(theta)  0  cos(theta) |   |x1 |   |d3*x1 |
结合（14）可以解得
sin(theta) = (d1-d3)/d2*x1*x3 = alpha1*alpha3*sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2)
cos(theta) = (d1*x3*x3 - d3*x1*x1)/d2 = (d2*d2+d1*d3)/((d1+d3)*d2)
由于n'.transpose()*n' = 1,(10)式两侧同时乘以n'得 D*n' = d'*R'*n' + t'   t' = D*n' - d2*R'*n' 即
     |d1  0  0| |x1|      |cos(theta)  0  -sin(theta)| |x1|  |d1*x1|      |cos(theta)*x1 - sin(theta)*x3|
t' = |0  d2  0|*|0 | - d2*|    0       1       0     |*|0 | =|  0  | - d2*|              0              |   (17)
     |0   0 d3| |x3|      |sin(theta)  0  cos(theta) | |x3|  |d3*x3|      |sin(theta)*x1 + cos(theta)*x3|
将（16）带入（17）可得
               |x1|
t' = (d1 - d3)*|0 |    ---------------------------------------------------------------------------------------------------------------------------------------------------------1.t'
               |x3|
2.d1 = d2 != d3
由(14)可得x1=x2=0,x3=+-1,此时n' = (0 0 +-1).transpose()
由于x1=x2=0,带入（11.1） d1*e1 = d'*R'*e1 由于d1 = d2即 d2*e1=d2*R'*e1 带入(11.2)  d2*e2 = d2*R'*e2  故而R' = I        ********************************************************2.R'
由于n'.transpose()*n' = 1,(10)式两侧同时乘以n'得 D*n' = d'*R'*n' + t'   t' = D*n' - d2*R'*n'   t' = D*n' - d2*I*n'=(D-d2*I)*n'
     |d1 - d2     0        0   |        |0  0    0  |  | 0 |   |    0    |
t' = |   0     d2 - d2     0   | * n' = |0  0    0  | *| 0 | = |    0    |      *****************************************************************************************2.t'
     |   0        0     d3 - d2|        |0  0  d3-d2|  |+-1|   |+-(d3-d2)|
3.d1 != d2 = d3
由（14）可得x1=+-1, x2=x3=0,此时n' = (+-1 0 0).transpose()
由于x2=x3=0,带入（11.2）d2*e2=d2*R'*e2  带入（11.3）d3*e3=d2*R'*e3 即 d3*e3=d3*R'*e3  故而 R'=I       @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 3.R'
由于n'.transpose()*n' = 1,(10)式两侧同时乘以n'得 D*n' = d'*R'*n' + t'   t' = D*n' - d2*R'*n'   t' = D*n' - d2*I*n'=(D-d2*I)*n'
     |d1 - d2     0        0   |        |d1 - d2  0  0|  | 0 |   |+-(d3-d2)|
t' = |   0     d2 - d2     0   | * n' = |   0     0  0| *| 0 | = |    0    |     @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  3.t'
     |   0        0     d3 - d2|        |   0     0  0|  |+-1|   |    0    |
4.d1=d2=d3
(14)中的值没有意义，此时x1,x2,x3未定义，此时（12.1）变为d2*R'*(x2*e1-x1*e2) = d2*x2*e1-d2*x1*e2
（12.2)变为d2*R'*(x3*e2-x2*e3) = d2*x3*e2-d2*x2*e3 
（12.3)变为d2*R'*(x1*e3-x3*e1) = d2*x1*e3-d2*x3*e1  可知 R'=I      ###########################################################################################################   4.R'
(11.1)变为d1*e1 = d1*e1 + t'*x1 
(11.2)变为d2*e2 = d2*e2 + t'*x2    
(11.3)变为d3*e3 = d3*e3 + t'*x3    可知 t'=0                      ############################################################################################################   4.t'
二、当d'=-d2<0
5.d1 != d2 != d3
此时 d' = -d2, 带入（11.2）可得  -e2 = R'*e2 此时R'可表示为
     |cos(theta)  0  sin(theta)|
R' = |     0     -1       0    |      $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$    5.R'
     |sin(theta)  0 -cos(theta)|
将R'带回(12.3)
     |cos(theta)  0  sin(theta)|   |-x3|   |-d1*x3|
d' * |     0     -1       0    | * | 0 | = |   0  |     (18)
     |sin(theta)  0 -cos(theta)|   | x1|   | d3*x1|
结合（14）可以解得
sin(theta) = (d1+d3)/d2*x1*x3 = alpha1*alpha3*sqrt((d1*d1-d2*d2)(d2*d2-d3*d3))/((d1-d3)*d2)
cos(theta) = (d3*x1*x1-d1*x3*x3)/d2 = (d1*d3-d2*d2)/((d1-d3)*d2)
由于n'.transpose()*n' = 1,(10)式两侧同时乘以n'得 D*n' = d'*R'*n' + t'   t' = D*n' - d'*R'*n' 即
     |d1  0  0| |x1|      |cos(theta)  0   sin(theta)| |x1|   |d1*x1|    |cos(theta)*x1+sin(theta)*x3|                     |x1|
t' = |0  d2  0|*|0 | - d'*|    0      -1       0     |*|0 | = |  0  | -d*|             0             | 由（18） = （d1+d3）*| 0|   $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$    5.t'
     |0   0 d3| |x3|      |sin(theta)  0  -cos(theta)| |x3|   |d3*x3|    |sin(theta)*x1-cos(theta)*x3|                     |x3|
6.d1 = d2 != d3
由（14）可得 x1=x2=0,x3=+-1   最终
     |-1  0  0 |
R' = | 0 -1  0 |   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^     6.R'
     | 0  0  1 |
t' = (d3+d1)*n'    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^      6.t'
7.d1!=d2=d3
8.d1=d2=d3
由（10）式可得-d'*I = d'*R'+t'*n'.transpose()，对于与n'垂直的向量x有 -d'*x=d'*R'*x+t'*n'.transpose()*x 即R'x=-x  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     8.R'
由householder变换知  R' = -I+2*n'*n'.transpose()    带入(10)得  -d'=d'*(-I+2*n'*n'.transpose())+t'*n'.transpose()    t' = -2*d'*n'     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     8.t'
对于上述两个大类，2、3、4都是1的特例
对于d'>0与d'<0两个大类，每类中 alpha1=+-1 alpha3=+-1共有四种解，共有2*4=8组解
考虑A=d*R+t*n.transpose() = d*K2.transpose()*H*K1
由于A已经确定，s=detU*detV确定，d'=s*d确定，8解剩下4个，对于解得的平面单位法向量n为n1、-n1、n2、-n2，由于n.transpose()*X=d
考虑空间点坐标（X,Y,Z）与归一化相机坐标系坐标（x,y,1）的关系X/x=Y/y=Z  有n.transpose()*X1/(d*Z1) = n.transpose()*x1/d>0故而只有两个n满足条件

论文中提到，当观测点不是全部靠肩一个相机光心原理另一个相机光心时，只有一个解满足条件。若d'>0,对上述两个解（n1',t1'）和（n2',t2'）
有t2' = (d1-d3)*n1'  n2'=t1'/(d1-d3) 也就是两个解是相同的
函数代码中，在求出8个解之后，对每一个进行分析，查看是否在两个相机前方并统计重投影误差较小的点的个数，找出8个解中点数多的那个解作为最终解
*/
bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.     piecewise分段的    planar平面的，二维的
    // International Journal of Pattern Recognition and Artificial Intelligence, 1988

    /*由于特征点是在图像坐标系，将H矩阵由相机坐标系换算到图像坐标系
    x2 = H * x1     K*X2 = H * K*X1    X2 = invK*H*K * X1     X2 = A * X1                    E = K.transpose()*F*K
    */
    cv::Mat invK = K.inv();
    cv::Mat A = invK*H21*K;

    cv::Mat U,w,Vt,V;
    cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
    V=Vt.t();                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

    float s = cv::determinant(U)*cv::determinant(Vt);

    float d1 = w.at<float>(0);
    float d2 = w.at<float>(1);
    float d3 = w.at<float>(2);

    //SVD分解正常是特征值降序排列
    if(d1/d2<1.00001 || d2/d3<1.00001)
    {
        return false;
    }

    vector<cv::Mat> vR, vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));//x1绝对值
    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));//x3绝对值
    float x1[] = {aux1,aux1,-aux1,-aux1};
    float x3[] = {aux3,-aux3,aux3,-aux3};

    //case d'=d2
    //sin(theta) = (d1-d3)/d2*x1*x3 = alpha1*alpha3*sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2)
    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};//对应于四种情况（1） e1=e3=1 （2） e1=1 e3=-1  （3） e1=-1 e3=1  （4）  e1=e3=-1

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        /*
        |cos(theta)    0    -sin(theta)|
        |    0         1         0     |
        |sin(theta)    0     cos(theta)|
        */
        Rp.at<float>(0,0)=ctheta;
        Rp.at<float>(0,2)=-stheta[i];
        Rp.at<float>(2,0)=stheta[i];
        Rp.at<float>(2,2)=ctheta;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=-x3[i];
        tp*=d1-d3;

        /*
        R' = s*U.transpose()*R*V  
        t' = U.transpose()*t      
        n' = V.transpose()*n
        可得
        R = 1/s*U*R'*V.transpose()
        t = U*t'
        n = V*n'      
        */
        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    //case d'=-d2
    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=cphi;
        Rp.at<float>(0,2)=sphi[i];
        Rp.at<float>(1,1)=-1;
        Rp.at<float>(2,0)=sphi[i];
        Rp.at<float>(2,2)=-cphi;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=x3[i];
        tp*=d1+d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }


    int bestGood = 0;
    int secondBestGood = 0;    
    int bestSolutionIdx = -1;
    float bestParallax = -1;
    vector<cv::Point3f> bestP3D;
    vector<bool> bestTriangulated;

    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
    for(size_t i=0; i<8; i++)
    {
        float parallaxi;
        vector<cv::Point3f> vP3Di;
        vector<bool> vbTriangulatedi;
        int nGood = CheckRT(vR[i],vt[i],mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K,vP3Di, 4.0*mSigma2, vbTriangulatedi, parallaxi);

        if(nGood>bestGood)
        {
            secondBestGood = bestGood;
            bestGood = nGood;
            bestSolutionIdx = i;
            bestParallax = parallaxi;
            bestP3D = vP3Di;
            bestTriangulated = vbTriangulatedi;
        }
        else if(nGood>secondBestGood)
        {
            secondBestGood = nGood;
        }
    }


    if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
    {
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        vP3D = bestP3D;
        vbTriangulated = bestTriangulated;

        return true;
    }

    return false;
}

void Initializer::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    /*
    x=PX 为方便表示，下面写为x=PC,由叉乘的定义可得  x为3*1齐次坐标，P为3*4,C为4*1齐次坐标
    [x]x(PC) = 0  
    | 0  -1  y |   |P(row0)C|
    | 1   0  -x| * |P(row1)C| = 0
    |-y   x  0 |   |P(row2)C|
    -P(row1)C+yP(row2)C = 0   (1)
    P(row0)C-xP(row2)C = 0    (2)
    -yP(row0)C+xP(row1)C = 0  (3)
    其中（3）可以由（1）与（2）线性表示
    */
    cv::Mat A(4,4,CV_32F);

    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}

void Initializer::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
    //Z-score标准化方法  x' = (x - mean)/standard_deviation
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();

    vNormalizedPoints.resize(N);

    for(int i=0; i<N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    meanX = meanX/N;//所有点横纵坐标的均值
    meanY = meanY/N;

    float meanDevX = 0;
    float meanDevY = 0;

    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}


int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{
    /*
    假设某相机 f=35mm 最高分辨率4256*2832  传感器尺寸36*23.9mm   
    根据以上定义有 u0=4256/2=2128  v0=2832/2=1416  dx=36/4256  dy=23.9/2832  fx=f/dx=4137.8  fy=f/dy=4147.3
    */
    // Calibration parameters
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    vbGood = vector<bool>(vKeys1.size(),false);
    vP3D.resize(vKeys1.size());

    vector<float> vCosParallax;
    vCosParallax.reserve(vKeys1.size());

    // Camera 1 Projection Matrix K[I|0]设定相机1无旋转无平移
    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
    K.copyTo(P1.rowRange(0,3).colRange(0,3));

    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

    // Camera 2 Projection Matrix K[R|t]
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = K*P2;

    /*
    Pc2 = [R|t]Pworld_homogeous =    某点在c2参考系下的坐标为其在世界坐标系下的坐标被R,t作用的结果
    Pc1 = Pworld
    Pc2 = [R|t]Pc1_homogeous = RPc1+t
    c2相机在c2坐标系下的坐标为0，求其在c1坐标系下也是世界坐标系下的坐标为  0=RPc1+t    -t=RPc1   Pc1=-R.inverse*t   由于旋转矩阵为正交阵Pc1=-R.transpose*t
    */
    cv::Mat O2 = -R.t()*t;

    int nGood=0;

    for(size_t i=0, iend=vMatches12.size();i<iend;i++)//对于这对Rt,循环所有匹配对
    {
        if(!vbMatchesInliers[i])
            continue;

        const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
        const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
        cv::Mat p3dC1;

        Triangulate(kp1,kp2,P1,P2,p3dC1);

        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            vbGood[vMatches12[i].first]=false;
            continue;
        }

        // Check parallax   parallax视差
        cv::Mat normal1 = p3dC1 - O1;
        float dist1 = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;
        float dist2 = cv::norm(normal2);

        float cosParallax = normal1.dot(normal2)/(dist1*dist2);//检查两个相机分别连接三维空间点的射线夹角

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        //角度较小时，两个相机光心与feature点连线几乎平行，此时若平行线稍有偏差则可能会相交在成像平面之后
        if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R*p3dC1+t;

        if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check reprojection error in first image,将点从相机1的相机坐标系投影到图像平面坐标系
        float im1x, im1y;
        float invZ1 = 1.0/p3dC1.at<float>(2);
        im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<float>(1)*invZ1+cy;

        float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);//求投影点与原始图像平面上点的欧氏距离的平方

        if(squareError1>th2)
            continue;

        // Check reprojection error in second image，将点从相机2的相机坐标系投影到图像平面坐标系
        float im2x, im2y;
        float invZ2 = 1.0/p3dC2.at<float>(2);
        im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
        im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

        float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);//求投影点与原始图像平面上点的欧氏距离的平方

        if(squareError2>th2)
            continue;

        vCosParallax.push_back(cosParallax);
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));//存储在相机1相机坐标系下的坐标
        nGood++;

        if(cosParallax<0.99998)
            vbGood[vMatches12[i].first]=true;
    }

    if(nGood>0)
    {
        sort(vCosParallax.begin(),vCosParallax.end());

        size_t idx = min(50,int(vCosParallax.size()-1));
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;

    return nGood;
}

void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    /*
        |  0 1 0 |      | 0 -1 0 |
    Z = | -1 0 0 |  W = | 1  0 0 |
        |  0 0 0 |      | 0  0 1 |
    E = [t]xR  可得其等价E=SR，从而用S去表征t
    若E的SVD分解为U*diag(1,1,0)*V',则E=SR可以有以下两种表示 E=U*diag(1, 1, 0)*V' = U*Z*W*V' = U*Z*U'*U*W*V' = (U*Z*U')*(U*W*V')
    S=UZU'  R=UWV'或R=UW'V'   t=U3=u(0,0,1)'
    */
    cv::Mat u,w,vt;
    cv::SVD::compute(E,w,u,vt);

    u.col(2).copyTo(t);
    t=t/cv::norm(t);

    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    R1 = u*W*vt;
    if(cv::determinant(R1)<0)//行列式
        R1=-R1;

    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;
}

} //namespace ORB_SLAM
