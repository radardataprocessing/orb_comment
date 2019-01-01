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

#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>

#include "Converter.h"

#include<mutex>




/*
关于图优化的知识：

当一个图中存在连接两个以上顶点的边时，称这个图为超图，slam问题可以表示为一个超图

假设一个带有n条边的图，其目标函数可以写为  min(sumof_k_from_1_to_n(ek(xk,zk).transpose()*Infok*ek(xk,zk))),信息矩阵Info是协方差矩阵的逆，是一个对称矩阵，它的每个元素Info(i,j)作为eiej的系
数，可以看成是我们对ei,ej这个误差相关性的一个预计。最简单的是把Info设置为对角矩阵，对角阵元素的大小表示对此项误差的重视程度。  这里的xk可以指一个顶点，两个顶点或多个顶点，取决于边的实际类型，更严谨的方式是
将其写为ek(zk,xk1,xk2,...),但是这样的写法太过复杂。由于zk已知，为了表示的简洁，将它写为ek(xk)的形式
总体优化问题变为n条边加权和的形式 minF(x)=sumof_k_from_1_to_n(ek(xk).transpose()*Infok*ek(xk))

如果将一条错误的边加入到图中，则图中有一条误差很大的边，优化算法试图调整这条边所连接的节点的估计值，使节点顺应这条边的错误约束，往往使算法专注于调整一个错误的值。于是就有了核函数，核函数保证每条边的误差不会
过大以至于忽略其他的边，具体的方式是将原先误差的二范数度量替换为一个增长得没有那么快的函数，同时考虑可导需要保持其光滑特性。因为它使整个优化过程更加鲁棒，故而称其为robust kernel(鲁棒核函数)
很多鲁棒核函数均为分段函数，在输入较大时给出线性的增长速率，例如cauchy核，huber核等。

huber函数定义| 1/2*a*a                        fabs(a)<thres
            | thres*(fabs(a)-1/2*thres)       otherwise

*/



namespace ORB_SLAM2
{


void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);
}


void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    g2o::SparseOptimizer optimizer;
    /*此处6,3代表6维的pose与3维的landmark表示*/
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // Set KeyFrame vertices 
    /*设置图优化问题的顶点，设置顶点的序号与位姿，将第一个关键帧的坐标固定*/
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();//相机位姿节点
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));//pKF->GetPose()返回的是4*4矩阵
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId==0);//若pKF->mnId==0则参数布尔值为真，即设置该点固定，若pKF->mnId!=0则参数布尔值为假，在优化的过程中不固定该点
        optimizer.addVertex(vSE3);
        if(pKF->mnId>maxKFid)//记录加入图中的关键帧的最大序号
            maxKFid=pKF->mnId;
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    /*对于地图点集合中的所有地图点，找到其所对应的关键帧及其在关键帧中的下标，若对应关键帧在优化图的顶点集合中，则添加边，对于是否是单目的情况，要选取不同的边类型*/
    for(size_t i=0; i<vpMP.size(); i++)//对于地图点集合中的所有地图点
    {
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();//特征点空间坐标节点
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        const int id = pMP->mnId+maxKFid+1;//图的节点中，地图点的下标在关键帧最大下标之后开始
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

       const map<KeyFrame*,size_t> observations = pMP->GetObservations();//observations存储观察到此点的关键帧以及该点在关键帧中的下标

        int nEdges = 0;
        //SET EDGES
        /*
        g2o的几种边包括
        1.EdgeSE3ProjectXYZ():BA中的重投影误差（3D-2D (u,v)误差），将地图点投影到相机坐标系下的相机平面
        2.EdgeSE3ProjectXYZOnlyPose():PoseEstimation中的重投影误差，将地图点投影到相机坐标系下的相机平面。优化变量只有pose，地图点位置固定，是一元边，双目中
          使用的是EdgeStereoSE3ProjectXYZOnlyPose()
        3.EdgeSim3():Sim3之间的相对误差，优化变量只有Sim3表示的pose，用于OptimizeEssentialGraph
        4.EdgeSim3ProjectXYZ():重投影误差。优化变量Sim3位姿与地图点，用于闭环检测中的OptimizeSim3
        */
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)//对于观察到该地图点的所有关键帧
        {

            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid)//若关键帧为坏帧或关键帧下标大于加入图中的关键帧的最大下标
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];//利用地图点在关键帧中的下标取出关键帧对应的关键点
            //单目或双目及rgbd情况下选取的边的类型不一样
            if(pKF->mvuRight[mit->second]<0)//单目情况
            {
                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;//取出像素点的横纵坐标

                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();//单目的边类型

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));//这里的id是地图点在图中的顶点集合中的序号
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));//pKF->mnId是关键帧在关键帧集合中的序号，也是在优化图中顶点集合中的序号
                e->setMeasurement(obs);//将地图点在关键帧中的像素坐标作为观测
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];//用关键点金字塔层数相关参数作为信息矩阵的权重
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                if(bRobust)//如果要增加鲁棒性
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);//将sqrt(5.99)作为huber核函数的delta
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                optimizer.addEdge(e);
            }
            else//双目或rgbd情况
            {
                Eigen::Matrix<double,3,1> obs;
                const float kp_ur = pKF->mvuRight[mit->second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;//取出左视图的横纵坐标以及右视图的横坐标

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();//双目的边类型

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;//焦距乘基线长

                optimizer.addEdge(e);
            }
        }

        if(nEdges==0)//若没有边连接这个地图点，则将这个顶点从优化中去除，且标记相关去除的bool为真
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data
    //将优化后的结果写回函数参数列表的两个向量中
    //Keyframes
    //如果nLoopKF==0，则用优化后的值设置关键帧的位姿；否则将优化后的值写入关键帧的mTcwGBA中，且将nLoopKF写入关键帧的mnBAGlobalForKF中
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        if(nLoopKF==0)
        {
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }
        else
        {
            pKF->mTcwGBA.create(4,4,CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            pKF->mnBAGlobalForKF = nLoopKF;
        }
    }

    //Points
    //若nLoopKF==0，则将优化后的点设置为地图点在世界坐标系下的坐标，并更新其方向与深度；否则将优化后的点写入地图点的mPosGBA中且将nLoopKF写入地图点的mnBAGlobalForKF中
    for(size_t i=0; i<vpMP.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];

        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));

        if(nLoopKF==0)
        {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA.create(3,1,CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }

}

/*参数是一个普通的帧*/
int Optimizer::PoseOptimization(Frame *pFrame)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    // Set Frame vertex  将这个帧作为图的第一个顶点给入优化问题中，在优化过程中不固定这个帧
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices 帧中关键点的数量
    const int N = pFrame->N;

    /*单目
    这里边用的是onlypose类型的边*/
    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    /*双目*/
    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);

    //每个有效地图点向优化问题增加一条一元边，边只连接了函数参数中的帧，不连接地图点，可以理解为只涉及优化帧位姿，不改变地图点
    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        for(int i=0; i<N; i++)
        {
            MapPoint* pMP = pFrame->mvpMapPoints[i];
            if(pMP)
            {
                // Monocular observation
                if(pFrame->mvuRight[i]<0)
                {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double,2,1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];//利用关键点金字塔相关参数来加权信息矩阵
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    cv::Mat Xw = pMP->GetWorldPos();
                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);//增加一条边
                    vnIndexEdgeMono.push_back(i);//将地图点下标放入向量中
                }
                else  // Stereo observation
                {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    //SET EDGE
                    Eigen::Matrix<double,3,1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    const float &kp_ur = pFrame->mvuRight[i];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaStereo);

                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    e->bf = pFrame->mbf;
                    cv::Mat Xw = pMP->GetWorldPos();
                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }
            }

        }
    }


    if(nInitialCorrespondences<3)//如果有效边数目太少，则返回0
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    /*进行4次优化，每次优化后将观测分为局内点与局外点，在下次优化中局外点不再被包含，但是最终它们可能重新被归类为局内点*/
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};    

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {

        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);//给定迭代次数 进行优化

        nBad=0;
        /*

        */
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)//对于向量中的所有边 单目情况
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];//取出图中的那条边

            const size_t idx = vnIndexEdgeMono[i];//取出这个地图点在帧中对应的下标

            if(pFrame->mvbOutlier[idx])//如果这个地图点被标记为局外点
            {
                e->computeError();//计算观测值与重投影值的差值，求出了边的数据成员_error
            }
            /*chi2为 ek(xk).transpose()*Infok*ek(xk) 即为 _error.dot(information()*_error)  _error为向量，经过这个计算变为一个标量了
            chi2只有在调用了computeError后调用才有效
            */
            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])//若这个误差值大于预设的阈值，则标记这个点为局外点，设置其level为1，将坏点的统计次数加1
            {                
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);//最后一次优化时不再使用核函数来抑制野值
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)//对于向量中所有的边，双目情况
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];//取出一条边，这条边对应着一个地图点

            const size_t idx = vnIndexEdgeStereo[i];//取出地图点对应的在该帧中的下标

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {                
                e->setLevel(0);
                pFrame->mvbOutlier[idx]=false;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        if(optimizer.edges().size()<10)//若边数少于10个，则不再进行下次迭代
            break;
    }    

    // Recover optimized pose and return number of inliers 取出优化后的位姿赋给这一帧
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);
    /*
    nInitialCorrespondences为四次迭代之前向图中加入的边，即有效的此帧观测到的地图点；nBad为误差过大的点数；二者相减函数的返回值即为局内点个数
    */
    return nInitialCorrespondences-nBad;
}

void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap)
{    
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);//将当前关键帧放入局部关键帧链表中
    pKF->mnBALocalForKF = pKF->mnId;//记当前关键帧的mnBALocalForKF为当前关键帧序号

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();//取出经过排序的共视关键帧
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;//记该共视关键帧的mnBALocalForKF为当前关键帧序号
        if(!pKFi->isBad())//若共视关键帧不是坏帧，则将该共视关键帧放入局部关键帧链表中
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    list<MapPoint*> lLocalMapPoints;
    /*
    对于局部关键帧链表中的所有关键帧，取出该关键帧中所有地图点，若地图点非空且不是坏点，将其mnBALocalForKF标记为当前关键帧的序号，且将该地图点放入局部地图点链表中
    */
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();//取出与关键帧中关键点对应的地图点
        /*对于该关键帧对应的所有地图点，若地图点非空且不是坏点，将其mnBALocalForKF标记为当前关键帧的序号，且将该地图点放入局部地图点链表中*/
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes 看到了局部地图点但是不是局部关键帧
    list<KeyFrame*> lFixedCameras;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,size_t> observations = (*lit)->GetObservations();//观测到该地图点的关键帧，以及地图点在关键帧中的下标
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;
            /*若观察到地图点的关键帧的mnBALocalForKF与mnBAFixedForKF不是当前帧的序号，则将其mnBAFixedForKF置为当前关键帧序号，若观察到地图点的关键帧不是坏帧，则将其放入lFixedCameras链表中*/
            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {                
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // Set Local KeyFrame vertices 设置局部关键帧（当前关键帧的共视帧）顶点
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId==0);//设置序号为0的关键帧固定
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)//记录图顶点中最大的关键帧序号
            maxKFid=pKFi->mnId;
    }

    // Set Fixed KeyFrame vertices 固定关键帧顶点，也观察到了地图点，但是不是当前关键帧的共视帧
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);//这些帧全部设置为固定帧
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)//记录图顶点中最大的关键帧序号，包括局部关键帧与固定帧
            maxKFid=pKFi->mnId;
    }

    // Set MapPoint vertices 图中的边数预计为（局部关键帧数量+固定帧数量）*局部地图点数量
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;//单目边
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;//单目关键帧边
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;//单目地图点边
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;//双目边
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;//双目关键帧边
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;//双目地图点边
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);
    /*对于所有局部地图点*/
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId+maxKFid+1;//点在优化图中的顶点下标从关键帧最大下标加一开始记
        vPoint->setId(id);
        vPoint->setMarginalized(true);//地图点可以被消元
        optimizer.addVertex(vPoint);//向优化图中添加这个地图点顶点

        const map<KeyFrame*,size_t> observations = pMP->GetObservations();//取出观察到地图点的关键帧及地图点在该帧的下标

        //Set edges
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad())
            {                
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];//取出关键帧中的关键点

                // Monocular observation单目
                if(pKFi->mvuRight[mit->second]<0)
                {
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;//将关键点坐标作为观测值

                    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));//地图点在优化图顶点中的下标
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));//观测到地图点的关键帧在优化图顶点中的下标
                    e->setMeasurement(obs);//将关键点坐标作为观测给入边
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];//用关键点所在金字塔层数相关参数为信息矩阵加权
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);//设置鲁棒核函数及核函数参数

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);//向图中加入边
                    vpEdgesMono.push_back(e);//将边放入相应向量中
                    vpEdgeKFMono.push_back(pKFi);//记录这条边的关键帧
                    vpMapPointEdgeMono.push_back(pMP);//记录这条边的地图点
                }//结束单目情况的处理
                else // Stereo observation双目
                {
                    Eigen::Matrix<double,3,1> obs;
                    const float kp_ur = pKFi->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;//关键点在左图横纵坐标以及右图横坐标

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();//加入一条双目边

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));//将当前地图点作为边的一个顶点
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));//将对应关键帧作为边的另一个顶点
                    e->setMeasurement(obs);//将关键点坐标作为观测给入
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];//将关键点所在金字塔层参数取出并用其对信息矩阵进行加权
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;//focla_length*baseline

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }//结束双目情况的处理
            }
        }
    }

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore= true;

    if(pbStopFlag)
        if(*pbStopFlag)
            bDoMore = false;

    if(bDoMore)//如果继续处理
    {

        // Check inlier observations
        /*对于所有边，取出地图点查看，若地图点是坏点则继续处理下一条边，若边的误差较大或不是正深度，则设置其level为1；对于此边不再设置核函数*/
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            MapPoint* pMP = vpMapPointEdgeStereo[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>7.815 || !e->isDepthPositive())
            {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
        }

        // Optimize again without the outliers 去除局外点后再次优化

        optimizer.initializeOptimization(0);
        optimizer.optimize(10);

    }

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

    // Check inlier observations 
    /*对于所有单目边，若边所连接的地图点为坏点，继续处理下一条边，若边的误差太大或深度不为正，则同时取出边所连接的关键帧，将边连接的关键帧与地图点同时放入vToErase，此处不再对优化器
    做任何更改，只是将地图点与关键帧记录下来*/      
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if(!vToErase.empty())
    {
        /*对于vToErase中存储的地图点与关键帧的对，从关键帧中去除该地图点，同时从地图点观测中去除该关键帧*/
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data

    //Keyframes  对于所有关键帧，利用序号将其从优化器的顶点中取出，利用优化器的估计重置其位姿
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    //Points 对于所有地图点，利用其序号加关键帧最大序号将其从优化器的顶点取出，利用优化器的估计重置其位置，并更新方向与深度
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }
}


void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
{
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);//是否输出log
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();//7维的sim3
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);
    //从地图中取出所有关键帧与地图点
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();
    //设置最大关键帧序号
    const unsigned int nMaxKFid = pMap->GetMaxKFid();
    //用关键帧个数初始化向量
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    const int minFeat = 100;

    // Set KeyFrame vertices  
    // 对于地图中的所有关键帧，若不是坏帧，则将其设置为优化问题的顶点，若此帧在CorrectedSim3这个map中，则从map取出Sim3作为其位姿，否则读出关键帧rt并假设尺度为1来作为其Sim3
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->mnId;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);//在关键帧与位姿的map中找到这个关键帧

        if(it!=CorrectedSim3.end())//若在map中找到了这个关键帧，将向量位姿以及图顶点的位姿设为map中取出的位姿
        {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else//若在CorrectedSim3这个map中没有找到这个关键帧，则从关键帧中取出旋转平移并将尺度设为1来构造sim3，，并用其设置向量位姿以及图顶点位姿
        {
            Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw,tcw,1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        if(pKF==pLoopKF)//如果这个关键帧就是回环关键帧，则设置这个顶点为固定
            VSim3->setFixed(true);

        VSim3->setId(nIDi);//设置顶点序号为关键帧的序号
        VSim3->setMarginalized(false);//不可消元
        VSim3->_fix_scale = bFixScale;//bFixScale为本函数的输入参数

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;//将该顶点放入优化问题中并放入相关向量中
    }


    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    // Set Loop edges 设置回环边 一个关键帧与一个set的关键帧的map
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;//目标关键帧
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*> &spConnections = mit->second;//与目标关键帧连接的关键帧
        const g2o::Sim3 Siw = vScw[nIDi];//取出目标关键帧的变换矩阵
        const g2o::Sim3 Swi = Siw.inverse();

        /*对于与目标关键帧连接的所有关键帧*/
        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;//取出连接关键帧的序号
            /*若连接关键帧不是pCurKF或pLoopKF且与目标关键帧的权重小于阈值，则跳过此帧继续处理下一个连接关键帧*/
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;
            //Sjw为连接关键帧的变换矩阵，Siw为目标关键帧的变换矩阵
            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sji = Sjw * Swi;//求出两个关键帧之间的相对变换

            //设置一条边连接两个关键帧，将上方求出的两个关键帧之间的相对变换作为观测给入边
            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;//将单位阵作为信息矩阵

            optimizer.addEdge(e);//将这条边给入优化问题

            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));//将两个关键帧序号小者在前大者在后放入pair，然后将pair插入sInsertedEdges中
        }
    }

    // Set normal edges 设置普通边
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if(iti!=NonCorrectedSim3.end())//若当前关键帧在NonCorrectedSim3这个map中，取出map中的位姿作为当前关键帧位姿
            Swi = (iti->second).inverse();
        else
            Swi = vScw[nIDi].inverse();//若当前关键帧不在map中，则取出vScw中的位姿作为当前关键帧位姿

        KeyFrame* pParentKF = pKF->GetParent();//取出与当前帧拥有最多共视点的关键帧

        // Spanning tree edge
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if(itj!=NonCorrectedSim3.end())//若父关键帧在NonCorrectedSim3这个map中，取出map中的位姿作为当前关键帧位姿
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];//若父关键帧不在map中，则取出vScw中的位姿作为当前关键帧位姿

            g2o::Sim3 Sji = Sjw * Swi;//求出两个关键帧的相对位姿

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;//将7*7单位阵作为信息矩阵
            optimizer.addEdge(e);//向优化中添加这条边
        }

        // Loop edges
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();//取出关键帧的回环边
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(pLKF->mnId<pKF->mnId)//若回环关键帧序号小于当前关键帧序号
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if(itl!=NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));//回环关键帧序号
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));//当前关键帧序号
                el->setMeasurement(Sli);//将两帧位姿差作为观测给入边
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);//取出权重大于特定阈值的一系列关键帧
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)//对于选取的高权重共视帧
        {
            KeyFrame* pKFn = *vit;
            /*若高权重共视帧非空，且不是当前关键帧的父帧或子帧，也不在当前关键帧的回环边中，不是坏帧，且序号小于当前关键帧的序号*/
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    /*若这对关键帧已经出现在sInsertedEdges中，则继续处理下一个高权重共视帧*/
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if(itn!=NonCorrectedSim3.end())//若高权重共视帧在NonCorrectedSim3中，则从NonCorrectedSim3中取值作为其Sim3
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];//若高权重共视帧不在NonCorrectedSim3中，则从vScw中取值作为其Sim3

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    /*将高权重共视帧与当前关键帧作为边的顶点，将两帧相对位姿作为边的观测，将边加入优化问题中*/
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }
    }//结束对地图中所有关键帧的循环

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(size_t i=0;i<vpKFs.size();i++)//对于地图中所有关键帧
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));//从优化器中取出该关键帧对应顶点
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();//.rotation()取出的是四元数，将其变化为旋转矩阵
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *=(1./s); //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR,eigt);

        pKFi->SetPose(Tiw);//用优化后的顶点设置关键帧的位姿
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    /*对于地图中的所有地图点，变换到未优化的参考帧位姿下，然后用优化的位姿变换回来*/
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;
        if(pMP->mnCorrectedByKF==pCurKF->mnId)//如果地图点的mnCorrectedByKF参数为函数参数pCurKF的序号，则取nIDr为地图点的mnCorrectedReference
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else//否则取出地图点的参考关键帧，取nIDr为参考关键帧的序号
        {
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }


        g2o::Sim3 Srw = vScw[nIDr];//取序号nIDr的vScw值为参考坐标
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];//取序号nIDr的vCorrectedSwc值为修正坐标

        cv::Mat P3Dw = pMP->GetWorldPos();//取出地图点世界坐标
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);//取出地图点世界坐标
        //先用世界坐标系到参考帧变换前的转换关系将地图点转到相对参考帧转换前的坐标，再用参考帧转换后的坐标将地图点转回世界坐标系
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        /*用修正后的坐标设置地图点的坐标，并更新地图点的方向与深度*/
        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth();
    }
}


/*

*/
int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();

    // Set Sim3 vertex  设置sim3顶点
    g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap();    
    vSim3->_fix_scale=bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->_principle_point1[0] = K1.at<float>(0,2);
    vSim3->_principle_point1[1] = K1.at<float>(1,2);
    vSim3->_focal_length1[0] = K1.at<float>(0,0);
    vSim3->_focal_length1[1] = K1.at<float>(1,1);
    vSim3->_principle_point2[0] = K2.at<float>(0,2);
    vSim3->_principle_point2[1] = K2.at<float>(1,2);
    vSim3->_focal_length2[0] = K2.at<float>(0,0);
    vSim3->_focal_length2[1] = K2.at<float>(1,1);
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();//vpMatches1为函数参数
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;
    vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;
    vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);

    const float deltaHuber = sqrt(th2);//th2为函数参数

    int nCorrespondences = 0;

    for(int i=0; i<N; i++)//对于函数参数向量中的所有地图点指针
    {
        //若为空指针则继续处理下一个指针
        if(!vpMatches1[i])
            continue;

        MapPoint* pMP1 = vpMapPoints1[i];//取出第一个关键帧中第i个地图点
        MapPoint* pMP2 = vpMatches1[i];//取出函数参数向量中第i个地图点

        const int id1 = 2*i+1;
        const int id2 = 2*(i+1);

        const int i2 = pMP2->GetIndexInKeyFrame(pKF2);//找到pMP2地图点在关键帧2中的下标

        //若两个地图点指针均不为空
        if(pMP1 && pMP2)
        {
            if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
            {
                //向优化图中增加顶点
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D1w = pMP1->GetWorldPos();
                cv::Mat P3D1c = R1w*P3D1w + t1w;//将地图点pMP1的坐标从世界坐标系变换到相机1坐标系
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);//优化图的第 2*i+1个顶点
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                cv::Mat P3D2c = R2w*P3D2w + t2w;//将地图点pMP2的坐标从世界坐标系变换到相机2坐标系
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);//优化图的第 2*(i+1)个顶点
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            /*若某个地图点为坏点或地图点在第二个关键帧中下标小于0*/
            else
                continue;
        }
        //若某个地图点指针为空
        else
            continue;

        nCorrespondences++;//增加对应的个数

        // Set edge x1 = S12*X2
        Eigen::Matrix<double,2,1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;//从关键帧1中取出i下标所对应关键点的像素坐标

        g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));//优化图的第 2*(i+1)个顶点，存储的是地图点在相机2坐标系的坐标
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));//顶点0是一个sim3顶点，记录了关键帧1,2各自的内参以及相对的sim3
        e12->setMeasurement(obs1);//观测是地图点在关键帧1下成像的像素坐标
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];//用地图点在关键帧1下关键点所在金字塔层的参数加权信息矩阵
        e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);//为边设置核函数并向优化图中添加边，这条边是由关键帧2坐标系下的地图点向关键帧1图像平面投影的约束

        // Set edge x2 = S21*X1
        Eigen::Matrix<double,2,1> obs2;
        const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];//i2为对应地图点在关键帧2中的下标
        obs2 << kpUn2.pt.x, kpUn2.pt.y;//取出地图点在关键帧2中的像素坐标
        /*e12是EdgeSim3ProjectXYZ类型，e21是EdgeInverseSim3ProjectXYZ类型*/
        g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));//优化图的第 2*i+1个顶点，存储的是地图点在相机1坐标系下的坐标
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));//顶点0是一个sim3顶点，记录了关键帧1,2各自的内参以及相对的sim3
        e21->setMeasurement(obs2);//观测是地图点在关键帧2下成像的像素坐标
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];//用地图点在关键帧2下关键点所在金字塔层的参数加权信息矩阵
        e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);//为边设置核函数并向优化图中添加边，这条边描述的是关键帧1坐标系下的地图点向关键帧2图像平面投影的约束

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);//第i对对应的地图点
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers核查局内点
    int nBad=0;
    /**/
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        /*如果取出的边为空，则继续处理下一对边*/
        if(!e12 || !e21)
            continue;
        /*若这对边中某一条边的误差大于阈值，则将vpMatches1对应下标的地图点指针置为空，同时从优化图中去除这两条边，然后将对应向量中两条边的指针置为空指针，最后将统计的局外点数加1*/
        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=static_cast<g2o::EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i]=static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;
        }
    }

    int nMoreIterations;
    if(nBad>0)//若局外点数大于0，则设置下次迭代数为10，若无局外点则下次优化迭代5次，下次迭代时不再有局外点只有局内点
        nMoreIterations=10;
    else
        nMoreIterations=5;

    if(nCorrespondences-nBad<10)//对应点数-局外点数=局内点数 ，若局内点过少，则返回0不再继续下面的处理
        return 0;

    // Optimize again only with inliers
    //只用局内点再优化一次
    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    /*
    对于所有的对应边，若某对边任一指针为空则不作处理，若任一边的误差大于阈值，则取出边所对应在vpMatches1中的下标，将vpMatches1在该下标的指针置为空，否则将统计的局内点数加1
    */
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
        }
        else
            nIn++;
    }

    // Recover optimized Sim3   将优化后的顶点0的sim3取出放回函数参数g2oS12中
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12= vSim3_recov->estimate();
    /*返回统计的局内点数*/
    return nIn;
}


} //namespace ORB_SLAM
