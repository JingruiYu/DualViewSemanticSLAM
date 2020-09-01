
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

#include "OdomG2oType.h"

#define USE_MY_TYPE

namespace ORB_SLAM2
{
void Optimizer::GlobalBundleAdjustemntWithBirdview(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    vector<MapPointBird*> vpMPBird = pMap->GetAllMapPointsBird();
    BundleAdjustmentWithBirdview(vpKFs,vpMP,vpMPBird,nIterations,pbStopFlag, nLoopKF, bRobust);
}

void Optimizer::BundleAdjustmentWithBirdview(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,const std::vector<MapPointBird*> &vpMPBird,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    cout<<"BundleAdjustmentWithBirdview..."<<endl;

    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    vector<bool> vbNotIncludedMPBird;
    vbNotIncludedMPBird.resize(vpMPBird.size());

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // Set KeyFrame vertices
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;

        VertexSE3Quat *vSE3 = new VertexSE3Quat();
        
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    int maxMPid = 0;
    // Set MapPoint vertices
    for(size_t i=0; i<vpMP.size(); i++)
    {
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        const int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        if(id>maxMPid)
            maxMPid = id;
     
        const map<KeyFrame*,size_t> observations = pMP->GetObservations();
        int nEdges = 0;
        //SET EDGES
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {
            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid)
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];

            if(pKF->mvuRight[mit->second]<0)
            {
                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                EdgeSE3ProjectXYZ2UVQuat *e = new EdgeSE3ProjectXYZ2UVQuat();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                optimizer.addEdge(e);
            }
            else
            {
                // Eigen::Matrix<double,3,1> obs;
                // const float kp_ur = pKF->mvuRight[mit->second];
                // obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                // g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                // e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                // e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                // e->setMeasurement(obs);
                // const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                // Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                // e->setInformation(Info);

                // if(bRobust)
                // {
                //     g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                //     e->setRobustKernel(rk);
                //     rk->setDelta(thHuber3D);
                // }

                // e->fx = pKF->fx;
                // e->fy = pKF->fy;
                // e->cx = pKF->cx;
                // e->cy = pKF->cy;
                // e->bf = pKF->mbf;

                // optimizer.addEdge(e);

                //TODO: Add Stereo Types
            }
        }

        if(nEdges==0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }

    // Set MapPointBird vertices
    for(size_t i=0; i<vpMPBird.size(); i++)
    {
        MapPointBird* pMPBird = vpMPBird[i];
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMPBird->GetWorldPos()));
        const int id = pMPBird->mnId+maxMPid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,size_t> observations = pMPBird->GetObservations();
        int nEdgesBird = 0;
        //SET EDGES Birdview
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {
            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid)
                continue;

            nEdgesBird++;

            const cv::Point3f pt = pKF->mvKeysBirdCamXYZ[mit->second];

            Eigen::Matrix<double,3,1> obs;
            obs << pt.x, pt.y,pt.z;

            EdgeSE3ProjectXYZ2XYZQuat *e = new EdgeSE3ProjectXYZ2XYZQuat();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
            e->setMeasurement(obs);
            // float scale = Frame::meter2pixel*Frame::meter2pixel;
            float scale = 3.0;
            const float &invSigma2 = pKF->mvInvLevelSigma2[pKF->mvKeysBird[mit->second].octave]*scale;
            e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

            if(bRobust)
            {
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                // rk->setDelta(thHuber2D);
                rk->setDelta(thHuber3D);
            }

            optimizer.addEdge(e);
        }

        if(nEdgesBird==0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMPBird[i]=true;
        }
        else
        {
            vbNotIncludedMPBird[i]=false;
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data

    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;

        VertexSE3Quat *vSE3 = static_cast<VertexSE3Quat*>(optimizer.vertex(pKF->mnId));
       
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

    // Birdview Points
    for(size_t i=0; i<vpMPBird.size(); i++)
    {
        if(vbNotIncludedMPBird[i])
            continue;

        MapPointBird* pMPBird = vpMPBird[i];
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMPBird->mnId+maxMPid+1));
        pMPBird->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
    }

}

int Optimizer::PoseOptimizationWithBirdview(Frame *pFrame, Frame* pRefFrame)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    // Set Frame vertex

    VertexSE3Quat *vSE3 = new VertexSE3Quat();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;
    vector<EdgeSE3ProjectXYZOnlyPoseQuat*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    vector<EdgePointTransformSE3Quat*> vpEdgesBird;
    vector<size_t> vnIndexEdgeBird;

    vector<EdgeSE3ProjectXYZ2XYZOnlyPoseQuat*> vpEdgesBird3D;
    vector<size_t> vnIndexEdgeBird3D;


    // vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    // vector<size_t> vnIndexEdgeStereo;
    // vpEdgesStereo.reserve(N);
    // vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    // const float deltaStereo = sqrt(7.815);
    // const float delta3D = sqrt(7.815);


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

                EdgeSE3ProjectXYZOnlyPoseQuat *e = new EdgeSE3ProjectXYZOnlyPoseQuat();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
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

                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
            else  // Stereo observation
            {
                // nInitialCorrespondences++;
                // pFrame->mvbOutlier[i] = false;

                // //SET EDGE
                // Eigen::Matrix<double,3,1> obs;
                // const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                // const float &kp_ur = pFrame->mvuRight[i];
                // obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                // g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                // e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                // e->setMeasurement(obs);
                // const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                // Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                // e->setInformation(Info);

                // g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                // e->setRobustKernel(rk);
                // rk->setDelta(deltaStereo);

                // e->fx = pFrame->fx;
                // e->fy = pFrame->fy;
                // e->cx = pFrame->cx;
                // e->cy = pFrame->cy;
                // e->bf = pFrame->mbf;
                // cv::Mat Xw = pMP->GetWorldPos();
                // e->Xw[0] = Xw.at<float>(0);
                // e->Xw[1] = Xw.at<float>(1);
                // e->Xw[2] = Xw.at<float>(2);

                // optimizer.addEdge(e);

                // vpEdgesStereo.push_back(e);
                // vnIndexEdgeStereo.push_back(i);

                //TODO: Add Stereo Types
            }
        }

    }

    vector<MapPointBird*> &vpMapPointsBird = pFrame->mvpMapPointsBird;
    for(int k=0,kend=vpMapPointsBird.size();k<kend;k++)
    {
        if(vpMapPointsBird[k])
        {
            Vector3d Xw = Converter::toVector3d(vpMapPointsBird[k]->GetWorldPos());
            Vector3d Xc;
            cv::Point3f p = pFrame->mvKeysBirdCamXYZ[k];
            Xc<<p.x,p.y,p.z;

            EdgeSE3ProjectXYZ2XYZOnlyPoseQuat *e = new EdgeSE3ProjectXYZ2XYZOnlyPoseQuat();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));

            // float scale = Frame::meter2pixel*Frame::meter2pixel;
            // float scale = Frame::meter2pixel;
            float scale = 3.0;
            const float invSigma2 = pFrame->mvInvLevelSigma2[pFrame->mvKeysBird[k].octave]*scale;
            e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            // rk->setDelta(delta3D);
            rk->setDelta(deltaMono);


            e->Xw = Xw;
            e->Xc = Xc;

            optimizer.addEdge(e);

            vpEdgesBird3D.push_back(e);
            vnIndexEdgeBird3D.push_back(k);
        }
    }

    if(pRefFrame)
    {
        if(pFrame->mnBirdviewRefFrameId!=pRefFrame->mnId)
        {
            cout<<"Reference Frame not Match."<<endl;
        }
        else
        {
            // cout<<"Optimize with Birdview."<<endl;
            // set reference frane vertex
            VertexSE3Quat *vSE3 = new VertexSE3Quat();
            vSE3->setEstimate(Converter::toSE3Quat(pRefFrame->mTcw));
            vSE3->setId(1);
            vSE3->setFixed(true);
            optimizer.addVertex(vSE3);
            // set edge connections
            std::vector<int> &vnBirdviewMatches21 = pFrame->mvnBirdviewMatches21;
            for(int k=0,kend=vnBirdviewMatches21.size();k<kend;k++)
            {
                if(vnBirdviewMatches21[k]>0)
                {
                    EdgePointTransformSE3Quat *e = new EdgePointTransformSE3Quat();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(1)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));

                    cv::Point3f &pc1cv = pRefFrame->mvKeysBirdCamXYZ[vnBirdviewMatches21[k]];
                    cv::Point3f &pc2cv = pFrame->mvKeysBirdCamXYZ[k];

                    e->pc1[0] = pc1cv.x;
                    e->pc1[1] = pc1cv.y;
                    e->pc1[2] = pc1cv.z;

                    e->pc2[0] = pc2cv.x;
                    e->pc2[1] = pc2cv.y;
                    e->pc2[2] = pc2cv.z;

                    // float scale = Frame::pixel2meter*Frame::pixel2meter;
                    // float scale = Frame::pixel2meter;
                    float scale = 3.0;
                    const float invSigma2 = pFrame->mvInvLevelSigma2[pFrame->mvKeysBird[k].octave]*scale;
                    // float meter2pixel = 1.0/Frame::pixel2meter;
                    // float meter2pixel2 = meter2pixel*meter2pixel;
                    // e->setInformation(Eigen::Matrix3d::Identity()*invSigma2*(pFrame->mnId-pRefFrame->mnId));
                    e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);
                    // e->setInformation(Eigen::Matrix3d::Identity());
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    optimizer.addEdge(e);

                    vpEdgesBird.push_back(e);
                    vnIndexEdgeBird.push_back(k);
                }
            }
        }
    }

    }


    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    // const float chi2Mono[4]={3.841,3.841,3.841,3.841};
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    // const float chi2Mono[4]={7.815,7.815,7.815, 7.815};
    // const float chi2Mono[4]={9.488,9.488,9.488, 9.488};
    // const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    
    
    // const float chi2Bird[4]={3.841,3.841,3.841,3.841};
    const float chi2Bird[4]={5.991,5.991,5.991,5.991};
    // const float chi2Bird[4]={7.815,7.815,7.815, 7.815};
    // const float chi2Bird[4]={9.488,9.488,9.488, 9.488};

    const int its[4]={10,10,10,10};    

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {

        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            EdgeSE3ProjectXYZOnlyPoseQuat *e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
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
                e->setRobustKernel(0);
        }

        int nGoodBird = 0;
        for(size_t i=0, iend=vpEdgesBird3D.size(); i<iend; i++)
        {
            EdgeSE3ProjectXYZ2XYZOnlyPoseQuat *e = vpEdgesBird3D[i];

            const size_t idx = vnIndexEdgeBird3D[i];

            if(!pFrame->mvbBirdviewInliers[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Bird[it])
            {                
                pFrame->mvbBirdviewInliers[idx]=false;
                e->setLevel(1);
            }
            else
            {
                pFrame->mvbBirdviewInliers[idx]=true;
                e->setLevel(0);
                nGoodBird++;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        if(optimizer.edges().size()<10)
            break;
    }

    // Recover optimized pose and return number of inliers
    VertexSE3Quat *vSE3_recov = static_cast<VertexSE3Quat*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);

    return nInitialCorrespondences-nBad;
}

int Optimizer::PoseOptimizationWithBirdviewPixel(Frame *pCurFrame, Frame* pRefFrame)
{
    // 0. graph model
    g2o::SparseOptimizer optimizer;

    // 1. linearSolver
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>(); // linear solver using dense cholesky decomposition
    
    // 2. blockSolver
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    // 3. optimizer algorithm
    g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    // 4. set solver 
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false); // for output

    // 5. vertex
    g2o::VertexSE3Expmap * vSE3c = new g2o::VertexSE3Expmap();
    vSE3c->setId(0);
    vSE3c->setEstimate(Converter::toSE3Quat(pCurFrame->mTcw));
    vSE3c->setFixed(false);
    optimizer.addVertex(vSE3c);

    // 6. edge
    const int Nf = pCurFrame->N;
    vector<EdgeSE3ProjectXYZOnlyWeightPose*> vpEdgesFront;
    // vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesFront;
    vector<size_t> vnIndexEdgeFront;
    vpEdgesFront.reserve(Nf);
    vnIndexEdgeFront.reserve(Nf);
    int nFrontInitialCorrespondences = 0;

    const int Nb = pCurFrame->mvbBirdviewInliers.size();
    vector<EdgeSE3ProjectPw2BirdPixel*> vpEdgesBird;
    vector<size_t> vnIndexEdgeBird;
    vpEdgesBird.reserve(Nb);
    vnIndexEdgeBird.reserve(Nb);
    int nBirdInitialCorrespondences = 0;

    
    const float deltaMono = sqrt(5.991);
    {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);
    // front pts
    for (int i = 0; i < Nf; i++)
    {
        MapPoint* pMP = pCurFrame->mvpMapPoints[i];
        if (pMP)
        {
            nFrontInitialCorrespondences++;
            pCurFrame->mvbOutlier[i] = false;

            Eigen::Matrix<double,2,1> obs;
            const cv::KeyPoint &kpUn = pCurFrame->mvKeysUn[i];
            obs << kpUn.pt.x, kpUn.pt.y;

            EdgeSE3ProjectXYZOnlyWeightPose * e = new EdgeSE3ProjectXYZOnlyWeightPose();
            // g2o::EdgeSE3ProjectXYZOnlyPose * e = new g2o::EdgeSE3ProjectXYZOnlyPose();
            e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0))); // dynamic_cast : type convert for class only
            e->setMeasurement(obs);
            const float invSigma2 = pCurFrame->mvInvLevelSigma2[kpUn.octave]; // uncertainty per pixel
            e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

            g2o::RobustKernelHuber * rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(deltaMono);

            e->fx = pCurFrame->fx;
            e->fy = pCurFrame->fy;
            e->cx = pCurFrame->cx;
            e->cy = pCurFrame->cy;

            cv::Mat Xw = pMP->GetWorldPos();
            e->Xw[0] = Xw.at<float>(0);
            e->Xw[1] = Xw.at<float>(1);
            e->Xw[2] = Xw.at<float>(2);
            
            e->w = 1;

            optimizer.addEdge(e);
            vpEdgesFront.push_back(e);
            vnIndexEdgeFront.push_back(i);
        }
    }

    // bird pts
    for (int i = 0; i < Nb; i++)
    {
        MapPointBird* pMP = pCurFrame->mvpMapPointsBird[i];
        if (pMP)
        {
            nBirdInitialCorrespondences++;
            pCurFrame->mvbBirdviewInliers[i] = true;
            
            Eigen::Matrix<double,2,1> obs;
            const cv::KeyPoint &kpUn = pCurFrame->mvKeysBird[i];
            obs << kpUn.pt.x, kpUn.pt.y;

            EdgeSE3ProjectPw2BirdPixel * e = new EdgeSE3ProjectPw2BirdPixel();
            e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e->setMeasurement(obs);
            const float invSigma2 = pCurFrame->mvInvLevelSigma2[pCurFrame->mvKeysBird[i].octave]; // set according to the uncertainty of pixel location
            e->setInformation(Eigen::Matrix2d::Identity()*invSigma2); 

            g2o::RobustKernelHuber * rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(deltaMono); // less, more linear

            e->Rbc = Converter::toMatrix3d(Frame::Tbc.rowRange(0,3).colRange(0,3));
            e->tbc = Converter::toVector3d(Frame::Tbc.rowRange(0,3).col(3));
            e->meter2pixel = Frame::meter2pixel;
            e->birdviewCols = Frame::birdviewCols;
            e->birdviewRows = Frame::birdviewRows;
            e->rear_axle_to_center = Frame::rear_axle_to_center;

            cv::Mat Xw = pMP->GetWorldPos();
            e->Xw[0] = Xw.at<float>(0);
            e->Xw[1] = Xw.at<float>(1);
            e->Xw[2] = Xw.at<float>(2);

            e->w = 1;

            optimizer.addEdge(e);
            vpEdgesBird.push_back(e);
            vnIndexEdgeBird.push_back(i);
        }
    }

    }

    if (nFrontInitialCorrespondences < 3 && nBirdInitialCorrespondences < 3)
        return 0;

    
    // 7. iteration optimization
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const int its[4]={10,10,10,10};
    int nFrontBad = 0;
    int nBirdBad = 0;

    for (size_t it = 0; it < 4; it++)
    {
        vSE3c->setEstimate(Converter::toSE3Quat(pCurFrame->mTcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nFrontBad = 0;
        for (size_t i = 0, iend = vpEdgesFront.size(); i < iend; i++)
        {
            EdgeSE3ProjectXYZOnlyWeightPose * e = vpEdgesFront[i];
            // g2o::EdgeSE3ProjectXYZOnlyPose * e = vpEdgesFront[i];
            const size_t idx = vnIndexEdgeFront[i];

            if (pCurFrame->mvbOutlier[idx])
                e->computeError();
            
            const float chi2 = e->chi2();
            if (chi2>chi2Mono[it])
            {
                pCurFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nFrontBad++;
            }
            else
            {
                pCurFrame->mvbOutlier[idx] = false;
                e->setLevel(0);
            }

            if (it==2)
                e->setRobustKernel(0); // means no kernel used
        }

        nBirdBad = 0;
        for (size_t i = 0, iend = vpEdgesBird.size(); i < iend; i++)
        {
            EdgeSE3ProjectPw2BirdPixel * e = vpEdgesBird[i];
            const size_t idx = vnIndexEdgeBird[i];

            if (!pCurFrame->mvbBirdviewInliers[idx])
                e->computeError();
            
            const float chi2 = e->chi2();
            if (chi2>chi2Mono[it])
            {
                pCurFrame->mvbBirdviewInliers[idx] = false;
                e->setLevel(1);
                nBirdBad++;
            }
            else
            {
                pCurFrame->mvbBirdviewInliers[idx] = true;
                e->setLevel(0);
            }
            
            if (it==2)
                e->setRobustKernel(0);
        }
    }

    // 8. get result
    g2o::VertexSE3Expmap * vSE3c_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3c_recov = vSE3c_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3c_recov);
    pCurFrame->SetPose(pose);

    bool suc = false;
    if ( (nFrontInitialCorrespondences-nFrontBad) > 0)
        suc = true;
    else
        cout << "\033[31m" << "nFrontInitialCorrespondences-nFrontBad < 0" << "\033[0m" << endl;
    
    if ( (nBirdInitialCorrespondences-nBirdBad) > 0)
        suc = true;
    else
        cout << "\033[31m" << "nBirdInitialCorrespondences-nBirdBad < 0" << "\033[0m" << endl;
    
    
    return suc;
}

void Optimizer::LocalBundleAdjustmentWithBirdview(KeyFrame *pKF, bool* pbStopFlag, Map* pMap)
{    
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    list<MapPoint*> lLocalMapPoints;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
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

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    list<KeyFrame*> lFixedCameras;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {                
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // Local Birdview MapPoints seen in Local KeyFrames
    list<MapPointBird*> lLocalMapPointsBird;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        vector<MapPointBird*> vpMPBirds = (*lit)->GetMapPointMatchesBird();
        for(vector<MapPointBird*>::iterator vit=vpMPBirds.begin(),vend=vpMPBirds.end();vit!=vend;vit++)
        {
            MapPointBird *pMPBird = *vit;
            if(pMPBird)
            {
                if(pMPBird->mnBALocalForKF!=pKF->mnId)
                {
                    lLocalMapPointsBird.push_back(pMPBird);
                    pMPBird->mnBALocalForKF=pKF->mnId;
                }
            }
        }
    }

    // Fixed KeyFrames for Birdview.
    for(list<MapPointBird*>::iterator lit=lLocalMapPointsBird.begin(), lend=lLocalMapPointsBird.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;
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

    // Set Local KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
#ifdef USE_MY_TYPE
        VertexSE3Quat *vSE3 = new VertexSE3Quat();
#else
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
#endif
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set Fixed KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
#ifdef USE_MY_TYPE
        VertexSE3Quat *vSE3 = new VertexSE3Quat();
#else
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
#endif
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set MapPoint vertices
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

#ifdef USE_MY_TYPE
    vector<EdgeSE3ProjectXYZ2UVQuat*> vpEdgesMono;
#else
    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
#endif
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<EdgeSE3ProjectXYZ2XYZQuat*> vpEdgesBird;
    vpEdgesBird.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFBird;
    vpEdgeKFBird.reserve(nExpectedSize);

    vector<MapPointBird*> vpMapPointEdgeBird;
    vpMapPointEdgeBird.reserve(nExpectedSize);


    const float thHuberMono = sqrt(5.991);
    // const float thHuberStereo = sqrt(7.815);
    // const float thHuber3D = sqrt(7.815);

    int maxMPid=0;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId+maxKFid+1;
        if(id>maxMPid)
            maxMPid=id;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        //Set edges
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad())
            {                
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                // Monocular observation
                if(pKFi->mvuRight[mit->second]<0)
                {
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

#ifdef USE_MY_TYPE
                    EdgeSE3ProjectXYZ2UVQuat *e = new EdgeSE3ProjectXYZ2UVQuat();
#else
                    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
#endif

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // Stereo observation
                {
                    // Eigen::Matrix<double,3,1> obs;
                    // const float kp_ur = pKFi->mvuRight[mit->second];
                    // obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    // g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    // e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    // e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    // e->setMeasurement(obs);
                    // const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    // Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    // e->setInformation(Info);

                    // g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    // e->setRobustKernel(rk);
                    // rk->setDelta(thHuberStereo);

                    // e->fx = pKFi->fx;
                    // e->fy = pKFi->fy;
                    // e->cx = pKFi->cx;
                    // e->cy = pKFi->cy;
                    // e->bf = pKFi->mbf;

                    // optimizer.addEdge(e);
                    // vpEdgesStereo.push_back(e);
                    // vpEdgeKFStereo.push_back(pKFi);
                    // vpMapPointEdgeStereo.push_back(pMP);

                    //TODO: Add Stereo Types
                }
            }
        }
    }

    // Birdview Points and Edges
    for(list<MapPointBird*>::iterator lit=lLocalMapPointsBird.begin(), lend=lLocalMapPointsBird.end(); lit!=lend; lit++)
    {
        MapPointBird* pMPBird = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMPBird->GetWorldPos()));
        int id = pMPBird->mnId+maxMPid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,size_t> observations = pMPBird->GetObservations();
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad())
            {                
                const cv::Point3f &pt = pKFi->mvKeysBirdCamXYZ[mit->second];

                Eigen::Matrix<double,3,1> obs;
                obs << pt.x, pt.y, pt.z;

                EdgeSE3ProjectXYZ2XYZQuat *e = new EdgeSE3ProjectXYZ2XYZQuat();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                e->setMeasurement(obs);
                // float scale = Frame::meter2pixel*Frame::meter2pixel;
                float scale = 5.0;
                const float &invSigma2 = pKFi->mvInvLevelSigma2[pKFi->mvKeysBird[mit->second].octave]*scale;
                e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                // rk->setDelta(thHuber3D);
                rk->setDelta(thHuberMono);


                optimizer.addEdge(e);
                vpEdgesBird.push_back(e);
                vpEdgeKFBird.push_back(pKFi);
                vpMapPointEdgeBird.push_back(pMPBird);
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

    if(bDoMore)
    {

    // Check inlier observations
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
#ifdef USE_MY_TYPE
        EdgeSE3ProjectXYZ2UVQuat *e = vpEdgesMono[i];
#else
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
#endif
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())  // 7.815, 5.991, 9.488
        {
            e->setLevel(1);
        }

        e->setRobustKernel(0);
    }
    // Check inlier observations bird view
    for(size_t i=0, iend=vpEdgesBird.size(); i<iend;i++)
    {
        EdgeSE3ProjectXYZ2XYZQuat *e = vpEdgesBird[i];

        // MapPointBird* pMPBird = vpMapPointEdgeBird[i];
        e->computeError();
        if(e->chi2()>5.991)  //3.841, 5.991, 7.815, 9.488
        {
            e->setLevel(1);
        }

        e->setRobustKernel(0);
    }

    // Optimize again without the outliers

    optimizer.initializeOptimization(0);
    optimizer.optimize(10);

    }

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    // vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());
    vToErase.reserve(vpEdgesMono.size());

    // Check inlier observations       
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
#ifdef USE_MY_TYPE
        EdgeSE3ProjectXYZ2UVQuat *e = vpEdgesMono[i];
#else
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
#endif
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())  // 3.841, 5.991, 7.815, 9.488
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    vector<pair<KeyFrame*,MapPointBird*> > vToEraseBird;
    vToEraseBird.reserve(vpEdgesBird.size());

    // Check inlier observations       
    for(size_t i=0, iend=vpEdgesBird.size(); i<iend;i++)
    {
        EdgeSE3ProjectXYZ2XYZQuat *e = vpEdgesBird[i];

        MapPointBird* pMPBird = vpMapPointEdgeBird[i];

        if(e->chi2()>5.991)  // 3.841, 5.991, 7.815, 9.488
        {
            KeyFrame* pKFi = vpEdgeKFBird[i];
            vToEraseBird.push_back(make_pair(pKFi,pMPBird));
        }
    }


    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);
    //cout<<"Erase "<<vToErase.size()<<" MapPoints, and "<<vToEraseBird.size()<<" Birdview Points."<<endl;

    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    if(!vToEraseBird.empty())
    {
        for(size_t i=0;i<vToEraseBird.size();i++)
        {
            KeyFrame* pKFi = vToEraseBird[i].first;
            MapPointBird* pMPi = vToEraseBird[i].second;
            pKFi->EraseMapPointMatchBird(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data

    //Keyframes
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
#ifdef USE_MY_TYPE
        VertexSE3Quat *vSE3 = static_cast<VertexSE3Quat*>(optimizer.vertex(pKF->mnId));
#else
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
#endif
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }

    // Birdview Points
    for(list<MapPointBird*>::iterator lit=lLocalMapPointsBird.begin(), lend=lLocalMapPointsBird.end(); lit!=lend; lit++)
    {
        MapPointBird *pMPBird = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMPBird->mnId+maxMPid+1));
        pMPBird->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
    }
}

void Optimizer::poseDirectEstimation(const Frame &ReferenceFrame, const Frame &CurrentFrame, cv::Mat &Tcw )
{
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,1> > DirectBlock;  // 求解的向量是6＊1的 //TODO
    DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense< DirectBlock::PoseMatrixType > ();
    DirectBlock* solver_ptr = new DirectBlock ( linearSolver );
    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr ); // G-N
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr ); // L-M
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );
    optimizer.setVerbose( true );

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setEstimate ( Converter::toSE3Quat(Tcw) );
    pose->setId ( 0 );
    optimizer.addVertex ( pose );

    // 添加边
    int id=1;
    vector<EdgeSE3ProjectDirect*> edges; //test
    for ( size_t i = 0; i < ReferenceFrame.mvMeasurement_p.size(); i++ )
    {
        EdgeSE3ProjectDirect* edge = new EdgeSE3ProjectDirect (
            Converter::toVector3d(ReferenceFrame.mvMeasurement_p[i]), 
            ReferenceFrame.Tcb.rowRange(0,3).colRange(0,3), 
            ReferenceFrame.Tcb.rowRange(0,3).col(3), 
            ReferenceFrame.Rro, ReferenceFrame.tro, 
            ReferenceFrame.Ror, ReferenceFrame.tor,
            CurrentFrame.mBirdviewContour
        );
        edge->setVertex ( 0, pose );
        edge->setMeasurement ( ReferenceFrame.mvMeasurement_g[i] );
        edge->setInformation ( Eigen::Matrix<double,1,1>::Identity() );
        edge->setId ( id++ );
        optimizer.addEdge ( edge );
        edges.push_back( edge ); //test
    }
    // cout<<"edges in graph: "<<optimizer.edges().size() <<endl;
    optimizer.initializeOptimization();
    optimizer.optimize ( 30 );
    Eigen::Isometry3d tmpTcw = pose->estimate();
    double adpStep = abs(tmpTcw(0,3)-Tcw.at<float>(0,3)) + abs(tmpTcw(1,3)-Tcw.at<float>(1,3)) + abs(tmpTcw(2,3)-Tcw.at<float>(2,3));
    if (adpStep < 18)
    {
        Tcw = Converter::toCvMat(tmpTcw);
    }
}

int Optimizer::poseOptimizationFull(Frame* pCurFrame, Frame* pRefFrame)
{
    // 0. graph model
    g2o::SparseOptimizer optimizer;

    // 1. linearSolver
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>(); // linear solver using dense cholesky decomposition
    // linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>(); using sparse cholesky solver from Eigen
    // linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolverX::PoseMatrixType>(); using CSparse
    // linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolverX::PoseMatrixType>(); basic solver 

    // 2. blockSolver
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    // 3. optimizer algorithm
    g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    // 4. set solver 
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false); // for output

    // 5. vertex
    g2o::VertexSE3Expmap * vSE3c = new g2o::VertexSE3Expmap();
    vSE3c->setId(0);
    vSE3c->setEstimate(Converter::toSE3Quat(pCurFrame->mTcw));
    vSE3c->setFixed(false);
    optimizer.addVertex(vSE3c);

    // g2o::VertexSE3Expmap * vSE3r = new g2o::VertexSE3Expmap();
    // vSE3r->setId(1);
    // vSE3r->setEstimate(Converter::toSE3Quat(pRefFrame->mTcw));
    // vSE3r->setFixed(false);
    // optimizer.addVertex(vSE3r); 

    // 6. edge
    const int Nf = pCurFrame->N;
    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesFront;
    vector<size_t> vnIndexEdgeFront;
    vpEdgesFront.reserve(Nf);
    vnIndexEdgeFront.reserve(Nf);
    int nFrontInitialCorrespondences = 0;

    const int Nb = pCurFrame->mvbBirdviewInliers.size();
    vector<EdgeSE3ProjectXYZ2XYZOnlyPoseQuat*> vpEdgesBird;
    vector<size_t> vnIndexEdgeBird;
    vpEdgesBird.reserve(Nb);
    vnIndexEdgeBird.reserve(Nb);
    int nBirdInitialCorrespondences = 0;
    
    const int Nd = 0; // pRefFrame->mvMeasurement_p.size(); // the number of these edge should be controlled
    vector<EdgeSE3ProjectDirect*> vpEdgesDirect;
    vector<size_t> vnIndexEdgeDirect;
    vector<bool> vOutlierDirect;
    vpEdgesDirect.reserve(Nd);
    vnIndexEdgeDirect.reserve(Nd);
    vOutlierDirect = vector<bool>(Nd,false);

    const float deltaMono = sqrt(5.991);
    {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);
    // front pts
    for (int i = 0; i < Nf; i++)
    {
        MapPoint* pMP = pCurFrame->mvpMapPoints[i];
        if (pMP)
        {
            nFrontInitialCorrespondences++;
            pCurFrame->mvbOutlier[i] = false;

            Eigen::Matrix<double,2,1> obs;
            const cv::KeyPoint &kpUn = pCurFrame->mvKeysUn[i];
            obs << kpUn.pt.x, kpUn.pt.y;

            g2o::EdgeSE3ProjectXYZOnlyPose * e = new g2o::EdgeSE3ProjectXYZOnlyPose();
            e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0))); // dynamic_cast : type convert for class only
            e->setMeasurement(obs);
            const float invSigma2 = pCurFrame->mvInvLevelSigma2[kpUn.octave]; // uncertainty per pixel
            e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

            g2o::RobustKernelHuber * rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(deltaMono);

            e->fx = pCurFrame->fx;
            e->fy = pCurFrame->fy;
            e->cx = pCurFrame->cx;
            e->cy = pCurFrame->cy;

            cv::Mat Xw = pMP->GetWorldPos();
            e->Xw[0] = Xw.at<float>(0);
            e->Xw[1] = Xw.at<float>(1);
            e->Xw[2] = Xw.at<float>(2);
            
            optimizer.addEdge(e);
            vpEdgesFront.push_back(e);
            vnIndexEdgeFront.push_back(i);
        }
    }

    // bird pts
    for (int i = 0; i < Nb; i++)
    {
        MapPointBird* pMP = pCurFrame->mvpMapPointsBird[i];
        if (pMP)
        {
            nBirdInitialCorrespondences++;
            pCurFrame->mvbBirdviewInliers[i] = true;
            
            Vector3d Xw = Converter::toVector3d(pMP->GetWorldPos());
            Vector3d Xc;
            cv::Point3f p = pCurFrame->mvKeysBirdCamXYZ[i];
            Xc << p.x, p.y, p.z;

            EdgeSE3ProjectXYZ2XYZOnlyPoseQuat * e = new EdgeSE3ProjectXYZ2XYZOnlyPoseQuat();
            e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e->Xc = Xc;
            float scale = 3.0;
            const float invSigma2 = pCurFrame->mvInvLevelSigma2[pCurFrame->mvKeysBird[i].octave]*scale; // set according to the uncertainty of pixel location
            e->setInformation(Eigen::Matrix3d::Identity()*invSigma2); 

            g2o::RobustKernelHuber * rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(deltaMono); // less, more linear

            e->Xw = Xw;

            optimizer.addEdge(e);
            vpEdgesBird.push_back(e);
            vnIndexEdgeBird.push_back(i);
        }
    }

    // bird direct
    for (int i = 0; i < Nd; i++)
    {
        EdgeSE3ProjectDirect * edge = new EdgeSE3ProjectDirect(
            Converter::toVector3d(pRefFrame->mvMeasurement_p[i]),
            pRefFrame->Tcb.rowRange(0,3).colRange(0,3),
            pRefFrame->Tcb.rowRange(0,3).col(3),
            pRefFrame->Rro, pRefFrame->tro,
            pRefFrame->Ror, pRefFrame->tor,
            pCurFrame->mBirdviewContour
        );
        edge->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        edge->setMeasurement(pRefFrame->mvMeasurement_g[i]);
        edge->setInformation(Eigen::Matrix<double,1,1>::Identity()*5);

        g2o::RobustKernelHuber * rk = new g2o::RobustKernelHuber;
        edge->setRobustKernel(rk);
        rk->setDelta(deltaMono);

        optimizer.addEdge(edge);
        vpEdgesDirect.push_back(edge);
        vnIndexEdgeDirect.push_back(i);
    }
        
    }

    // cout << "the feature of front: ... " << vnIndexEdgeFront.size() << endl;
    // cout << "the feature of bird : ... " << vnIndexEdgeBird.size() << endl;
    // cout << "the directio of bird: ... " << vnIndexEdgeDirect.size() << endl << endl;

    if (nFrontInitialCorrespondences < 3 && nBirdInitialCorrespondences < 3)
        return 0;

    
    
    // 7. iteration optimization
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const int its[4]={10,10,10,10};
    int nFrontBad = 0;
    int nBirdBad = 0;
    int nDirectBad = 0;

    for (size_t it = 0; it < 4; it++)
    {
        vSE3c->setEstimate(Converter::toSE3Quat(pCurFrame->mTcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nFrontBad = 0;
        for (size_t i = 0, iend = vpEdgesFront.size(); i < iend; i++)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose * e = vpEdgesFront[i];
            const size_t idx = vnIndexEdgeFront[i];

            if (pCurFrame->mvbOutlier[idx])
                e->computeError();
            
            const float chi2 = e->chi2();
            if (chi2>chi2Mono[it])
            {
                pCurFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nFrontBad++;
            }
            else
            {
                pCurFrame->mvbOutlier[idx] = false;
                e->setLevel(0);
            }

            if (it==2)
                e->setRobustKernel(0); // means no kernel used
        }

        nBirdBad = 0;
        for (size_t i = 0, iend = vpEdgesBird.size(); i < iend; i++)
        {
            EdgeSE3ProjectXYZ2XYZOnlyPoseQuat * e = vpEdgesBird[i];
            const size_t idx = vnIndexEdgeBird[i];

            if (!pCurFrame->mvbBirdviewInliers[idx])
                e->computeError();
            
            const float chi2 = e->chi2();
            if (chi2>chi2Mono[it])
            {
                pCurFrame->mvbBirdviewInliers[idx] = false;
                e->setLevel(1);
                nBirdBad++;
            }
            else
            {
                pCurFrame->mvbBirdviewInliers[idx] = true;
                e->setLevel(0);
            }
            
            if (it==2)
                e->setRobustKernel(0);
        }

        nDirectBad = 0;
        for (size_t i = 0, iend = vpEdgesDirect.size(); i < iend; i++)
        {
            EdgeSE3ProjectDirect * edge = vpEdgesDirect[i];
            const size_t idx = vnIndexEdgeDirect[i];

            if (!vOutlierDirect[idx])
                edge->computeError();
            
            const float chi2 = edge->chi2();
            if (chi2>chi2Mono[it])
            {
                vOutlierDirect[idx] = true;
                edge->setLevel(1);
                nDirectBad++;
            }
            else
            {
                vOutlierDirect[idx] = false;
                edge->setLevel(0);
            }
        }
        
        if (optimizer.edges().size() < 10)
            break;
    }
    
    // 8. get result
    g2o::VertexSE3Expmap * vSE3c_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3c_recov = vSE3c_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3c_recov);
    if (1)
    {
        pCurFrame->SetPose(pose);
    }

    return nFrontInitialCorrespondences-nFrontBad;
}


int Optimizer::poseOptimizationWeight(Frame* pCurFrame, Frame* pRefFrame)
{
    // 0. graph model
    g2o::SparseOptimizer optimizer;

    // 1. linearSolver
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>(); // linear solver using dense cholesky decomposition
    // linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>(); using sparse cholesky solver from Eigen
    // linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolverX::PoseMatrixType>(); using CSparse
    // linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolverX::PoseMatrixType>(); basic solver 

    // 2. blockSolver
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    // 3. optimizer algorithm
    g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    // 4. set solver 
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false); // for output

    // 5. vertex
    g2o::VertexSE3Expmap * vSE3c = new g2o::VertexSE3Expmap();
    vSE3c->setId(0);
    vSE3c->setEstimate(Converter::toSE3Quat(pCurFrame->mTcw));
    vSE3c->setFixed(false);
    optimizer.addVertex(vSE3c);

    // g2o::VertexSE3Expmap * vSE3r = new g2o::VertexSE3Expmap();
    // vSE3r->setId(1);
    // vSE3r->setEstimate(Converter::toSE3Quat(pRefFrame->mTcw));
    // vSE3r->setFixed(false);
    // optimizer.addVertex(vSE3r); 

    // 6. edge
    const int Nf = pCurFrame->N;
    vector<EdgeSE3ProjectXYZOnlyWeightPose*> vpEdgesFront;
    vector<size_t> vnIndexEdgeFront;
    vpEdgesFront.reserve(Nf);
    vnIndexEdgeFront.reserve(Nf);
    int nFrontInitialCorrespondences = 0;

    const int Nb = pCurFrame->mvbBirdviewInliers.size();
    vector<EdgeSE3ProjectXYZ2XYZOnlyPoseQuat*> vpEdgesBird;
    vector<size_t> vnIndexEdgeBird;
    vpEdgesBird.reserve(Nb);
    vnIndexEdgeBird.reserve(Nb);
    int nBirdInitialCorrespondences = 0;
    
    const int Nd = 0; // pRefFrame->mvMeasurement_p.size(); // the number of these edge should be controlled
    vector<EdgeSE3ProjectDirect*> vpEdgesDirect;
    vector<size_t> vnIndexEdgeDirect;
    vector<bool> vOutlierDirect;
    vpEdgesDirect.reserve(Nd);
    vnIndexEdgeDirect.reserve(Nd);
    vOutlierDirect = vector<bool>(Nd,false);

    const float deltaMono = sqrt(5.991);
    {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);
    // front pts
    for (int i = 0; i < Nf; i++)
    {
        MapPoint* pMP = pCurFrame->mvpMapPoints[i];
        if (pMP)
        {
            nFrontInitialCorrespondences++;
            pCurFrame->mvbOutlier[i] = false;

            Eigen::Matrix<double,2,1> obs;
            const cv::KeyPoint &kpUn = pCurFrame->mvKeysUn[i];
            obs << kpUn.pt.x, kpUn.pt.y;

            EdgeSE3ProjectXYZOnlyWeightPose * e = new EdgeSE3ProjectXYZOnlyWeightPose();
            e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0))); // dynamic_cast : type convert for class only
            e->setMeasurement(obs);
            const float invSigma2 = pCurFrame->mvInvLevelSigma2[kpUn.octave]; // uncertainty per pixel
            double w = 1;
            e->setInformation(Eigen::Matrix2d::Identity()*invSigma2*w);

            g2o::RobustKernelHuber * rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(deltaMono);

            e->w = 0.2;
            e->fx = pCurFrame->fx;
            e->fy = pCurFrame->fy;
            e->cx = pCurFrame->cx;
            e->cy = pCurFrame->cy;

            cv::Mat Xw = pMP->GetWorldPos();
            e->Xw[0] = Xw.at<float>(0);
            e->Xw[1] = Xw.at<float>(1);
            e->Xw[2] = Xw.at<float>(2);
            
            optimizer.addEdge(e);
            vpEdgesFront.push_back(e);
            vnIndexEdgeFront.push_back(i);
        }
    }

    // bird pts
    for (int i = 0; i < Nb; i++)
    {
        MapPointBird* pMP = pCurFrame->mvpMapPointsBird[i];
        if (pMP)
        {
            nBirdInitialCorrespondences++;
            pCurFrame->mvbBirdviewInliers[i] = true;
            
            Vector3d Xw = Converter::toVector3d(pMP->GetWorldPos());
            Vector3d Xc;
            cv::Point3f p = pCurFrame->mvKeysBirdCamXYZ[i];
            Xc << p.x, p.y, p.z;

            EdgeSE3ProjectXYZ2XYZOnlyPoseQuat * e = new EdgeSE3ProjectXYZ2XYZOnlyPoseQuat();
            e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e->Xc = Xc;
            float scale = 3.0;
            const float invSigma2 = pCurFrame->mvInvLevelSigma2[pCurFrame->mvKeysBird[i].octave]*scale; // set according to the uncertainty of pixel location
            e->setInformation(Eigen::Matrix3d::Identity()*invSigma2); 

            g2o::RobustKernelHuber * rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(deltaMono); // less, more linear

            e->Xw = Xw;

            optimizer.addEdge(e);
            vpEdgesBird.push_back(e);
            vnIndexEdgeBird.push_back(i);
        }
    }

    // bird direct
    for (int i = 0; i < Nd; i++)
    {
        EdgeSE3ProjectDirect * edge = new EdgeSE3ProjectDirect(
            Converter::toVector3d(pRefFrame->mvMeasurement_p[i]),
            pRefFrame->Tcb.rowRange(0,3).colRange(0,3),
            pRefFrame->Tcb.rowRange(0,3).col(3),
            pRefFrame->Rro, pRefFrame->tro,
            pRefFrame->Ror, pRefFrame->tor,
            pCurFrame->mBirdviewContour
        );
        edge->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        edge->setMeasurement(pRefFrame->mvMeasurement_g[i]);
        edge->setInformation(Eigen::Matrix<double,1,1>::Identity()*5);

        g2o::RobustKernelHuber * rk = new g2o::RobustKernelHuber;
        edge->setRobustKernel(rk);
        rk->setDelta(deltaMono);

        optimizer.addEdge(edge);
        vpEdgesDirect.push_back(edge);
        vnIndexEdgeDirect.push_back(i);
    }
        
    }

    // cout << "the feature of front: ... " << vnIndexEdgeFront.size() << endl;
    // cout << "the feature of bird : ... " << vnIndexEdgeBird.size() << endl;
    // cout << "the directio of bird: ... " << vnIndexEdgeDirect.size() << endl << endl;

    if (nFrontInitialCorrespondences < 3 && nBirdInitialCorrespondences < 3)
        return 0;

    
    
    // 7. iteration optimization
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const int its[4]={10,10,10,10};
    int nFrontBad = 0;
    int nBirdBad = 0;
    int nDirectBad = 0;

    for (size_t it = 0; it < 4; it++)
    {
        vSE3c->setEstimate(Converter::toSE3Quat(pCurFrame->mTcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        // g2o::VertexSE3Expmap * vSE3c_tmp = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
        // g2o::SE3Quat SE3c_tmp = vSE3c_tmp->estimate();
        // cv::Mat pose = Converter::toCvMat(SE3c_tmp);
        // pCurFrame->mTcw.rowRange(0,3).col(3).copyTo(pose.rowRange(0,3).col(3));
        // vSE3c->setEstimate(Converter::toSE3Quat(pose));

        nFrontBad = 0;
        for (size_t i = 0, iend = vpEdgesFront.size(); i < iend; i++)
        {
            EdgeSE3ProjectXYZOnlyWeightPose * e = vpEdgesFront[i];
            const size_t idx = vnIndexEdgeFront[i];

            if (pCurFrame->mvbOutlier[idx])
                e->computeError();
            
            const float chi2 = e->chi2();
            if (chi2>chi2Mono[it])
            {
                pCurFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nFrontBad++;
            }
            else
            {
                pCurFrame->mvbOutlier[idx] = false;
                e->setLevel(0);
            }

            if (it==2)
                e->setRobustKernel(0); // means no kernel used
        }

        nBirdBad = 0;
        for (size_t i = 0, iend = vpEdgesBird.size(); i < iend; i++)
        {
            EdgeSE3ProjectXYZ2XYZOnlyPoseQuat * e = vpEdgesBird[i];
            const size_t idx = vnIndexEdgeBird[i];

            if (!pCurFrame->mvbBirdviewInliers[idx])
                e->computeError();
            
            const float chi2 = e->chi2();
            if (chi2>chi2Mono[it])
            {
                pCurFrame->mvbBirdviewInliers[idx] = false;
                e->setLevel(1);
                nBirdBad++;
            }
            else
            {
                pCurFrame->mvbBirdviewInliers[idx] = true;
                e->setLevel(0);
            }
            
            if (it==2)
                e->setRobustKernel(0);
        }

        nDirectBad = 0;
        for (size_t i = 0, iend = vpEdgesDirect.size(); i < iend; i++)
        {
            EdgeSE3ProjectDirect * edge = vpEdgesDirect[i];
            const size_t idx = vnIndexEdgeDirect[i];

            if (!vOutlierDirect[idx])
                edge->computeError();
            
            const float chi2 = edge->chi2();
            if (chi2>chi2Mono[it])
            {
                vOutlierDirect[idx] = true;
                edge->setLevel(1);
                nDirectBad++;
            }
            else
            {
                vOutlierDirect[idx] = false;
                edge->setLevel(0);
            }
        }
        
        if (optimizer.edges().size() < 10)
            break;
    }
    
    // 8. get result
    g2o::VertexSE3Expmap * vSE3c_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3c_recov = vSE3c_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3c_recov);
    if (1)
    {
        pCurFrame->SetPose(pose);
    }

    return nFrontInitialCorrespondences-nFrontBad;
}


int Optimizer::poseOptimizationRotation(Frame* pCurFrame, Frame* pRefFrame)
{
    // 0. graph model
    g2o::SparseOptimizer optimizer;

    // 1. linearSolver
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>(); // linear solver using dense cholesky decomposition
    // linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>(); using sparse cholesky solver from Eigen
    // linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolverX::PoseMatrixType>(); using CSparse
    // linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolverX::PoseMatrixType>(); basic solver 

    // 2. blockSolver
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    // 3. optimizer algorithm
    g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    // 4. set solver 
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false); // for output

    // 5. vertex
    VertexRotation * vRc = new VertexRotation();
    vRc->setId(0);
    cv::Mat mRcw = pCurFrame->mTcw.rowRange(0,3).colRange(0,3);
    cv::Mat mtcw = pCurFrame->mTcw.rowRange(0,3).col(3);
    vRc->setEstimate(Converter::toQuaterniond(mRcw));
    vRc->setFixed(false);
    optimizer.addVertex(vRc);

    // 6. edge
    const int Nf = pCurFrame->N;
    vector<EdgeSO3ProjectXYZOnlyRotation*> vpEdgesFront;
    vector<size_t> vnIndexEdgeFront;
    vpEdgesFront.reserve(Nf);
    vnIndexEdgeFront.reserve(Nf);
    int nFrontInitialCorrespondences = 0;

    const float deltaMono = sqrt(5.991);
    {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);
    // front pts
    for (int i = 0; i < Nf; i++)
    {
        MapPoint* pMP = pCurFrame->mvpMapPoints[i];
        if (pMP)
        {
            nFrontInitialCorrespondences++;
            pCurFrame->mvbOutlier[i] = false;

            Eigen::Matrix<double,2,1> obs;
            const cv::KeyPoint &kpUn = pCurFrame->mvKeysUn[i];
            obs << kpUn.pt.x, kpUn.pt.y;

            EdgeSO3ProjectXYZOnlyRotation * e = new EdgeSO3ProjectXYZOnlyRotation();
            e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0))); // dynamic_cast : type convert for class only
            e->setMeasurement(obs);
            const float invSigma2 = pCurFrame->mvInvLevelSigma2[kpUn.octave]; // uncertainty per pixel
            e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

            g2o::RobustKernelHuber * rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(deltaMono);

            e->fx = pCurFrame->fx;
            e->fy = pCurFrame->fy;
            e->cx = pCurFrame->cx;
            e->cy = pCurFrame->cy;

            cv::Mat Xw = pMP->GetWorldPos();
            e->Xw[0] = Xw.at<float>(0);
            e->Xw[1] = Xw.at<float>(1);
            e->Xw[2] = Xw.at<float>(2);

            Vector3d tcw = Converter::toVector3d(mtcw);
            e->tcw = tcw;
            
            optimizer.addEdge(e);
            vpEdgesFront.push_back(e);
            vnIndexEdgeFront.push_back(i);
        }
    }

    }

    if (nFrontInitialCorrespondences < 3)
    {
        cout << "nFrontInitialCorrespondences is : " << nFrontInitialCorrespondences << " less than 3";
        return 0;
    }
           
    
    // 7. iteration optimization
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const int its[4]={10,10,10,10};
    int nFrontBad = 0;
    for (size_t it = 0; it < 4; it++)
    {
        vRc->setEstimate(Converter::toQuaterniond(mRcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nFrontBad = 0;
        for (size_t i = 0, iend = vpEdgesFront.size(); i < iend; i++)
        {
            EdgeSO3ProjectXYZOnlyRotation * e = vpEdgesFront[i];
            const size_t idx = vnIndexEdgeFront[i];

            if (pCurFrame->mvbOutlier[idx])
                e->computeError();
            
            const float chi2 = e->chi2();
            if (chi2>chi2Mono[it])
            {
                pCurFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nFrontBad++;
            }
            else
            {
                pCurFrame->mvbOutlier[idx] = false;
                e->setLevel(0);
            }

            if (it==2)
                e->setRobustKernel(0); // means no kernel used
        }
    }
    
    // 8. get result
    VertexRotation * vRc_rec = static_cast<VertexRotation*>(optimizer.vertex(0));
    Eigen::Matrix3d R_rec = vRc_rec->estimate().matrix();
    mRcw = Converter::toCvMat(R_rec);
    cv::Mat Tcw_rec = cv::Mat::eye(4,4,CV_32F);
    mRcw.copyTo(Tcw_rec.rowRange(0,3).colRange(0,3));
    mtcw.copyTo(Tcw_rec.rowRange(0,3).col(3));   

    pCurFrame->SetPose(Tcw_rec);

    return nFrontInitialCorrespondences;

    return 0;
}


int Optimizer::poseOptimizationTranslation(Frame* pCurFrame, Frame* pRefFrame)
{
    // 0. graph model
    g2o::SparseOptimizer optimizer;

    // 1. linearSolver
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>(); // linear solver using dense cholesky decomposition
    // linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>(); using sparse cholesky solver from Eigen
    // linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolverX::PoseMatrixType>(); using CSparse
    // linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolverX::PoseMatrixType>(); basic solver 

    // 2. blockSolver
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    // 3. optimizer algorithm
    g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    // 4. set solver 
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false); // for output

    // 5. vertex
    g2o::VertexSE3Expmap * vSE3c = new g2o::VertexSE3Expmap();
    vSE3c->setId(0);
    vSE3c->setEstimate(Converter::toSE3Quat(pCurFrame->mTcw));
    vSE3c->setFixed(false);
    optimizer.addVertex(vSE3c);

    // g2o::VertexSE3Expmap * vSE3r = new g2o::VertexSE3Expmap();
    // vSE3r->setId(1);
    // vSE3r->setEstimate(Converter::toSE3Quat(pRefFrame->mTcw));
    // vSE3r->setFixed(false);
    // optimizer.addVertex(vSE3r); 

    // 6. edge
    const int Nf = pCurFrame->N;
    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesFront;
    vector<size_t> vnIndexEdgeFront;
    vpEdgesFront.reserve(Nf);
    vnIndexEdgeFront.reserve(Nf);
    int nFrontInitialCorrespondences = 0;

    const int Nb = pCurFrame->mvbBirdviewInliers.size();
    vector<EdgeSE3ProjectXYZ2XYZOnlyPoseQuat*> vpEdgesBird;
    vector<size_t> vnIndexEdgeBird;
    vpEdgesBird.reserve(Nb);
    vnIndexEdgeBird.reserve(Nb);
    int nBirdInitialCorrespondences = 0;
    
    const int Nd = 10; // pRefFrame->mvMeasurement_p.size(); // the number of these edge should be controlled
    vector<EdgeSE3ProjectDirect*> vpEdgesDirect;
    vector<size_t> vnIndexEdgeDirect;
    vector<bool> vOutlierDirect;
    vpEdgesDirect.reserve(Nd);
    vnIndexEdgeDirect.reserve(Nd);
    vOutlierDirect = vector<bool>(Nd,false);

    const float deltaMono = sqrt(5.991);
    {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);
    // front pts
    for (int i = 0; i < Nf; i++)
    {
        MapPoint* pMP = pCurFrame->mvpMapPoints[i];
        if (pMP)
        {
            nFrontInitialCorrespondences++;
            pCurFrame->mvbOutlier[i] = false;

            Eigen::Matrix<double,2,1> obs;
            const cv::KeyPoint &kpUn = pCurFrame->mvKeysUn[i];
            obs << kpUn.pt.x, kpUn.pt.y;

            g2o::EdgeSE3ProjectXYZOnlyPose * e = new g2o::EdgeSE3ProjectXYZOnlyPose();
            e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0))); // dynamic_cast : type convert for class only
            e->setMeasurement(obs);
            const float invSigma2 = pCurFrame->mvInvLevelSigma2[kpUn.octave]; // uncertainty per pixel
            e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

            g2o::RobustKernelHuber * rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(deltaMono);

            e->fx = pCurFrame->fx;
            e->fy = pCurFrame->fy;
            e->cx = pCurFrame->cx;
            e->cy = pCurFrame->cy;

            cv::Mat Xw = pMP->GetWorldPos();
            e->Xw[0] = Xw.at<float>(0);
            e->Xw[1] = Xw.at<float>(1);
            e->Xw[2] = Xw.at<float>(2);
            
            optimizer.addEdge(e);
            vpEdgesFront.push_back(e);
            vnIndexEdgeFront.push_back(i);
        }
    }

    // bird pts
    for (int i = 0; i < Nb; i++)
    {
        MapPointBird* pMP = pCurFrame->mvpMapPointsBird[i];
        if (pMP)
        {
            nBirdInitialCorrespondences++;
            pCurFrame->mvbBirdviewInliers[i] = true;
            
            Vector3d Xw = Converter::toVector3d(pMP->GetWorldPos());
            Vector3d Xc;
            cv::Point3f p = pCurFrame->mvKeysBirdCamXYZ[i];
            Xc << p.x, p.y, p.z;

            EdgeSE3ProjectXYZ2XYZOnlyPoseQuat * e = new EdgeSE3ProjectXYZ2XYZOnlyPoseQuat();
            e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e->Xc = Xc;
            float scale = 3.0;
            const float invSigma2 = pCurFrame->mvInvLevelSigma2[pCurFrame->mvKeysBird[i].octave]*scale; // set according to the uncertainty of pixel location
            e->setInformation(Eigen::Matrix3d::Identity()*invSigma2); 

            g2o::RobustKernelHuber * rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(deltaMono); // less, more linear

            e->Xw = Xw;

            optimizer.addEdge(e);
            vpEdgesBird.push_back(e);
            vnIndexEdgeBird.push_back(i);
        }
    }

    // bird direct
    for (int i = 0; i < Nd; i++)
    {
        EdgeSE3ProjectDirect * edge = new EdgeSE3ProjectDirect(
            Converter::toVector3d(pRefFrame->mvMeasurement_p[i]),
            pRefFrame->Tcb.rowRange(0,3).colRange(0,3),
            pRefFrame->Tcb.rowRange(0,3).col(3),
            pRefFrame->Rro, pRefFrame->tro,
            pRefFrame->Ror, pRefFrame->tor,
            pCurFrame->mBirdviewContour
        );
        edge->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        edge->setMeasurement(pRefFrame->mvMeasurement_g[i]);
        edge->setInformation(Eigen::Matrix<double,1,1>::Identity()*5);

        g2o::RobustKernelHuber * rk = new g2o::RobustKernelHuber;
        edge->setRobustKernel(rk);
        rk->setDelta(deltaMono);

        optimizer.addEdge(edge);
        vpEdgesDirect.push_back(edge);
        vnIndexEdgeDirect.push_back(i);
    }
        
    }

    // cout << "the feature of front: ... " << vnIndexEdgeFront.size() << endl;
    // cout << "the feature of bird : ... " << vnIndexEdgeBird.size() << endl;
    // cout << "the directio of bird: ... " << vnIndexEdgeDirect.size() << endl << endl;

    if (nFrontInitialCorrespondences < 3 && nBirdInitialCorrespondences < 3)
        return 0;

    
    
    // 7. iteration optimization
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const int its[4]={10,10,10,10};
    int nFrontBad = 0;
    int nBirdBad = 0;
    int nDirectBad = 0;

    for (size_t it = 0; it < 4; it++)
    {
        vSE3c->setEstimate(Converter::toSE3Quat(pCurFrame->mTcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nFrontBad = 0;
        for (size_t i = 0, iend = vpEdgesFront.size(); i < iend; i++)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose * e = vpEdgesFront[i];
            const size_t idx = vnIndexEdgeFront[i];

            if (pCurFrame->mvbOutlier[idx])
                e->computeError();
            
            const float chi2 = e->chi2();
            if (chi2>chi2Mono[it])
            {
                pCurFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nFrontBad++;
            }
            else
            {
                pCurFrame->mvbOutlier[idx] = false;
                e->setLevel(0);
            }

            if (it==2)
                e->setRobustKernel(0); // means no kernel used
        }

        nBirdBad = 0;
        for (size_t i = 0, iend = vpEdgesBird.size(); i < iend; i++)
        {
            EdgeSE3ProjectXYZ2XYZOnlyPoseQuat * e = vpEdgesBird[i];
            const size_t idx = vnIndexEdgeBird[i];

            if (!pCurFrame->mvbBirdviewInliers[idx])
                e->computeError();
            
            const float chi2 = e->chi2();
            if (chi2>chi2Mono[it])
            {
                pCurFrame->mvbBirdviewInliers[idx] = false;
                e->setLevel(1);
                nBirdBad++;
            }
            else
            {
                pCurFrame->mvbBirdviewInliers[idx] = true;
                e->setLevel(0);
            }
            
            if (it==2)
                e->setRobustKernel(0);
        }

        nDirectBad = 0;
        for (size_t i = 0, iend = vpEdgesDirect.size(); i < iend; i++)
        {
            EdgeSE3ProjectDirect * edge = vpEdgesDirect[i];
            const size_t idx = vnIndexEdgeDirect[i];

            if (!vOutlierDirect[idx])
                edge->computeError();
            
            const float chi2 = edge->chi2();
            if (chi2>chi2Mono[it])
            {
                vOutlierDirect[idx] = true;
                edge->setLevel(1);
                nDirectBad++;
            }
            else
            {
                vOutlierDirect[idx] = false;
                edge->setLevel(0);
            }
        }
        
        if (optimizer.edges().size() < 10)
            break;
    }
    
    // 8. get result
    g2o::VertexSE3Expmap * vSE3c_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3c_recov = vSE3c_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3c_recov);
    if (1)
    {
        pCurFrame->SetPose(pose);
    }

    return nFrontInitialCorrespondences-nFrontBad;
}

}  // namespace ORB_SLAM2