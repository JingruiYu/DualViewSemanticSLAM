/*************************************************************************
	> File Name: semanticDB.cpp
	> Author: 
	> Mail: 
	> Created Time: 2020年06月17日 星期三 10时52分36秒
 ************************************************************************/

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
#include <ctime>
#include <climits>

#include<opencv2/opencv.hpp>

#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

using namespace std;
using namespace g2o;

struct Measurement
{
    Measurement ( Eigen::Vector3d p, float g ) : pos_world ( p ), grayscale ( g ) {}
    Eigen::Vector3d pos_world;
    float grayscale;
};

void LoadDataset(const string &strFile, vector<string> &vstrImageFilenames, vector<string> &vstrBirdviewFilenames, 
                vector<string> &vstrBirdviewMaskFilenames, vector<cv::Vec3d> &vodomPose, vector<double> &vTimestamps);

inline Eigen::Vector3d project2Dto3D ( int x, int y, float fx, float fy, float cx, float cy )
{
    float zz = 1;
    float xx = zz* ( x-cx ) /fx;
    float yy = zz* ( y-cy ) /fy;
    return Eigen::Vector3d ( xx, yy, zz );
}

inline Eigen::Vector2d project3Dto2D ( float x, float y, float z, float fx, float fy, float cx, float cy )
{
    float u = fx*x/z+cx;
    float v = fy*y/z+cy;
    return Eigen::Vector2d ( u,v );
}

// 直接法估计位姿
// 输入：测量值（空间点的灰度），新的灰度图，相机内参； 输出：相机位姿
// 返回：true为成功，false失败
bool poseEstimationDirect ( const vector<Measurement>& measurements, const cv::Mat& color, Eigen::Matrix3f& intrinsics, Eigen::Isometry3d& Tcw );


// project a 3d point into an image plane, the error is photometric error
// an unary edge with one vertex SE3Expmap (the pose of camera)
class EdgeSE3ProjectDirect: public BaseUnaryEdge< 1, double, VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectDirect() {}

    EdgeSE3ProjectDirect ( Eigen::Vector3d point, float fx, float fy, float cx, float cy, const cv::Mat& image )
        : x_world_ ( point ), fx_ ( fx ), fy_ ( fy ), cx_ ( cx ), cy_ ( cy )
    {
        image_ = image.clone();
    }

    virtual void computeError()
    {
        const VertexSE3Expmap* v  =static_cast<const VertexSE3Expmap*> ( _vertices[0] );
        Eigen::Vector3d x_local = v->estimate().map ( x_world_ );
        float x = x_local[0]*fx_/x_local[2] + cx_;
        float y = x_local[1]*fy_/x_local[2] + cy_;
        // check x,y is in the image
        if ( x-4<0 || ( x+4 ) >image_.cols || ( y-4 ) <0 || ( y+4 ) >image_.rows )
        {
            _error ( 0,0 ) = 0.0;
            this->setLevel ( 1 );
        }
        else
        {
            _error ( 0,0 ) = getPixelValue ( x,y ) - _measurement;
        }
    }

    // plus in manifold
    virtual void linearizeOplus( )
    {
        if ( level() == 1 )
        {
            _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
            return;
        }
        VertexSE3Expmap* vtx = static_cast<VertexSE3Expmap*> ( _vertices[0] );
        Eigen::Vector3d xyz_trans = vtx->estimate().map ( x_world_ );   // q in book

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0/xyz_trans[2];
        double invz_2 = invz*invz;

        float u = x*fx_*invz + cx_;
        float v = y*fy_*invz + cy_;

        // jacobian from se3 to u,v
        // NOTE that in g2o the Lie algebra is (\omega, \epsilon), where \omega is so(3) and \epsilon the translation
        Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;

        jacobian_uv_ksai ( 0,0 ) = - x*y*invz_2 *fx_;
        jacobian_uv_ksai ( 0,1 ) = ( 1+ ( x*x*invz_2 ) ) *fx_;
        jacobian_uv_ksai ( 0,2 ) = - y*invz *fx_;
        jacobian_uv_ksai ( 0,3 ) = invz *fx_;
        jacobian_uv_ksai ( 0,4 ) = 0;
        jacobian_uv_ksai ( 0,5 ) = -x*invz_2 *fx_;

        jacobian_uv_ksai ( 1,0 ) = - ( 1+y*y*invz_2 ) *fy_;
        jacobian_uv_ksai ( 1,1 ) = x*y*invz_2 *fy_;
        jacobian_uv_ksai ( 1,2 ) = x*invz *fy_;
        jacobian_uv_ksai ( 1,3 ) = 0;
        jacobian_uv_ksai ( 1,4 ) = invz *fy_;
        jacobian_uv_ksai ( 1,5 ) = -y*invz_2 *fy_;

        Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;

        jacobian_pixel_uv ( 0,0 ) = ( getPixelValue ( u+1,v )-getPixelValue ( u-1,v ) ) /2;
        jacobian_pixel_uv ( 0,1 ) = ( getPixelValue ( u,v+1 )-getPixelValue ( u,v-1 ) ) /2;

        _jacobianOplusXi = jacobian_pixel_uv*jacobian_uv_ksai;
    }

    // dummy read and write functions because we don't care...
    virtual bool read ( std::istream& in ) {}
    virtual bool write ( std::ostream& out ) const {}

protected:
    // get a gray scale value from reference image (bilinear interpolated)
    inline float getPixelValue ( float x, float y )
    {
        int xx = int(x);
        int yy = int(y);
        float colorscale = float (image_.at<cv::Vec3b>(y,x)[0] + image_.at<cv::Vec3b>(y,x)[1] + image_.at<cv::Vec3b>(y,x)[2]);

        return colorscale;
    }
public:
    Eigen::Vector3d x_world_;   // 3D point in world frame
    float cx_=0, cy_=0, fx_=0, fy_=0; // Camera intrinsics
    cv::Mat image_;    // reference image
};


int main(int argc, char const *argv[])
{
	/******* Get Data *******/
	if ( argc != 2)
	{
		cout<<"usage: demo path_to_dataset"<<endl;
        return 1;
	}

	vector<string> vstrImageFilenames;
    vector<string> vstrBirdviewFilenames;
    vector<string> vstrBirdviewMaskFilenames;
    vector<double> vTimestamps;
    vector<cv::Vec3d> vodomPose;
	string strFile = string(argv[1])+"/associate.txt";
	LoadDataset(strFile, vstrImageFilenames, vstrBirdviewFilenames, vstrBirdviewMaskFilenames, vodomPose, vTimestamps);

	Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();
    cv::Mat prev_gray, prev_color, color, gray;
    vector<Measurement> measurements;
	for (int i = 0; i < vstrBirdviewFilenames.size(); i++)
	{
		color = cv::imread(string(argv[1])+"/"+vstrBirdviewFilenames[i], CV_LOAD_IMAGE_UNCHANGED);
        if ( color.data==nullptr)
            continue;

        double correction = 1.7;
        double pixel2meter = 0.03984*correction;
        double rear_axle_to_center = 1.393;
        float cx = color.cols * 0.5;
        float cy = color.rows * 0.5 + rear_axle_to_center / pixel2meter;
        float fx = pixel2meter * -1.0;
        float fy = pixel2meter * -1.0;
        Eigen::Matrix3f K;
        K<<fx,0.f,cx,0.f,fy,cy,0.f,0.f,1.0f;

		// cv::cvtColor ( color, gray, cv::COLOR_BGR2GRAY );
		/******* Get KeyPoint *******/
		if ( i == 0 )
        {
            // select the pixels with high gradiants 
            for ( int x=10; x<color.cols-10; x++ )
                for ( int y=10; y<color.rows-10; y++ )
                {
                    // Eigen::Vector2d delta (
                    //     gray.ptr<uchar>(y)[x+1] - gray.ptr<uchar>(y)[x-1], 
                    //     gray.ptr<uchar>(y+1)[x] - gray.ptr<uchar>(y-1)[x]
                    // );
                    
                    Eigen::Vector2d delta (
                        (color.at<cv::Vec3b>(y,x+1)[0] + color.at<cv::Vec3b>(y,x+1)[1] + color.at<cv::Vec3b>(y,x+1)[2])
                      - (color.at<cv::Vec3b>(y,x-1)[0] + color.at<cv::Vec3b>(y,x-1)[1] + color.at<cv::Vec3b>(y,x-1)[2]), 
                        (color.at<cv::Vec3b>(y+1,x)[0] + color.at<cv::Vec3b>(y+1,x)[1] + color.at<cv::Vec3b>(y+1,x)[2])
                      - (color.at<cv::Vec3b>(y-1,x)[0] + color.at<cv::Vec3b>(y-1,x)[1] + color.at<cv::Vec3b>(y-1,x)[2])
                    );
                    if ( delta.norm() < 50 )
                        continue;
                    
                    Eigen::Vector3d p3d = project2Dto3D ( x, y, fx, fy, cx, cy );
                    // float grayscale = float ( gray.ptr<uchar> (y) [x] );
                    float grayscale = float (color.at<cv::Vec3b>(y,x)[0] + color.at<cv::Vec3b>(y,x)[1] + color.at<cv::Vec3b>(y,x)[2]);
                    measurements.push_back ( Measurement ( p3d, grayscale ) );	
                }
            prev_color = color.clone();
            cout<<"add total "<<measurements.size()<<" measurements."<<endl;
            continue;
        }

		/******* Cal Pose *******/
		chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        poseEstimationDirect( measurements, color, K, Tcw ); // TODO
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
        cout<<"direct method costs time: "<<time_used.count() <<" seconds."<<endl;
        cout<<"Tcw="<<Tcw.matrix() <<endl;

		/******* Drawer Points *******/
		cv::Mat img_show ( color.rows*2, color.cols, CV_8UC3 );
        prev_color.copyTo ( img_show ( cv::Rect ( 0,0,color.cols, color.rows ) ) );
        color.copyTo ( img_show ( cv::Rect ( 0,color.rows,color.cols, color.rows ) ) );
        for ( Measurement m:measurements )
        {
            if ( rand() > RAND_MAX/5 )  //TODO
                continue;
            Eigen::Vector3d p = m.pos_world;
            Eigen::Vector2d pixel_prev = project3Dto2D ( p ( 0,0 ), p ( 1,0 ), p ( 2,0 ), fx, fy, cx, cy );
            Eigen::Vector3d p2 = Tcw*m.pos_world;
            Eigen::Vector2d pixel_now = project3Dto2D ( p2 ( 0,0 ), p2 ( 1,0 ), p2 ( 2,0 ), fx, fy, cx, cy );
            if ( pixel_now(0,0)<0 || pixel_now(0,0)>=color.cols || pixel_now(1,0)<0 || pixel_now(1,0)>=color.rows )
                continue;

            float b = 0;
            float g = 250;
            float r = 0;
            img_show.ptr<uchar>( pixel_prev(1,0) )[int(pixel_prev(0,0))*3] = b;
            img_show.ptr<uchar>( pixel_prev(1,0) )[int(pixel_prev(0,0))*3+1] = g;
            img_show.ptr<uchar>( pixel_prev(1,0) )[int(pixel_prev(0,0))*3+2] = r;
            
            img_show.ptr<uchar>( pixel_now(1,0)+color.rows )[int(pixel_now(0,0))*3] = b;
            img_show.ptr<uchar>( pixel_now(1,0)+color.rows )[int(pixel_now(0,0))*3+1] = g;
            img_show.ptr<uchar>( pixel_now(1,0)+color.rows )[int(pixel_now(0,0))*3+2] = r;

            cv::circle ( img_show, cv::Point2d ( pixel_prev ( 0,0 ), pixel_prev ( 1,0 ) ), 4, cv::Scalar ( b,g,r ), 2 );
            cv::circle ( img_show, cv::Point2d ( pixel_now ( 0,0 ), pixel_now ( 1,0 ) + color.rows ), 4, cv::Scalar ( b,g,r ), 2 );
        }
        cv::imshow ( "result", img_show );
        cv::waitKey ( 0 );
	}
	return 0;
}



void LoadDataset(const string &strFile, vector<string> &vstrImageFilenames, vector<string> &vstrBirdviewFilenames, 
                vector<string> &vstrBirdviewMaskFilenames, vector<cv::Vec3d> &vodomPose, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            double x,y,theta;
            string image;
            ss >> t;
            vTimestamps.push_back(t);
            ss>>x>>y>>theta;
            vodomPose.push_back(cv::Vec3d(x,y,theta));
            ss >> image;
            vstrImageFilenames.push_back("image/"+image);
            vstrBirdviewFilenames.push_back("contour/"+image);
            vstrBirdviewMaskFilenames.push_back("mask/"+image);
        }
    }
    // double t0=vTimestamps[0];
    // for_each(vTimestamps.begin(),vTimestamps.end(),[t0](double &t){t-=t0;});
}

bool poseEstimationDirect ( const vector< Measurement >& measurements, const cv::Mat& color, Eigen::Matrix3f& K, Eigen::Isometry3d& Tcw )
{
    // 初始化g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> DirectBlock;  // 求解的向量是6＊1的
    DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense< DirectBlock::PoseMatrixType > ();
    DirectBlock* solver_ptr = new DirectBlock ( linearSolver );
    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr ); // G-N
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr ); // L-M
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );
    optimizer.setVerbose( true );

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setEstimate ( g2o::SE3Quat ( Tcw.rotation(), Tcw.translation() ) );
    pose->setId ( 0 );
    optimizer.addVertex ( pose );

    // 添加边
    int id=1;
    vector<EdgeSE3ProjectDirect*> edges; //test
    for ( Measurement m: measurements )
    {
        EdgeSE3ProjectDirect* edge = new EdgeSE3ProjectDirect (
            m.pos_world,
            K ( 0,0 ), K ( 1,1 ), K ( 0,2 ), K ( 1,2 ), color
        );
        edge->setVertex ( 0, pose );
        edge->setMeasurement ( m.grayscale );
        edge->setInformation ( Eigen::Matrix<double,1,1>::Identity() );
        edge->setId ( id++ );
        optimizer.addEdge ( edge );
        edges.push_back( edge ); //test
    }
    cout<<"edges in graph: "<<optimizer.edges().size() <<endl;
    optimizer.initializeOptimization();
    optimizer.optimize ( 30 );
    Tcw = pose->estimate();

    //test
    double errSum = 0.0;
    for ( auto e:edges )
    {
        e->computeError();
        errSum += e->chi2();
    }
    cout << "sum of error is " << errSum << endl; 
}
