#ifndef ODOM_G2O_TYPE_H_
#define ODOM_G2O_TYPE_H_

// #include "Frame.h"
#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"
#include "Thirdparty/g2o/g2o/core/base_binary_edge.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include "Converter.h"

#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{

using namespace Eigen;

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

class VertexSE3Quat : public g2o::BaseVertex<6, g2o::SE3Quat>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	bool read(std::istream& is){return false;}
	bool write(std::ostream& os) const{return false;}
	virtual void setToOriginImpl()
	{
		_estimate = g2o::SE3Quat();
	}
	virtual void oplusImpl(const double* update_)
	{
		Eigen::Map<const Vector6d> update(update_);
		setEstimate(g2o::SE3Quat::exp(update)*estimate());
	}
};

class EdgeSE3ProjectXYZOnlyPoseQuat: public g2o::BaseUnaryEdge<2, Vector2d, VertexSE3Quat>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	EdgeSE3ProjectXYZOnlyPoseQuat(){}
	bool read(std::istream& is){return false;}
	bool write(std::ostream& os) const{return false;}

	void computeError()
	{
		const VertexSE3Quat *v = static_cast<const VertexSE3Quat*>(_vertices[0]);
		Vector3d p = v->estimate().map(Xw);
		_error = Vector2d(_measurement)-cam_project(p);
	}

	bool isDepthPositive()
	{
		const VertexSE3Quat *v = static_cast<const VertexSE3Quat*>(_vertices[0]);
		return (v->estimate().map(Xw))(2)>0.0;
	}

	virtual void linearizeOplus();

	Vector2d cam_project(const Vector3d & trans_xyz) const;
	Matrix3d skew(Vector3d phi);

	Vector3d Xw;
	double fx, fy, cx, cy;
};

class EdgeSE3ProjectXYZ2XYZOnlyPoseQuat: public g2o::BaseUnaryEdge<3, Vector3d, VertexSE3Quat>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	EdgeSE3ProjectXYZ2XYZOnlyPoseQuat(){}
	bool read(std::istream& is){return false;}
	bool write(std::ostream& os) const{return false;}

	void computeError()
	{
		const VertexSE3Quat *v = static_cast<const VertexSE3Quat*>(_vertices[0]);
		Vector3d p = v->estimate().map(Xw);
		_error = Xc-p;
	}

	virtual void linearizeOplus();

	Matrix3d skew(Vector3d phi);

	Vector3d Xw,Xc;
};

class EdgePointTransformSE3Quat :public g2o::BaseBinaryEdge<3, Vector3d, VertexSE3Quat, VertexSE3Quat>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	virtual bool read(std::istream& is) { return false; }
	virtual bool write(std::ostream& os) const { return false; }
	virtual void computeError()
	{
		//pose1:reference frame; pose2:current frame
		g2o::SE3Quat pose1 = (static_cast<VertexSE3Quat*> (_vertices[0]))->estimate();
		g2o::SE3Quat pose2 = (static_cast<VertexSE3Quat*> (_vertices[1]))->estimate();
		g2o::SE3Quat pose21 = pose2 * pose1.inverse();
		//_error = pc2 - (pose21.rotation_matrix()*pc1+pose21.translation());
		_error = pc2 - pose21 * pc1;
	}
	virtual void linearizeOplus();

	Matrix3d skew(Vector3d phi);

	Vector3d pc1,pc2;
};

class EdgeSE3ProjectXYZ2UVQuat :public g2o::BaseBinaryEdge<2, Vector2d, g2o::VertexSBAPointXYZ, VertexSE3Quat>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	virtual bool read(std::istream& is) { return false; }
	virtual bool write(std::ostream& os) const { return false; }
	virtual void computeError()
	{
		g2o::SE3Quat pose = (static_cast<VertexSE3Quat*> (_vertices[1]))->estimate();
		Vector3d point = (static_cast<g2o::VertexSBAPointXYZ*> (_vertices[0]))->estimate();

		Vector3d p = pose * point;
		_error = Vector2d(_measurement) - cam_project(p);
	}
	virtual void linearizeOplus();

	Matrix3d skew(Vector3d phi);
	Vector2d cam_project(const Vector3d & trans_xyz) const;
	bool isDepthPositive();
	
	double fx, fy, cx, cy;
};

class EdgeSE3ProjectXYZ2XYZQuat :public g2o::BaseBinaryEdge<3, Vector3d, g2o::VertexSBAPointXYZ, VertexSE3Quat>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	virtual bool read(std::istream& is) { return false; }
	virtual bool write(std::ostream& os) const { return false; }
	virtual void computeError()
	{
		g2o::SE3Quat pose = (static_cast<VertexSE3Quat*> (_vertices[1]))->estimate();
		Vector3d point = (static_cast<g2o::VertexSBAPointXYZ*> (_vertices[0]))->estimate();

		Vector3d p = pose * point;
		_error = Vector3d(_measurement) - p;
	}
	virtual void linearizeOplus();

	Matrix3d skew(Vector3d phi);
};

//templete params: BaseBinaryEdge<ErrorDimension,MeasurementType,Vertex1,Vertex2>
class EdgeSE3Quat : public g2o::BaseBinaryEdge<6, g2o::SE3Quat, VertexSE3Quat, VertexSE3Quat>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	bool read(std::istream& is){return false;}
	bool write(std::ostream& os) const{return false;}
	virtual void computeError()
	{
		g2o::SE3Quat v1 = (static_cast<VertexSE3Quat*> (_vertices[0]))->estimate();
		g2o::SE3Quat v2 = (static_cast<VertexSE3Quat*> (_vertices[1]))->estimate();
		_error = (_measurement.inverse()*v1*v2.inverse()).log();
	}
	virtual void linearizeOplus();
	Matrix6d JRInv(Vector6d e);
	Matrix3d skew(Vector3d phi);
};


// project a 3d point into an image plane, the error is photometric error
// an unary edge with one vertex SE3Expmap (the pose of camera)
class EdgeSE3ProjectDirect: public g2o::BaseUnaryEdge< 1, double, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectDirect() {}

    EdgeSE3ProjectDirect ( Eigen::Vector3d point, const cv::Mat& Rcb, const cv::Mat& tcb, const cv::Mat& Rro, const cv::Mat& tro, const cv::Mat& Ror, const cv::Mat& tor, const cv::Mat& image )
        : x_world_ ( point )
    {
		Rcb_ = Rcb.clone();
		tcb_ = tcb.clone();
		Rro_ = Rro.clone();
		tro_ = tro.clone();
        Ror_ = Ror.clone();
		tor_ = tor.clone();
		image_ = image.clone();
    }

    virtual void computeError()
    {
        const g2o::VertexSE3Expmap* v  =static_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
        Eigen::Vector3d x_local = v->estimate().map ( x_world_ );
        cv::Mat p_c = Converter::toCvMat(x_local);
		cv::Mat uv = Ror_ * Rcb_.t() * ( p_c - tcb_) + tor_; 
        float x = uv.at<float>(0);
        float y = uv.at<float>(1);

        // check x,y is in the image
        if ( x-4<0 || ( x+4 ) >image_.cols || ( y-4 ) <0 || ( y+4 ) >image_.rows || std::isnan(x) || std::isnan(y) )
        {
            _error ( 0,0 ) = 0.0 - _measurement;
            this->setLevel ( 1 );
        }
        else
        {
			float tmpPixel = getPixelValue ( x,y );
			if (tmpPixel < 10)
                this->setLevel ( 1 );

			_error ( 0,0 ) = tmpPixel - _measurement;
        }
    }

	virtual void linearizeOplus();

    // dummy read and write functions because we don't care...
    virtual bool read ( std::istream& in ) {return false;}
    virtual bool write ( std::ostream& out ) const {return false;}

protected:
    // get a gray scale value from reference image (bilinear interpolated)
    inline float getPixelValue ( float x, float y )
    {
        int col = int(x);
        int row = int(y);
		float colorscale = 0.0;
		if (!image_.empty() && row > 2 && row < image_.rows-3 && col > 2 && col < image_.cols - 3)
		{
			// colorscale = float (image_.at<cv::Vec3b>(yy,xx)[0] + image_.at<cv::Vec3b>(yy,xx)[1] + image_.at<cv::Vec3b>(yy,xx)[2]);
			colorscale = float (image_.at<cv::Vec3b>(row,col)[0] + image_.at<cv::Vec3b>(row-1,col+1)[0]
								+ image_.at<cv::Vec3b>(row-2,col)[0] + image_.at<cv::Vec3b>(row+2,col)[0] 
								+ image_.at<cv::Vec3b>(row,col-2)[0] + image_.at<cv::Vec3b>(row,col+2)[0] 
								+ image_.at<cv::Vec3b>(row+1,col+1)[0] + image_.at<cv::Vec3b>(row-1,col-1)[0]);
                 
		}

        return colorscale;
    }
public:
    Eigen::Vector3d x_world_;   // 3D point in world frame
    cv::Mat Rro_,tro_,Ror_,tor_,Rcb_,tcb_;
	cv::Mat image_;    // reference image
};


}  //namespace ORB_SLAM2

#endif  //ODOM_G2O_TYPE_H_
