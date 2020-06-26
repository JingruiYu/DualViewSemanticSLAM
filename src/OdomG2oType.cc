#include"OdomG2oType.h"

namespace ORB_SLAM2
{

void EdgeSE3ProjectXYZOnlyPoseQuat::linearizeOplus() 
{
	const VertexSE3Quat *v = static_cast<const VertexSE3Quat*>(_vertices[0]);
	Vector3d p = v->estimate().map(Xw);

  	double x = p[0], y = p[1], z = p[2];
	double z2 = z * z;

	Matrix<double, 2, 3> jacobian_e_p;
	jacobian_e_p << fx / z, 0, -fx * x / z2,
		0, fy / z, -fy * y / z2;
	jacobian_e_p = -jacobian_e_p;

	Matrix<double, 3, 6> jacobian_p_ksi;
	jacobian_p_ksi << -skew(p), Matrix3d::Identity();

	_jacobianOplusXi = jacobian_e_p * jacobian_p_ksi;
}
Vector2d EdgeSE3ProjectXYZOnlyPoseQuat::cam_project(const Vector3d & trans_xyz) const
{
	Vector2d res;
	res[0] = fx * trans_xyz[0] / trans_xyz[2] + cx;
	res[1] = fy * trans_xyz[1] / trans_xyz[2] + cy;
	return res;
}
Matrix3d EdgeSE3ProjectXYZOnlyPoseQuat::skew(Vector3d phi)
{
	Matrix3d Phi;
	Phi << 0, -phi[2], phi[1],
		phi[2], 0, -phi[0],
		-phi[1], phi[0], 0;
	return Phi;
}


void EdgeSE3ProjectXYZ2XYZOnlyPoseQuat::linearizeOplus() 
{
	const VertexSE3Quat *v = static_cast<const VertexSE3Quat*>(_vertices[0]);
	Vector3d p = v->estimate().map(Xw);

	Matrix<double, 3, 6> jacobian_p_ksi;
	jacobian_p_ksi << -skew(p), Matrix3d::Identity();

	_jacobianOplusXi = -jacobian_p_ksi;
}
Matrix3d EdgeSE3ProjectXYZ2XYZOnlyPoseQuat::skew(Vector3d phi)
{
	Matrix3d Phi;
	Phi << 0, -phi[2], phi[1],
		phi[2], 0, -phi[0],
		-phi[1], phi[0], 0;
	return Phi;
}


Matrix3d EdgePointTransformSE3Quat::skew(Vector3d phi)
{
	Matrix3d Phi;
	Phi << 0, -phi[2], phi[1],
		phi[2], 0, -phi[0],
		-phi[1], phi[0], 0;
	return Phi;
}
void EdgePointTransformSE3Quat::linearizeOplus()
{
	g2o::SE3Quat pose1 = (static_cast<VertexSE3Quat*> (_vertices[0]))->estimate();
	g2o::SE3Quat pose2 = (static_cast<VertexSE3Quat*> (_vertices[1]))->estimate();
	g2o::SE3Quat pose21 = pose2 * pose1.inverse();
	// _error = pc2 - (pose21.rotation_matrix()*pc1+pose21.translation());
	//Vector3d pc2p = pose21.rotation_matrix()*pc1+pose21.translation();
	Vector3d pc2p = pose21 * pc1;

	Matrix<double, 3, 6> jacobian_p_ksi2;
	jacobian_p_ksi2 << -skew(pc2p), Matrix3d::Identity();

	Matrix<double, 3, 6> jacobian_p_ksi1;
	jacobian_p_ksi1 << -skew(pc1), Matrix3d::Identity();

	_jacobianOplusXi = pose21.rotation().toRotationMatrix()*jacobian_p_ksi1;
	_jacobianOplusXj = -jacobian_p_ksi2;
}


void EdgeSE3ProjectXYZ2UVQuat::linearizeOplus()
{
	g2o::SE3Quat pose = (static_cast<VertexSE3Quat*> (_vertices[1]))->estimate();
	Vector3d point = (static_cast<g2o::VertexSBAPointXYZ*> (_vertices[0]))->estimate();

	Vector3d p = pose * point;

	double x = p[0], y = p[1], z = p[2];
	double z2 = z * z;

	Matrix<double, 2, 3> jacobian_e_p;
	jacobian_e_p << fx / z, 0, -fx * x / z2,
		0, fy / z, -fy * y / z2;
	jacobian_e_p = -jacobian_e_p;

	Matrix<double, 3, 6> jacobian_p_ksi;
	jacobian_p_ksi <<-skew(p), Matrix3d::Identity();

	_jacobianOplusXj = jacobian_e_p * jacobian_p_ksi;
	_jacobianOplusXi = jacobian_e_p * pose.rotation().toRotationMatrix();
}
Matrix3d EdgeSE3ProjectXYZ2UVQuat::skew(Vector3d phi)
{
	Matrix3d Phi;
	Phi << 0, -phi[2], phi[1],
		phi[2], 0, -phi[0],
		-phi[1], phi[0], 0;
	return Phi;
}
Vector2d EdgeSE3ProjectXYZ2UVQuat::cam_project(const Vector3d & trans_xyz) const
{
	Vector2d res;
	res[0] = fx * trans_xyz[0] / trans_xyz[2] + cx;
	res[1] = fy * trans_xyz[1] / trans_xyz[2] + cy;
	return res;
}
bool EdgeSE3ProjectXYZ2UVQuat::isDepthPositive()
{
	g2o::SE3Quat pose = (static_cast<VertexSE3Quat*> (_vertices[1]))->estimate();
	Vector3d point = (static_cast<g2o::VertexSBAPointXYZ*> (_vertices[0]))->estimate();

	Vector3d p = pose * point;

	return (p[2] > 0.0);
}



void EdgeSE3ProjectXYZ2XYZQuat::linearizeOplus()
{
	g2o::SE3Quat pose = (static_cast<VertexSE3Quat*> (_vertices[1]))->estimate();
	Vector3d point = (static_cast<g2o::VertexSBAPointXYZ*> (_vertices[0]))->estimate();

	Vector3d p = pose * point;

	Matrix<double, 3, 6> jacobian_p_ksi;
	jacobian_p_ksi <<-skew(p), Matrix3d::Identity();

	_jacobianOplusXj = -jacobian_p_ksi;
	_jacobianOplusXi = -pose.rotation().toRotationMatrix();
}
Matrix3d EdgeSE3ProjectXYZ2XYZQuat::skew(Vector3d phi)
{
	Matrix3d Phi;
	Phi << 0, -phi[2], phi[1],
		phi[2], 0, -phi[0],
		-phi[1], phi[0], 0;
	return Phi;
}


Matrix6d EdgeSE3Quat::JRInv(Vector6d e)
{
	Matrix6d J;
	J.block(0, 0, 3, 3) = skew(e.head(3));
	J.block(3, 3, 3, 3) = skew(e.head(3));
	J.block(3, 0, 3, 3) = skew(e.tail(3));
	J.block(0, 3, 3, 3) = Eigen::Matrix3d::Zero();
	
	J = 0.5 * J + Matrix6d::Identity();
	return J;
}
void EdgeSE3Quat::linearizeOplus()
{
	g2o::SE3Quat v1 = (static_cast<VertexSE3Quat*> (_vertices[0]))->estimate();
	g2o::SE3Quat v2 = (static_cast<VertexSE3Quat*> (_vertices[1]))->estimate();
//  Sophus::SE3 e = _measurement.inverse() * v1.inverse() * v2;
	g2o::SE3Quat e = _measurement.inverse() * v1 * v2.inverse();
	Matrix6d J = JRInv(e.log());

//   _jacobianOplusXi = -J * v2.inverse().Adj();
//   _jacobianOplusXj = J * v2.inverse().Adj();

_jacobianOplusXi = J*v2.adj()*v1.inverse().adj();
_jacobianOplusXj = -J;
}
Matrix3d EdgeSE3Quat::skew(Vector3d phi)
{
	Matrix3d Phi;
	Phi << 0, -phi[2], phi[1],
		phi[2], 0, -phi[0],
		-phi[1], phi[0], 0;
	return Phi;
}


// plus in manifold
void EdgeSE3ProjectDirect::linearizeOplus()
{
	if ( level() == 1 )
	{
		_jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
		return;
	}
	g2o::VertexSE3Expmap* vtx = static_cast<g2o::VertexSE3Expmap*> ( _vertices[0] );
	Eigen::Vector3d xyz_trans = vtx->estimate().map ( x_world_ );   // q in book

	cv::Mat p_c = Converter::toCvMat(xyz_trans);
	cv::Mat uv = Ror_ * Rcb_.t() * ( p_c - tcb_) + tor_; 

	double x = xyz_trans[0];
	double y = xyz_trans[1];
	double z = xyz_trans[2];
	
	double u = uv.at<float>(0);
	double v = uv.at<float>(1);
	
	double f = Ror_.at<float>(0,1);
	double f1 = f * Rcb_.at<float>(1,0);
	double f2 = f * Rcb_.at<float>(1,1);
	double f3 = f * Rcb_.at<float>(1,2);
	double f4 = f * Rcb_.at<float>(0,0);
	double f5 = f * Rcb_.at<float>(0,1);
	double f6 = f * Rcb_.at<float>(0,2);

	// jacobian from se3 to u,v
	// NOTE that in g2o the Lie algebra is (\omega, \epsilon), where \omega is so(3) and \epsilon the translation
	Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;

	jacobian_uv_ksai ( 0,0 ) = -f2*z+f3*y;
	jacobian_uv_ksai ( 0,1 ) = f1*z-f3*x;
	jacobian_uv_ksai ( 0,2 ) = -f1*y+f2*x;
	jacobian_uv_ksai ( 0,3 ) = f1; 
	jacobian_uv_ksai ( 0,4 ) = f2;
	jacobian_uv_ksai ( 0,5 ) = f3;

	jacobian_uv_ksai ( 1,0 ) = -f5*z+f6*y;
	jacobian_uv_ksai ( 1,1 ) = f4*z-f6*x;
	jacobian_uv_ksai ( 1,2 ) = -f4*y+f5*x;
	jacobian_uv_ksai ( 1,3 ) = f4;
	jacobian_uv_ksai ( 1,4 ) = f5;
	jacobian_uv_ksai ( 1,5 ) = f6;

	Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;

	jacobian_pixel_uv ( 0,0 ) = ( getPixelValue ( u+1,v )-getPixelValue ( u-1,v ) ) /2;
	jacobian_pixel_uv ( 0,1 ) = ( getPixelValue ( u,v+1 )-getPixelValue ( u,v-1 ) ) /2;

	_jacobianOplusXi = jacobian_pixel_uv*jacobian_uv_ksai;
}


}  //namespace ORB_SLAM2
