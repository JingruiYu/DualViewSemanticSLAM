#include"OdomG2oType.h"

namespace ORB_SLAM2
{


bool EdgeSE3ProjectXYZOnlyWeightPose::read(std::istream& is){
  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectXYZOnlyWeightPose::write(std::ostream& os) const {

  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

void EdgeSE3ProjectXYZOnlyWeightPose::linearizeOplus() {
  g2o::VertexSE3Expmap * vi = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
  Vector3d xyz_trans = vi->estimate().map(Xw);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double invz = 1.0/xyz_trans[2];
  double invz_2 = invz*invz;

  _jacobianOplusXi(0,0) = x*y*invz_2 *fx;
  _jacobianOplusXi(0,1) = -(1+(x*x*invz_2)) *fx;
  _jacobianOplusXi(0,2) = y*invz *fx;
  _jacobianOplusXi(0,3) = -invz *fx;
  _jacobianOplusXi(0,4) = 0;
  _jacobianOplusXi(0,5) = x*invz_2 *fx;

  _jacobianOplusXi(1,0) = (1+y*y*invz_2) *fy;
  _jacobianOplusXi(1,1) = -x*y*invz_2 *fy;
  _jacobianOplusXi(1,2) = -x*invz *fy;
  _jacobianOplusXi(1,3) = 0;
  _jacobianOplusXi(1,4) = -invz *fy;
  _jacobianOplusXi(1,5) = y*invz_2 *fy;

  _jacobianOplusXi = w*_jacobianOplusXi;
}

Vector2d EdgeSE3ProjectXYZOnlyWeightPose::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj;
  proj(0) = trans_xyz(0)/trans_xyz(2);
  proj(1) = trans_xyz(1)/trans_xyz(2);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}

/////////////


bool EdgeSE3ProjectXYZOnlyWeightPoseDebug::read(std::istream& is){
  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectXYZOnlyWeightPoseDebug::write(std::ostream& os) const {

  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

void EdgeSE3ProjectXYZOnlyWeightPoseDebug::linearizeOplus() {
  g2o::VertexSE3Expmap * vi = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
  Vector3d xyz_trans = vi->estimate().map(Xw);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double invz = 1.0/xyz_trans[2];
  double invz_2 = invz*invz;

  _jacobianOplusXi(0,0) = x*y*invz_2 *fx;
  _jacobianOplusXi(0,1) = -(1+(x*x*invz_2)) *fx;
  _jacobianOplusXi(0,2) = y*invz *fx;
  _jacobianOplusXi(0,3) = -invz *fx;
  _jacobianOplusXi(0,4) = 0;
  _jacobianOplusXi(0,5) = x*invz_2 *fx;

  _jacobianOplusXi(1,0) = (1+y*y*invz_2) *fy;
  _jacobianOplusXi(1,1) = -x*y*invz_2 *fy;
  _jacobianOplusXi(1,2) = -x*invz *fy;
  _jacobianOplusXi(1,3) = 0;
  _jacobianOplusXi(1,4) = -invz *fy;
  _jacobianOplusXi(1,5) = y*invz_2 *fy;

  _jacobianOplusXi = w*_jacobianOplusXi;
}

Vector2d EdgeSE3ProjectXYZOnlyWeightPoseDebug::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj;
  proj(0) = trans_xyz(0)/trans_xyz(2);
  proj(1) = trans_xyz(1)/trans_xyz(2);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}

/////////////

Vector2d EdgeSO3ProjectXYZOnlyRotation::cam_project(const Eigen::Quaterniond & q) const{
  Eigen::Matrix3d R = q.matrix();
  Vector3d trans_xyz = R * Xw + tcw;
  Vector2d proj;
  proj(0) = trans_xyz(0)/trans_xyz(2);
  proj(1) = trans_xyz(1)/trans_xyz(2);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}

void EdgeSO3ProjectXYZOnlyRotation::linearizeOplus() 
{
	const VertexRotation* v1 = static_cast<const VertexRotation*>(_vertices[0]);
	Eigen::Matrix3d R = v1->estimate().matrix();
	Vector3d Pr = R * Xw;
	Vector3d Pt = Pr + tcw;

	double x = Pt[0], y = Pt[1], z = Pt[2];
	double z2 = z * z;
	Matrix<double, 2, 3> jacobian_e_p;
	jacobian_e_p << -fx / z, 0, 	fx * x / z2,
					0, 		-fy / z,fy * y / z2;

	double xr = Pr[0], yr = Pr[1], zr = Pr[2];
	Matrix<double, 3, 3> jacobian_p_r;
	jacobian_p_r << 0, 	zr,	-yr,
					-zr,0, 	xr,
					yr,	-xr,0;

	_jacobianOplusXi = jacobian_e_p * jacobian_p_r;
}

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

	_jacobianOplusXi = -w*(jacobian_p_ksi);
}
Matrix3d EdgeSE3ProjectXYZ2XYZOnlyPoseQuat::skew(Vector3d phi)
{
	Matrix3d Phi;
	Phi << 0, -phi[2], phi[1],
		phi[2], 0, -phi[0],
		-phi[1], phi[0], 0;
	return Phi;
}

Vector2d EdgeSE3ProjectPw2BirdPixelPoint::cam_project(const Vector3d & Xc) const{
  
  Vector3d Xb = Rbc * Xc + tbc;
  Vector2d res;
  res[0] = birdviewCols/2- Xb[1] *meter2pixel;
  res[1] = birdviewRows/2-(Xb[0]-rear_axle_to_center)*meter2pixel;
  res = res;

  return res;
}

void EdgeSE3ProjectPw2BirdPixelPoint::linearizeOplus() {
  g2o::SE3Quat pose = (static_cast<VertexSE3Quat*> (_vertices[1]))->estimate();
  Vector3d point = (static_cast<g2o::VertexSBAPointXYZ*> (_vertices[0]))->estimate();

  Vector3d Xc = pose * point;
  
  Matrix<double, 3, 6> jacobian_pc_pw;
  jacobian_pc_pw << -skew(Xc), Matrix3d::Identity();

  Matrix<double, 3, 3> jacobian_pb_pc = Rbc;

  Matrix<double, 2, 3> jacobian_pixel_pb;
  jacobian_pixel_pb << 0, meter2pixel, 0,
					meter2pixel, 0, 0;
  
  _jacobianOplusXi = w * jacobian_pixel_pb * jacobian_pb_pc * pose.rotation().toRotationMatrix();
  _jacobianOplusXj = w * jacobian_pixel_pb * jacobian_pb_pc * jacobian_pc_pw;
}

Matrix3d EdgeSE3ProjectPw2BirdPixelPoint::skew(Vector3d phi)
{
	Matrix3d Phi;
	Phi << 0, -phi[2], phi[1],
		phi[2], 0, -phi[0],
		-phi[1], phi[0], 0;
	return Phi;
}

Matrix6d EdgePose2Pose::JRInv(Vector6d e)
{
	Matrix6d J;
	J.block(0, 0, 3, 3) = skew(e.head(3));
	J.block(3, 3, 3, 3) = skew(e.head(3));
	J.block(3, 0, 3, 3) = skew(e.tail(3));
	J.block(0, 3, 3, 3) = Eigen::Matrix3d::Zero();
	
	J = 0.5 * J + Matrix6d::Identity();
	return J;
}

void EdgePose2Pose::linearizeOplus()
{
	g2o::SE3Quat v1 = (static_cast<VertexSE3Quat*> (_vertices[0]))->estimate();
	g2o::SE3Quat v2 = (static_cast<VertexSE3Quat*> (_vertices[1]))->estimate();

	g2o::SE3Quat e = _measurement.inverse() * v1 * v2.inverse();
	Matrix6d J = JRInv(e.log());

	_jacobianOplusXi = w*J*v2.adj()*v1.inverse().adj();
	_jacobianOplusXj = -w*J;
}

Matrix3d EdgePose2Pose::skew(Vector3d phi)
{
	Matrix3d Phi;
	Phi << 0, -phi[2], phi[1],
		phi[2], 0, -phi[0],
		-phi[1], phi[0], 0;
	return Phi;
}

Vector2d EdgeSE3ProjectPw2BirdPixel::cam_project(const Vector3d & Xc) const{
  
  Vector3d Xb = Rbc * Xc + tbc;
  Vector2d res;
  res[0] = birdviewCols/2- Xb[1] *meter2pixel;
  res[1] = birdviewRows/2-(Xb[0]-rear_axle_to_center)*meter2pixel;
  res = res;

  return res;
}

void EdgeSE3ProjectPw2BirdPixel::linearizeOplus() {
  g2o::VertexSE3Expmap * vi = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
  Vector3d Xc = vi->estimate().map(Xw);
  
  Matrix<double, 3, 6> jacobian_pc_pw;
  jacobian_pc_pw << -skew(Xc), Matrix3d::Identity();

  Matrix<double, 3, 3> jacobian_pb_pc = Rbc;

  Matrix<double, 2, 3> jacobian_pixel_pb;
  jacobian_pixel_pb << 0, meter2pixel, 0,
					meter2pixel, 0, 0;

  _jacobianOplusXi = w * jacobian_pixel_pb * jacobian_pb_pc * jacobian_pc_pw;
}

Matrix3d EdgeSE3ProjectPw2BirdPixel::skew(Vector3d phi)
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
