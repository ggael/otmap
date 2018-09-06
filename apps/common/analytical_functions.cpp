// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2017-2018 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2017 Georges Nader
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "analytical_functions.h"

#include <cmath>
#include <vector>
#include <Eigen/Geometry>

using namespace Eigen;
using namespace surface_mesh;

double linear1D(const Eigen::Vector2d& p)
{
  return 2.*p.x();
}

double single_black(const Eigen::Vector2d& p)
{
  Eigen::Vector2d c(0.5,0.5);
  double eps = 1./10000.;
  if((p-c).norm()<eps)
    return 0;
  else
    return 1;
}

double circles(const Eigen::Vector2d& p, bool boundary)
{
  double t1 = 1./3.;
  double r1 = 0.1;
  double r2 = 0.05;
  Vector2d c1(t1,t1);
  double l1 = (p.array()-0.5).abs().maxCoeff();
  if(boundary && (l1>0.5-r2)) return 1;
  if(  (p-Vector2d(   t1,   t1)).norm()<r1
    || (p-Vector2d(   t1,1.-t1)).norm()<r1
    || (p-Vector2d(1.-t1,1.-t1)).norm()<r1
    || (p-Vector2d(1.-t1,   t1)).norm()<r1)
  {
    return 1;
  }
  return 0;
}

double sincos(const Eigen::Vector2d& p, double exponent)
{
  return std::pow(sin(p.x()*6)*cos(p.y()*6)+1.001 , exponent);
}

double brdf(const Eigen::Vector2d& p, const Eigen::Vector2d& l, double nu, double nv, const Eigen::Vector2d& u = Eigen::Vector2d(1,0))
{
  using namespace Eigen;
  double fresnel = 1;

  Vector2d q = p.array() * 2. - 1.;

  auto ortho_mapping = [](const Eigen::Vector2d& p)
  {
    double w = std::sin(p.norm()*M_PI/2.);
    double z = std::sqrt(1.-w*w);
    Vector3d p3;
    p3 << p.normalized()*w, z;
    return p3;
  };

  auto paraboloid_mapping = [](const Eigen::Vector2d& p)
  {
    Vector3d p3;
    p3.head<2>() = p;
    p3.z() = 0.5 - 0.5*(p.squaredNorm());
    return p3.normalized();
  };

  auto unwrap_mapping = [](Eigen::Vector2d p)
  {
    using std::cos;
    using std::sin;
    p *= M_PI/2.;
    Vector3d p3( cos(p.x())*sin(p.y()), sin(p.x()), cos(p.x())*cos(p.y()) );
    return p3;
  };

  Vector3d q3 = unwrap_mapping(q);
  Vector3d l3 = ortho_mapping(l);
  Vector3d h = (q3+l3).normalized();
  Vector2d v = u.unitOrthogonal();

  double exponent = ( nu*numext::abs2(h.head<2>().dot(u)) + nv*numext::abs2(h.head<2>().dot(v)) ) / ( 1. - numext::abs2(h.z()) );

  return 1e-3+std::pow(h.z(), exponent) * fresnel;
}

// see [DCFCL08]
double DCFCL08_example1(const Eigen::Vector2d& p)
{
  Eigen::Vector2d center(0.5, 0.5);
  double radius = (p-center).norm();

  return 1./(2.+cos(8*M_PI*radius));
}

double DCFCL08_example2(const Eigen::Vector2d& p)
{
  Eigen::Vector2d center(0.7, 0.5);
  double radius = (p-center).norm();
  double theta  = atan2(p.y()-center.y(),p.x()-center.x());

  double c = 10.*radius*cos(theta-20.*radius*radius);
  return 1.+ 9./(1.+c*c);
}

double DCFCL08_example3(const Eigen::Vector2d& p)
{
  Eigen::Vector2d c1(0.5, 0.5);
  Eigen::Vector2d c2(0.7, 0.7);

  double r1 = (p-c1).norm();
  double r2 = (p-c2).norm();

  return exp(-2.*r1)/r2;
}

// From https://www.birs.ca/workshops/2011/11w5086/files/oberman.pdf
// psi + x^2/2 = -sqrt(2-x^2)
double singularity1(const Eigen::Vector2d& p)
{
  return 2./numext::abs2(2-p.squaredNorm());
}

//Analytical example from :
//  Numerical solutions of the optimal transportation problem using
//  the monge-ampere equation. [BFO12]

double q(double x)
{
  return (-(x*x)/(8.*M_PI) + 1./(256.*M_PI*M_PI*M_PI) + 1./(32.*M_PI)) * cos(8.*M_PI*x) + x*sin(8.*M_PI*x)/(32.*M_PI*M_PI);
}

double dq(double x)
{
  return (4.*x*x-1.)*sin(8.*M_PI*x)/4;
}

double d2q(double x)
{
  return 2.*(x*sin(8.*M_PI*x) + (4.*M_PI*x*x-M_PI)*cos(8.*M_PI*x));
}

double BFO12_example(const Eigen::Vector2d& p)
{
  Eigen::Vector2d center(0.5, 0.5);
  Eigen::Vector2d pt = p-center;

  double qx = q(pt.x()); double dqx = dq(pt.x()); double d2qx = d2q(pt.x());
  double qy = q(pt.y()); double dqy = dq(pt.y()); double d2qy = d2q(pt.y());

  return 1. + 4.*(d2qx*qy + qx*d2qy) + 16.*(qx*qy*d2qx*d2qy - dqx*dqx*dqy*dqy);
}

VectorXd BF012_error(const std::vector<Eigen::Vector2d>& points, const std::vector<Eigen::Vector2d>& map)
{
  int n = points.size();

  VectorXd err;
  err.resize(n); err.setZero();

  for(unsigned int i=0; i<n; ++i){
    Eigen::Vector2d center(0.5, 0.5);
    Eigen::Vector2d pt = points[i]-center;

    //compute the exact gradient
    double qx = q(pt.x()); double dqx = dq(pt.x());
    double qy = q(pt.y()); double dqy = dq(pt.y());

    Eigen::Vector2d ref_grad(pt.x()+4.*dqx*qy, pt.y()+4.*qx*dqy);
    Eigen::Vector2d num_grad = (map[i])-center;
    err(i) = (ref_grad-num_grad).norm();
  }

  return err;
}


double BFO12_ellipse1(const Eigen::Vector2d& p)
{
  Eigen::Vector2d center(0.5, 0.5);
  Vector2d q = p - center;
  Matrix2d M;
  M << 0.8, 0. ,
       0. , 0.4;
  M *= 0.5;
  return (q.transpose() * (M*M).inverse() * q).value() < 1 ? 1 : 0;
}
double BFO12_ellipse2(const Eigen::Vector2d& p)
{
  Eigen::Vector2d center(0.5, 0.5);
  Vector2d q = p - center;
  Matrix2d M;
  M << 0.6, 0.2,
       0.2, 0.8;
  M *= 0.5;
  return (q.transpose() * (M*M).inverse() * q).value() < 1 ? 1 : 0;
}

double BFO12_gauss(const Vector2d& p, const Vector2d& c, double s) {
  return 2 + exp(-0.5*(p-c).squaredNorm() / numext::abs2(s))/numext::abs2(s);
}
double BFO12_1gauss(const Vector2d& p)
{
  const Vector2d& q = p.array()*2-1;
  return BFO12_gauss(q, Vector2d(0,0), 0.2);
}
double BFO12_4gauss(const Vector2d& p)
{
  const Vector2d& q = p.array()*2-1;
  if(q.x()<0 && q.y()<0)
    return BFO12_gauss(q, Vector2d(-1,-1), 0.2);
  else if(q.x()<0 && q.y()>=0)
    return BFO12_gauss(q, Vector2d(-1,1), 0.2);
  else if(q.x()>=0 && q.y()<0)
    return BFO12_gauss(q, Vector2d(1,-1), 0.2);
  else // if(q.x()>=0 && q.y()>=0)
    return BFO12_gauss(q, Vector2d(1,1), 0.2);
}

double func(FuncName::Enum fn, const Eigen::Vector2d& p)
{
  using namespace FuncName;
  double val = 1;
  switch (fn) {
    case CONSTANT:      val = 1.; break;
    case LIN1D:         val = linear1D(p); break;
    case CIRCLES_BOUND: val = circles(p, true); break;
    case SINGULARITY1:  val = singularity1(p); break;
    case CIRCLES:       val = circles(p, false); break;
    case SINGLE_BLACK:  val = single_black(p); break;
    case SINGLE_WHITE:  val = 1.-single_black(p); break;

    case BFO12EX:       val = BFO12_example(p); break;
    case BFO12ELLIPSE1: val = BFO12_ellipse1(p); break;
    case BFO12ELLIPSE2: val = BFO12_ellipse2(p); break;
    case BFO12_1GAUSS:  val = BFO12_1gauss(p); break;
    case BFO12_4GAUSS:  val = BFO12_4gauss(p); break;

    case DCFCL08EX1:    val = DCFCL08_example1(p); break;
    case DCFCL08EX2:    val = DCFCL08_example2(p); break;
    case DCFCL08EX3:    val = DCFCL08_example3(p); break;

    case SINCOS1:       val = sincos(p,1); break;
    case SINCOS3:       val = sincos(p,3); break;
    case SINCOS5:       val = sincos(p,5); break;

    case BRDF0:         val = brdf(p,Vector2d(-0.5,0), 50, 3); break;
    case BRDF1:         val = brdf(p,Vector2d(-0.5,0), 35, 4.5); break;
    case BRDF2:         val = brdf(p,Vector2d(-0.5,0), 23, 6.5); break;
    case BRDF3:         val = brdf(p,Vector2d(-0.5,0), 8, 8); break;

    default:
      break;
  }
  return val;
}

// ----------------------------------------------------------------------------

/** computes an analytical scalar function on 2D regular grid
  * the function is evaluated at the center of grid cells
  */
void eval_func_to_grid(Ref<MatrixXd> density, int fn)
{
  int resX = density.rows();
  int resY = density.cols();
  density.setZero();

  double dx = 1./double(resX);
  double dy = 1./double(resY);

  for(int j=0; j<resY; ++j){
    for(int i=0; i<resX; ++i){
      Eigen::Vector2d p(0.5*dx + double(j)*dx, 0.5*dy + double(i)*dy);

      density(i,j) = func(FuncName::Enum(fn), p);
    }
  }
  // scale to [0,1]
  density = density/density.maxCoeff();
}

