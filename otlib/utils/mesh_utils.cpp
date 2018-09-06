// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2017-2018 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "mesh_utils.h"
#include <limits>

using namespace Eigen;
using namespace surface_mesh;

namespace otmap {

// works for any quads, but q must be inside the quad
bool bilinear_coordinates_in_quad(const Eigen::Vector2d& q, const Eigen::Vector2d *p, double &u, double &v)
{
  using std::abs;
  using std::sqrt;

  double A0 = (signed_area(q,p[0],p[1]));
  double A1 = (signed_area(q,p[1],p[2]));
  double A2 = (signed_area(q,p[2],p[3]));
  double A3 = (signed_area(q,p[3],p[0]));
  double B0 = (signed_area(q,p[3],p[1]));
  double B1 = (signed_area(q,p[0],p[2]));
  double B3 = (signed_area(q,p[2],p[0]));
  double D = B0*B0+B1*B1+2*A0*A2+2*A1*A3;
  if(D<0)
    return false;
  double E0 = 2*A0-B0-B1+sqrt(D);
  double E3 = 2*A3-B3-B0+sqrt(D);

  double zero = 4*std::numeric_limits<double>::min();
  v = abs(A0)<=zero ? 0 : 2*A0/E0;
  u = abs(A3)<=zero ? 0 : 2*A3/E3;

  double E0p = 2*A0-B0-B1-sqrt(D);
  double E3p = 2*A3-B3-B0-sqrt(D);
  double v_alt = abs(A0)<=zero ? 0 : 2*A0/E0p;
  double u_alt = abs(A3)<=zero ? 0 : 2*A3/E3p;
  double eps = std::sqrt(std::numeric_limits<double>::epsilon());
  if(abs(u)<eps) u = 0;
  if(abs(v)<eps) v = 0;
  if(abs(u-1.)<eps) u = 1;
  if(abs(v-1.)<eps) v = 1;
  if(abs(u_alt)<eps) u_alt = 0;
  if(abs(v_alt)<eps) v_alt = 0;
  if(abs(u_alt-1.)<eps) u_alt = 1;
  if(abs(v_alt-1.)<eps) v_alt = 1;
  if((u<0 || v<0 || u>1 || v>1) && (u_alt>=0 && v_alt>=0 && u_alt<=1 && v_alt<=1)){
    std::swap(u,u_alt);
    std::swap(v,v_alt);
  }

  return true;
}


bool inside_quad(const Vector2d& q, const Vector2d *p)
{
  double eps = 1e-8;
  Vector2d uv = bilinear_coordinates_in_triangle(q,p[0],p[1],p[2]);
  if((uv.array()>=-eps).all() && uv.sum()<=1.+eps) {
    return true;
  }
  uv = bilinear_coordinates_in_triangle(q,p[0],p[2],p[3]);
  if((uv.array()>=-eps).all() && uv.sum()<=1.+eps) {
    return true;
  }
  return false;
}

bool bilinear_coordinates_in_quad(const Eigen::Vector2d& q, const Eigen::Vector2d *p, Eigen::Ref<Eigen::Vector4d> w)
{
  double eps = 4*(std::numeric_limits<double>::epsilon());
  double u,v;
  if(!bilinear_coordinates_in_quad(q, p, u, v))
    return false;
  w << (1.-u)*(1-v), (u)*(1.-v), (u)*(v), (1.-u)*(v);
  for(int k=0;k<4;++k) {
    if(std::abs(w(k))<=eps) w(k) = 0;
    else if(std::abs(w(k)-1.)<=eps) w(k) = 1;
  }
  return true;
}

void generate_quad_mesh(int m, int n, Surface_mesh &mesh, bool inclusive)
{
  using namespace surface_mesh;
  using namespace Eigen;

  mesh.clear();

  mesh.reserve(m*n, (m-1)*n+ m*(n-1), (m-1)*(n-1));

  double dx = 1./double(n-1);
  double dy = 1./double(m-1);

  Eigen::Array<Surface_mesh::Vertex,Dynamic,Dynamic> ids(m,n);
  for(int i=0;i<n;++i)
    for(int j=0;j<m;++j)
      if(inclusive)
        ids(i,j) = mesh.add_vertex(Point((i+0.5)/double(n),(j+0.5)/double(m)));
      else
        ids(i,j) = mesh.add_vertex(Point(double(i)*dx, double(j)*dy));


  for(int i=0; i<n-1; ++i) {
    for(int j=0; j<m-1; ++j) {
      Surface_mesh::Vertex v0,v1,v2,v3;
      v0 = ids(i+0,j+0);
      v1 = ids(i+1,j+0);
      v2 = ids(i+1,j+1);
      v3 = ids(i+0,j+1);

      mesh.add_quad(v0, v1, v2, v3);
    }
  }
}

void
prune_empty_faces(Surface_mesh &mesh, VectorXd density)
{
  int nf = mesh.faces_size();
  if(nf == 2*density.size()) {
    // grid represented as triangles, duplicate entries:
    density.conservativeResize(nf);
    for(int i=nf-1; i>=0; i--)
      density(i) = density(i/2);
  }
  for(int i=0; i<nf;++i)
  {
    if(density(i)==0)
      mesh.delete_face(Surface_mesh::Face(i));
  }
  mesh.garbage_collection();
}

}
