// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2018 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "transport_map.h"
#include "utils/bvh2d.h"
#include "utils/BenchTimer.h"
#include "utils/eigen_addons.h"

using namespace Eigen;
using namespace surface_mesh;

namespace otmap
{


TransportMap::TransportMap( std::shared_ptr<surface_mesh::Surface_mesh> origin_mesh,
                            std::shared_ptr<surface_mesh::Surface_mesh> fwd_mesh,
                            std::shared_ptr<Eigen::VectorXd> density)
  : m_origin_mesh(origin_mesh), m_fwd_mesh(fwd_mesh), m_density(density), m_bvh(0)
{}

void TransportMap::init_inverse() const
{
  if(m_bvh==nullptr)
  {
    m_bvh = new BVH2D;
    m_bvh->build(m_fwd_mesh.get(),4,24);
  }
}

TransportMap::~TransportMap()
{
  delete m_bvh;
}

Eigen::Vector2d TransportMap::inv_impl(const Eigen::Vector2d& p_in,bool fast_mode) const
{
  // snap to [0,1]:
  Vector2d p = p_in.array().max(0.).min(1.);

  if(!fast_mode)
  {
    // If the target density is given,
    // then let's find all overlaping cells,
    // and keep the one with largest density (<=> largest area)

    const VectorXd& density = *m_density;

    std::vector<BVH2D::Hit> hits;
    m_bvh->query_all(p,hits);
    if(hits.size()==0)
    {
      std::cerr << "Error: no face found. " << p.transpose() << "\n";
    }

    Surface_mesh::Face f = hits[0].face_id;
    double* w = hits[0].bary_coord;
    if(hits.size()>1)
    {
      // find intersecting face with highest density (area)
      double best_area = density(f.idx());
      for(int k=1; k<hits.size(); ++k)
      {
        if(density(hits[k].face_id.idx()) > best_area)
        {
          f = hits[k].face_id;
          w = hits[0].bary_coord;
          best_area = density(hits[k].face_id.idx())>best_area;
        }
      }
    }

    int indices[4];
    int j = 0;
    for(auto v : m_fwd_mesh->vertices(f))
      indices[j++] = v.idx();

    Vector2d res = w[0]*m_origin_mesh->points()[indices[0]];
    for(int i=1;i<j;++i)
      res += w[i]*m_origin_mesh->points()[indices[i]];
    return res;
  }
  else
  {
    // otherwise, pick the first (faster)
    return m_bvh->interpolate_at(p, m_origin_mesh->points());
  }
}


void
apply_inverse_map(const otmap::TransportMap& tmap, std::vector<Vector2d> &points, int verbose_level)
{
  BenchTimer timer;

  double bvh_init = 0.;
  double bvh_queries = 0.;

  timer.start();
  tmap.init_inverse();
  timer.stop();
  bvh_init = timer.value(REAL_TIMER);

  timer.start();
  for(int i=0; i<points.size(); ++i)
  {
    points[i] = tmap.inv(points[i]);
  }
  timer.stop();
  bvh_queries = timer.value(REAL_TIMER);

  if(verbose_level>=2)
    std::cout << "Inversion: bvh_init(" << bvh_init << ") + bvh_queries(" << bvh_queries << ") = " << bvh_init+bvh_queries << "\n";
}


double
transport_cost(const Surface_mesh &src_mesh, const Surface_mesh &dst_mesh, const VectorXd &density_per_face, VectorXd *cost_per_face)
{
  int ncells = density_per_face.size();
  int nfaces = src_mesh.faces_size();

  if(cost_per_face!=0 && cost_per_face->size() != ncells){
    cost_per_face->resize(ncells);
    cost_per_face->setZero();
  }

  double res = 0;
  double integral = 0;
  //for each face
  for(int i=0; i<nfaces; ++i){
    Surface_mesh::Face f(i);

    int id = i;
    double face_area = 0;
    double face_cost = 0;

    if(nfaces==2*density_per_face.size())
    {
      // uniform grid mode -> pair this triangle with the next one to form a quad
      Surface_mesh::Face f2(i+1);
      Surface_mesh::Halfedge h2(src_mesh.halfedge(f2));

      // find common halfedge
      if(src_mesh.face(src_mesh.opposite_halfedge(h2))!=f)
      {
        h2 = src_mesh.next_halfedge(h2);
        if(src_mesh.face(src_mesh.opposite_halfedge(h2))!=f)
          h2 = src_mesh.next_halfedge(h2);
      }
      assert(src_mesh.face(src_mesh.opposite_halfedge(h2))==f);

      Surface_mesh::Vertex v1 = src_mesh.to_vertex(h2);
      Surface_mesh::Vertex v2 = src_mesh.to_vertex(src_mesh.next_halfedge(src_mesh.opposite_halfedge(h2)));
      Surface_mesh::Vertex v3 = src_mesh.from_vertex(h2);
      Surface_mesh::Vertex v4 = src_mesh.to_vertex(src_mesh.next_halfedge(h2));

      face_area = signed_area(src_mesh.position(v1), src_mesh.position(v2), src_mesh.position(v3), src_mesh.position(v4));

      const double z1 =  sqrt(1./3.)/2.+0.5;
      const double z2 = -sqrt(1./3.)/2.+0.5;

      auto dist2 = [&](double u, double v) {
        return  ((1-u) * (1-v) * (dst_mesh.position(v1) - src_mesh.position(v1))
              +     u  * (1-v) * (dst_mesh.position(v2) - src_mesh.position(v2))
              +     u  *    v  * (dst_mesh.position(v3) - src_mesh.position(v3))
              +  (1-u) *    v  * (dst_mesh.position(v4) - src_mesh.position(v4))).squaredNorm();
      };

      face_cost = (dist2(z1,z1) + dist2(z1,z2) + dist2(z2,z2) + dist2(z2,z1)) / 4.;

      id = i/2;
      ++i;
    }
    else
    {
      Surface_mesh::Halfedge h(src_mesh.halfedge(f));
      Surface_mesh::Vertex  v1 = src_mesh.from_vertex(h);
      Surface_mesh::Vertex  v2 = src_mesh.to_vertex(h);
      Surface_mesh::Vertex  v3 = src_mesh.to_vertex(src_mesh.next_halfedge(h));

      face_area = signed_area(src_mesh.position(v1), src_mesh.position(v2), src_mesh.position(v3));

      auto dist2 = [&](double l1, double l2, double l3) {
        return (  l1*(dst_mesh.position(v1) - src_mesh.position(v1))
                + l2*(dst_mesh.position(v2) - src_mesh.position(v2))
                + l3*(dst_mesh.position(v3) - src_mesh.position(v3)) ).squaredNorm();
      };

      face_cost = (dist2(0.5,0.5,0)+dist2(0,0.5,0.5)+dist2(0.5,0,0.5)) / 3.0;
    }
    integral += density_per_face(id) * face_area;
    face_cost = density_per_face(id) * face_area * face_cost;
    res += face_cost;
    if(cost_per_face!=0)
      (*cost_per_face)(i) = face_cost;
  }
  if(cost_per_face!=0)
    (*cost_per_face) /= integral;
  return res/integral;
}

}
