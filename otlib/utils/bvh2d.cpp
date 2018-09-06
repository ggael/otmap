// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2016-2018 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "bvh2d.h"

#include <iostream>
#include <Eigen/Geometry>
#include "mesh_utils.h"

using namespace surface_mesh;
using namespace Eigen;

namespace otmap
{

BVH2D::BVH2D()
{
}

BVH2D::~BVH2D()
{
}

void BVH2D::build(surface_mesh::Surface_mesh *mesh, int targetCellSize, int maxDepth)
{
    mesh_ = mesh;

    m_points = mesh->get_vertex_property<Point>("v:point").vector();

    // Number of vertices in the mesh
    int n_vtx = m_points.size();

    m_centroids.clear();
    faces_.clear();

    nodes_.resize(1);
    nodes_.reserve( std::min<int>(2<<maxDepth, std::max<int>(1,std::log(mesh_->n_faces()/targetCellSize)) ) );
    m_centroids.resize(mesh_->n_faces());
    faces_.resize(mesh_->n_faces());

    Surface_mesh::Face_iterator fit;
    int i=0;
    for (fit = mesh_->faces_begin(); fit != mesh_->faces_end(); ++fit, ++i)
    {
        m_centroids[i] = Vector2d::Zero();
        Surface_mesh::Vertex_around_face_circulator vfc, vfc_end;
        vfc = vfc_end = mesh_->vertices(*fit);
        int j=0;
        do {
            m_centroids[i] += m_points[(*vfc).idx()];
            j++;
        } while (++vfc != vfc_end);
        m_centroids[i] /= j;
        faces_[i] = (*fit);
    }

    nodes_.resize(1);
    buildNode(0, 0, mesh_->n_faces(), 0, targetCellSize, maxDepth);
}

Surface_mesh::Face BVH2D::query(const Eigen::Vector2d &p, double *w) const
{
  if(nodes_[0].box.contains(p))
  {
    std::vector<Hit> hit;
    intersectNode(0, p, hit, true);
    if(hit.size()==1)
    {
      Vector4d::Map(w) = Vector4d::Map(hit[0].bary_coord);
      return hit[0].face_id;
    }
  }
  else
    std::cerr <<  "OOPS, query is out of main bounding box\n";

  return Surface_mesh::Face();
}

void BVH2D::query_all(const Eigen::Vector2d &q, std::vector<BVH2D::Hit> &hits) const
{
  if(nodes_[0].box.contains(q))
  {
    intersectNode(0, q, hits, false);
  }
  else
    std::cerr <<  "OOPS, query is out of main bounding box\n";
}

void BVH2D::intersectNode(int nodeId, const Eigen::Vector2d &target, std::vector<Hit> &hits, bool stop_at_first) const
{
  const Node& node = nodes_[nodeId];

  if(node.is_leaf)
  {
    int end = node.first_child_id+node.nb_faces;
    for(int i=node.first_child_id; i<end; ++i)
    {
      Vector2d pts[4];
      int j = 0;
      for(auto v:mesh_->vertices(faces_[i]))
        pts[j++] = m_points[v.idx()];

      if(j==3)
      {
        Vector2d uv = bilinear_coordinates_in_triangle(target,pts[0],pts[1],pts[2]);
        double eps = 1e-8;
        if((uv.array()>=-eps).all() && uv.sum()<=1.+eps) {
          Hit hit;
          hit.bary_coord[0] = uv.x();
          hit.bary_coord[1] = uv.y();
          hit.bary_coord[2] = 1.-uv.sum();
          hit.bary_coord[3] = 0;
          hit.face_id = faces_[i];
          hits.push_back(hit);
          if(stop_at_first)
            return;
        }
      }
      else if(j==4)
      {
        if(inside_quad(target, pts))
        {
          double w[4];
          if(bilinear_coordinates_in_quad(target,pts, Vector4d::Map(w)))
          {
            if((Array4d::Map(w) <= 1.0f).all() && (Array4d::Map(w) >= 0.0f).all())
            {
              Hit hit;
              Vector4d::Map(hit.bary_coord) = Vector4d::Map(w);
              hit.face_id = faces_[i];
              hits.push_back(hit);
              if(stop_at_first)
                return;
            }
          }
        }
      }
      else
        std::cerr << "Invalid polygon with " << j << " vertices\n";
    }
  }
  else
  {
    int child_id1 = node.first_child_id;
    int child_id2 = node.first_child_id+1;
    if(nodes_[child_id1].box.contains(target)) {
      intersectNode(child_id1, target, hits, stop_at_first);
      if(stop_at_first && hits.size()>0)
        return;
    }
    if(nodes_[child_id2].box.contains(target)) {
      intersectNode(child_id2, target, hits, stop_at_first);
      if(stop_at_first && hits.size()>0)
        return;
    }
  }
}

/** Sorts the faces with respect to their centroid along the dimension \a dim and spliting value \a split_value.
  * \returns the middle index
  */
int BVH2D::split(int start, int end, int dim, float split_value)
{
  int l(start), r(end-1);
  while(l<r)
  {
    // find the first on the left
    while(l<end && m_centroids[l](dim) < split_value) ++l;
    while(r>=start && m_centroids[r](dim) >= split_value) --r;
    if(l>=r) break;
    std::swap(m_centroids[l], m_centroids[r]);
    std::swap(faces_[l], faces_[r]);
    ++l;
    --r;
  }
  return m_centroids[l][dim]<=split_value ? std::min(end,l+1) : l;
}

void BVH2D::buildNode(int nodeId, int start, int end, int level, int targetCellSize, int maxDepth)
{
  Node& node = nodes_[nodeId];

  // compute bounding box
  Eigen::AlignedBox2d aabb;
  aabb.setNull();
  for(int i=start; i<end; ++i)
  {
      Surface_mesh::Vertex_around_face_circulator vfc, vfc_end;
      vfc = vfc_end = mesh_->vertices(faces_[i]);
      do {
           aabb.extend(m_points[(*vfc).idx()]);
      } while (++vfc != vfc_end);
  }
  node.box = aabb;
  Eigen::Array2d diag = aabb.max() - aabb.min();
  // enlarge by epsilon
  node.box.min().array() -= (diag*(2.*NumTraits<double>::epsilon()) + NumTraits<double>::epsilon());
  node.box.max().array() += (diag*(2.*NumTraits<double>::epsilon()) + NumTraits<double>::epsilon());

  // stopping criteria
  if(end-start <= targetCellSize || level>=maxDepth)
  {
    // we got a leaf !
    node.is_leaf = true;
    node.first_face_id = start;
    node.nb_faces = std::max(0,end-start);
    return;
  }
  node.is_leaf = false;

  // Split along the largest dimension
  int dim;
  diag.maxCoeff(&dim);
  // Split at the middle
  float split_value = 0.5 * (node.box.max()[dim] + node.box.min()[dim]);

  // Sort the faces according to the split plane
  int mid_id = split(start, end, dim, split_value);

  // second stopping criteria
  if(mid_id==start || mid_id==end)
  {
    // no improvement
    node.is_leaf = true;
    node.first_face_id = start;
    node.nb_faces = std::max(0,end-start);
    return;
  }

  // create the children
  int child_id = node.first_child_id = nodes_.size();
  nodes_.resize(nodes_.size()+2);
  // node is not a valid reference anymore !

  buildNode(child_id  , start, mid_id, level+1, targetCellSize, maxDepth);
  buildNode(child_id+1, mid_id, end, level+1, targetCellSize, maxDepth);
}

}
