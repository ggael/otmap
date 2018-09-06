// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2016-2018 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <Eigen/Geometry>
#include <surface_mesh/Surface_mesh.h>

namespace otmap
{

// Acceleration data-structure to query the triangle face of a mesh containing a given point q,
// and interpolate given attributes attached to vertices:
// Usage:
//  Surface_mesh mesh;
//  BVH2D bvh(mesh);
//  std::vector<ValueType> attributes(mesh.nb_vertices());
//  attributes = ...;
//  for(...) {
//    ValueType value = bvh.interpolate_at(q, attributes);
//    ...
//  }
class BVH2D
{

    struct Node {
      Eigen::AlignedBox2d box;
      union {
        int first_child_id; // for inner nodes
        int first_face_id;  // for leaves
      };
      unsigned short nb_faces;
      short is_leaf;
    };

    typedef std::vector<Node> NodeList;

  public:
    BVH2D();
    ~BVH2D();

    void build(surface_mesh::Surface_mesh *mesh, int targetCellSize=4, int maxDepth=10);

    surface_mesh::Surface_mesh::Face query(const Eigen::Vector2d &q, double *w) const;

    struct Hit {
      surface_mesh::Surface_mesh::Face face_id;
      double bary_coord[4];
    };
    void query_all(const Eigen::Vector2d &q, std::vector<Hit> &hits) const;

    template<typename Data>
    typename Data::value_type interpolate_at(const Eigen::Vector2d &q, const Data& data);

  protected:

    void intersectNode(int nodeId, const Eigen::Vector2d &target, std::vector<Hit> &hits, bool stop_at_first) const;

    int split(int start, int end, int dim, float split_value);

    void buildNode(int nodeId, int start, int end, int level, int targetCellSize, int maxDepth);

    surface_mesh::Surface_mesh* mesh_;
    std::vector<Eigen::Vector2d> m_points;
    NodeList nodes_;
    std::vector<surface_mesh::Surface_mesh::Face> faces_;
    std::vector<Eigen::Vector2d> m_centroids;
};

template<typename Data>
typename Data::value_type BVH2D::interpolate_at(const Eigen::Vector2d &q, const Data& data)
{
  double w[4];
  surface_mesh::Surface_mesh::Face f = query(q,w);
  if(!f.is_valid())
  {
    std::cerr << "Error: no face found. " << q.transpose() << "\n";
    return typename Data::value_type();
  }

  int indices[4];
  int j = 0;
  for(auto v : mesh_->vertices(f))
    indices[j++] = v.idx();

  typename Data::value_type res = w[0]*data[indices[0]];
  for(int i=1;i<j;++i)
    res += w[i]*data[indices[i]];
  return res;
}

}
