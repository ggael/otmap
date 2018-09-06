
// This code has been heavily adapted from 'uraster' initially written by Steven Braeger
// I chose to keep the  original MIT license below.

// The MIT License (MIT)

// Copyright (c) 2015 Steven Braeger
// Copyright (C) 2017-2018 Gael Guennebaud <gael.guennebaud@inria.fr>

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "rasterizer.h"
#include "mesh_utils.h"
#include <array>

using namespace Eigen;
using namespace surface_mesh;

namespace otmap {

struct BarycentricTransform
{
  using Vector2 = Eigen::Vector2d;
  using Matrix2 = Eigen::Matrix2d;
  using Vector3 = Eigen::Vector3d;
private:
  Vector2 offset;
  Matrix2 Ti;
public:
  BarycentricTransform(const Vector2& s1,const Vector2& s2,const Vector2& s3)
    : offset(s3)
  {
    Matrix2 T;
    T << (s1-s3),(s2-s3);
    Ti=T.inverse();
  }
  Vector3 operator()(const Vector2& v) const
  {
    Vector2 b;
    b=Ti*(v-offset);
    return Vector3(b[0],b[1],1.0f-b[0]-b[1]);
  }
};

struct Vert {
  Array2d p;
  float value;

  Vert() : p(0,0), value(0) {}

  operator Array2d () const { return p; }
  operator Vector2d () const { return p; }

  Vert& operator+=(const Vert& tp)
  {
    p += tp.p;
    value += tp.value;
    return *this;
  }
  Vert& operator*=(const float& f)
  {
    p *= f;
    value *= f;
    return *this;
  }
};


//This function takes in 3 vertices in the [0,1]^2 domain, generate fragments, and call fragment_shader for each.
template<class VertexType,class FragShader>
void rasterize_face(int rows, int cols, std::array<VertexType,3> verts, FragShader fragment_shader)
{
  using Vector3 = Eigen::Vector3d;
  using Array2i = Eigen::Array2i;
  using Array2  = Eigen::Array2d;

  Array2 ss1 = verts[0];
  Array2 ss2 = verts[1];
  Array2 ss3 = verts[2];

  // calculate the bounding box of the triangle in screen space floating point.
  Array2 bb_ul = ss1.min(ss2).min(ss3);
  Array2 bb_lr = ss1.max(ss2).max(ss3);
  Array2i isz(cols,rows);

  // convert bounding box to fixed point.
  // and clamp the bounding box to the framebuffer size if necessary
  Array2i ibb_ul = (bb_ul*isz.cast<double>()).cast<int>().max(Array2i(0,0));
  Array2i ibb_lr = ((bb_lr*isz.cast<double>()).cast<int>()+1).min(isz); //add one pixel of coverage

  BarycentricTransform bt(ss1.matrix(),ss2.matrix(),ss3.matrix());

  //for all the pixels in the bounding box
  for(int y=ibb_ul[1];y<ibb_lr[1];y++)
  for(int x=ibb_ul[0];x<ibb_lr[0];x++)
  {
    Array2 ssc(x,y);
    ssc = (ssc+0.5)/isz.cast<double>(); // move pixel to relative coordinates

    // Compute barycentric coordinates of the pixel center
    Vector3 bary = bt(ssc);

    //if the pixel has valid barycentric coordinates, the pixel is in the triangle
    if((bary.array() <= 1.0f).all() && (bary.array() >= 0.0f).all())
    {
      //interpolate varying parameters
      VertexType v;
      for(int i=0;i<3;i++)
      {
        VertexType vt = verts[0];
        vt *= bary[i];
        v += vt;
      }
      //call the fragment processor
      fragment_shader(x,y,v);
    }
  }
}


//This function takes in a quad as 4 vertices in the [0,1]^2 domain, generate fragments, and call fragment_shader for each.
template<class VertexType,class FragShader>
void rasterize_face(int rows, int cols, std::array<VertexType,4> verts, FragShader fragment_shader)
{
  using Array2i = Eigen::Array2i;
  using Array2  = Eigen::Array2d;
  using Vector2  = Eigen::Vector2d;

  Vector2 ss[4];
  ss[0] = verts[0];
  ss[1] = verts[1];
  ss[2] = verts[2];
  ss[3] = verts[3];

  // calculate the bounding box of the triangle in screen space floating point.
  Array2 bb_ul = ss[0].cwiseMin(ss[1]).cwiseMin(ss[2]).cwiseMin(ss[3]);
  Array2 bb_lr = ss[0].cwiseMax(ss[1]).cwiseMax(ss[2]).cwiseMax(ss[3]);
  Array2i isz(cols,rows);

  // convert bounding box to fixed point.
  // and clamp the bounding box to the framebuffer size if necessary
  Array2i ibb_ul = (bb_ul*isz.cast<double>()).cast<int>().max(Array2i(0,0));
  Array2i ibb_lr = ((bb_lr*isz.cast<double>()).cast<int>()+1).min(isz); //add one pixel of coverage

  //for all the pixels in the bounding box
  for(int y=ibb_ul[1];y<ibb_lr[1];y++)
  for(int x=ibb_ul[0];x<ibb_lr[0];x++)
  {
    Array2 ssc(x,y);
    ssc = (ssc+0.5)/isz.cast<double>(); // move pixel to relative coordinates

    //if the pixel has valid barycentric coordinates, the pixel is in the triangle
    Eigen::Vector4d bary;
    bilinear_coordinates_in_quad(ssc, ss, bary);
    if((bary.array() <= 1.0f).all() && (bary.array() >= 0.0f).all())
    {
      //interpolate varying parameters
      VertexType v;
      for(int i=0;i<4;i++)
      {
        VertexType vt = verts[0];
        vt *= bary[i];
        v += vt;
      }
      //call the fragment processor
      fragment_shader(x,y,v);
    }
  }
}


void rasterize_image(const Surface_mesh &mesh, const VectorXd &density_per_Face, MatrixXd& img, RasterImageOption opt)
{
  // create indexed-face-set
  auto& vpositions = mesh.get_vertex_property<Point>("v:point").vector();

  float scale = 1./density_per_Face.size();

  int rows = img.rows();
  int cols = img.cols();
  img.setZero();

  if(opt==RIO_PerFaceDensity)
  {
    for(auto f:mesh.faces())
    {
      std::array<int,4> indices;
      int i = 0;
      for(auto v:mesh.vertices(f))
        indices[i++] = v.idx();
      double in_mass = scale*density_per_Face(f.idx());
      if(i==3)
      {
        double area = std::abs(signed_area(vpositions[indices[0]].head<2>(),
                                           vpositions[indices[1]].head<2>(),
                                           vpositions[indices[2]].head<2>() ));
        rasterize_face(rows,cols,
                       std::array<Vector2d,3>{vpositions[indices[0]].head<2>(),
                                              vpositions[indices[1]].head<2>(),
                                              vpositions[indices[2]].head<2>()},
                       [&] (int x, int y, const Vector2d&) { img(y,x) = in_mass/area; }
                      );
      }
      else if(i==4)
      {
        double area = std::abs(signed_area(vpositions[indices[0]].head<2>(),
                                           vpositions[indices[1]].head<2>(),
                                           vpositions[indices[2]].head<2>(),
                                           vpositions[indices[3]].head<2>()
                                          ));
        rasterize_face(rows,cols,
                       std::array<Vector2d,4>{vpositions[indices[0]].head<2>(),
                                              vpositions[indices[1]].head<2>(),
                                              vpositions[indices[2]].head<2>(),
                                              vpositions[indices[3]].head<2>()
                      },
                       [&] (int x, int y, const Vector2d&) { img(y,x) = in_mass/area; }
                      );
      }
    }
  }
  else // PerVertex
  {
    // first compute per-vertex densities
    std::vector<Vert> vertices(mesh.n_vertices());
    for(int i=0; i<mesh.n_vertices(); ++i)
    {
      vertices[i].p = vpositions[i].head<2>();
      vertices[i].value = 0;
    }

    std::vector<double> divisors(mesh.n_vertices(),0);
    int i=0, j=0;
    for(auto f:mesh.faces())
    {
      std::array<int,4> indices;
      int i = 0;
      for(auto v:mesh.vertices(f))
        indices[i++] = v.idx();
      double area = 0;
      if(i==3)
        area = std::abs(signed_area(vpositions[indices[0]].head<2>(), vpositions[indices[1]].head<2>(),vpositions[indices[2]].head<2>() ));
      else if(i==4)
        area = std::abs(signed_area(vpositions[indices[0]].head<2>(),
                                    vpositions[indices[1]].head<2>(),
                                    vpositions[indices[2]].head<2>(),
                                    vpositions[indices[3]].head<2>()
                                   ));
      for(auto v:mesh.vertices(f))
      {
        vertices[v.idx()].value += 1./area;
        divisors[v.idx()] += 1.;
      }
    }
    for(int i=0; i<mesh.n_vertices(); ++i)
      vertices[i].value = scale * vertices[i].value / divisors[i];

    // then rasterize each face
    for(auto f:mesh.faces())
    {
      std::array<int,4> indices;
      int i = 0;
      for(auto v:mesh.vertices(f))
        indices[i++] = v.idx();
      if(i==3)
      {
        rasterize_face(rows,cols,
                       std::array<Vert,3>{vertices[indices[0]],
                                          vertices[indices[1]],
                                          vertices[indices[2]]},
                       [&] (int x, int y, const Vert& f) { img(y,x) = f.value; }
                      );
      }
      else if(i==4)
      {
        rasterize_face(rows,cols,
                       std::array<Vert,4>{vertices[indices[0]],
                                          vertices[indices[1]],
                                          vertices[indices[2]],
                                          vertices[indices[3]]},
                       [&] (int x, int y, const Vert& f) { img(y,x) = f.value; }
                      );
      }
    }
  }

}

void rasterize_image(const surface_mesh::Surface_mesh& mesh, MatrixXd& img, RasterImageOption opt)
{
  rasterize_image(mesh, VectorXd::Ones(mesh.faces_size()), img, opt);
}

} // namespace otmap
