// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2017-2018 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2017 Georges Nader
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "image_utils.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include <CImg/CImg.h>

using namespace Eigen;
using namespace surface_mesh;

void load_image(const char* filename, MatrixXd &data)
{
  cimg_library::CImg<double> img(filename);

  if(img.is_empty())
  {
    std::cerr << "ERROR image \"" << filename << "\" not found or empty\n";
    data.resize(0,0);
    return;
  }

  int h = img.height();
  int w = img.width();

  data.resize(h, w);

  for(int j=0; j<w; ++j)
    for(int i=0; i<h; ++i)
      if(img.depth()==1)
        data(i,j) = img(i,j,0,0)/255.f;
      else
        data(i,j) = (img(i,j,0,0)*11.0 + img(i,j,0,1) * 16.0 + img(i,j,0,2)*5.0)/(32.*255.);
  double maxval = data.maxCoeff();
  if(maxval>1)
    data /= maxval;
}

void save_image(const char* filename, Ref<const MatrixXd> data)
{
  int h = data.cols();
  int w = data.rows();

  cimg_library::CImg<double> temp(w, h, 1, 3, 0);

  for(int i=0; i<w; ++i){
      for(int j=0; j<h; ++j){

        double v = data(i, j)*255.;
        v = std::max(0., std::min(255., v)); //truncate between 0 and 255

        temp(i, j, 0., 0) = v;
        temp(i, j, 0., 1) = v;
        temp(i, j, 0., 2) = v;
      }
    }

  temp.save(filename);
}

void save_matrix_as_image(const char* filename, Ref<const MatrixXd> mat, double max)
{
  MatrixXd tmp = mat;
  if(max<0)
    tmp = (tmp.array() - tmp.minCoeff()) / (tmp.maxCoeff() - tmp.minCoeff());
  else
    tmp = (tmp.array() / max);

  save_image(filename, tmp);
}

void
gaussian_blur(Ref<const MatrixXd> in, Ref<MatrixXd> out, int kernel_size)
{
  // mirror border conditions
  auto mirror = [] (int i, int n)
  {
    if(i<0)
      i = (-i) -1;
    if(i>=n)
      i = (n-1) - (i-n);
    return i;
  };

  out.resize( in.size() );
  out.setZero();

  //compute kernel
  VectorXd ker;
  ker.resize(kernel_size); ker.setZero();

  double sigma = double(kernel_size)/5.;
  double sigma2_sq = 2.*sigma*sigma;
  double xc = double(ker.size())/2. - 1.;
  double a = 1./sqrt(M_PI*sigma2_sq);
  for(int i=0; i<ker.size(); ++i){
    double x = double(i)-xc;
    ker(i) = a * exp(-(x*x)/sigma2_sq);
  }
  ker /= ker.sum();

  MatrixXd temp = out;
  // vertical convolution
  for(int j=0; j<in.cols(); ++j){
    for(int i=0; i<in.rows(); ++i){
      double sum = 0;
      for(int k=0; k<ker.size(); ++k)
        sum += in(mirror(i-ker.size()/2 + k,in.rows()), j) * ker(k);
      temp(i, j) = sum;
    }
  }

  // horizontal convolution
  for(int i=0; i<in.rows(); ++i){
    for(int j=0; j<in.cols(); ++j){
      double sum = 0;
      for(int k=0; k<ker.size(); ++k)
        sum += temp(i, mirror(j - ker.size()/2 + k, in.cols())) * ker(k);
      out(i,j) = sum;
    }
  }
}

void make_unit(std::vector<Vector3d> &pts)
{
  Eigen::AlignedBox2d aabb;
  for (auto p: pts)
  {
    aabb.extend(p.head<2>());
  }
  //std::cout << "min: " << aabb.min().transpose() << " ; max: " << aabb.max().transpose() << "\n";
  double scale = 1./(aabb.max()-aabb.min()).maxCoeff();
  for (auto& p: pts)
    p.head<2>() = (p.head<2>()-aabb.min())*scale;
}


void
generate_blue_noise_tile(int n, std::vector<Eigen::Vector2d> &pts, const std::vector<Eigen::Vector2d> &tile)
{
  int nb = tile.size();
  int N = std::sqrt(nb);

  int nb_tiles = (n+N-1)/N;
  double scale = double(N)/double(n);

  int expected_npts = n*n;
  pts.reserve(10*expected_npts/9);
  
  for(int i=0; i<nb_tiles; i++)
  {
    for(int j=0; j<nb_tiles; j++)
    {
      for(int k=0;k<nb;++k)
      {
        Vector2d pt = (Vector2d(double(i),double(j))+tile[k])*scale;
        if(pt.x()<=1. && pt.y()<=1.) {
          pts.push_back(pt);
        }
      }
    }
  }
}

void
generate_blue_noise_tile(int n, std::vector<Eigen::Vector2d> &pts, const std::string &tile_filename)
{
  std::vector<Eigen::Vector2d> tile;
  if(!load_point_cloud_dat(tile_filename, tile))
  {
    std::cerr << "Error loading tile \"" << tile_filename << "\"\n";
    abort();
  }
  generate_blue_noise_tile(n, pts, tile);
}


bool
save_point_cloud_dat(const std::string &filename, const std::vector<Eigen::Vector2d> &points)
{
  std::ofstream file;
  file.open(filename);

  if(file.is_open()){
    
    for(unsigned int i=0; i<points.size(); ++i){
      file << points[i].y()-0.5 << " " << -(points[i].x()-0.5) << "\n";
    }
    
    file.close();

    return true;
  }

  return false;
}

bool
load_point_cloud_dat(const std::string &filename, std::vector<Eigen::Vector2d> &points)
{
  std::ifstream file;
  file.open(filename);

  points.reserve(1000);

  if(file.is_open()){
    Eigen::Vector2d p = Eigen::Vector2d::Zero();
    Eigen::Vector2d q = Eigen::Vector2d::Zero();
    while(!file.eof()) {
      file >> p.x() >> p.y();
      q.x() = 0.5 - p.y();
      q.y() = p.x() + 0.5;
      points.push_back(p);
    }
    
    file.close();

    return true;
  }

  return false;
}

bool
save_point_cloud_eps(const std::string &filename, const std::vector<Eigen::Vector2d> &points, double radius_scale)
{
  const char* footer = "grestore\n";

  std::ofstream file;
  file.open(filename);

  if(file.is_open()){
    file << "%!PS-Adobe-3.1 EPSF-3.0\n";
    file << "%%HiResBoundingBox: 0 0 512 512\n";
    file << "%%BoundingBox: 0 0 512 512\n";
    file << "%%CropBox: 0 0 512 512\n";
    file << "/radius { " << 0.002*radius_scale << " } def\n";
    file << "/p { radius 0 360 arc closepath fill stroke } def\n";
    file << "gsave 512 512 scale\n";
    file << "gsave 0.5 0.5 translate\n";
    file << "0 0 0 setrgbcolor\n";
    for(unsigned int i=0; i<points.size(); ++i){
      file << points[i].y()-0.5 << " " << -(points[i].x()-0.5) << " p \n";
    }
    file << footer;
    file.close();

    return true;
  }

  return false;
}
