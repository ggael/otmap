// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2017-2018 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2017 Georges Nader
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <Eigen/Core>
#include <surface_mesh/Surface_mesh.h>

#include <map>
#include <ostream>
#include <string>


// Image IO ----------------

// Load an image as a matrix (colors are converted to gray levels
void load_image(const char* filename, Eigen::MatrixXd &img);

// Save a matrix as a gray level image
void save_image(const char* filename, Eigen::Ref<const Eigen::MatrixXd> img);

// Save a matrix as a 2D gray level image normalized to 0-1 (if max<0) or rescaled to max otherwise:
void save_matrix_as_image(const char* filename, Eigen::Ref<const Eigen::MatrixXd> img, double max=-1.);

// Image operations --------

void gaussian_blur(Eigen::Ref<const Eigen::MatrixXd> in, Eigen::Ref<Eigen::MatrixXd> out, int kernel_size=5);

// sampling ----------------

bool save_point_cloud_dat(const std::string& filename,
                          const std::vector<Eigen::Vector2d>& points);

bool load_point_cloud_dat(const std::string& filename,
                          std::vector<Eigen::Vector2d>& points);

bool save_point_cloud_eps(const std::string& filename,
                          const std::vector<Eigen::Vector2d>& points,
                          double radius_scale = 1);

void
make_unit(std::vector<Eigen::Vector3d> &pts);

void
generate_blue_noise_tile(int n, std::vector<Eigen::Vector2d>& pts, const std::string &tile_filename);

void
generate_blue_noise_tile(int n, std::vector<Eigen::Vector2d> &pts, const std::vector<Eigen::Vector2d> &tile);

