// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2018 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <string>
#include <vector>
#include <Eigen/Core>
#include <surface_mesh/Surface_mesh.h>

#include "otsolver_options.h"
#include "transport_map.h"

bool load_input_density(const std::string& filename, Eigen::MatrixXd& density);

void generate_transport_maps(const std::vector<std::string>& inputs, std::vector<otmap::TransportMap>& tmaps, const CLI_OTSolverOptions& opts,
                             std::function<void(Eigen::MatrixXd&)> filter = [](Eigen::MatrixXd&){});

void synthetize_and_export_image(const surface_mesh::Surface_mesh& map, int img_res, const Eigen::VectorXd& target, const std::string base_filename, const Eigen::VectorXd& input_density = Eigen::VectorXd(0), double gamma = 1);

