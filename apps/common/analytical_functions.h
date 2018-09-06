// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2017-2018 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2017 Georges Nader
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <vector>
#include <Eigen/Core>
#include <surface_mesh/Surface_mesh.h>

// ANALYTICAL FUNCTIONS -------------------------------------------------------
// a set of analytical function for compartive purposes

namespace FuncName {
  enum Enum {

    CONSTANT       =   1,
    LIN1D          =   5,
    CIRCLES_BOUND  =   6,
    SINGULARITY1   =   7,
    CIRCLES        =   8,
    SINGLE_BLACK   =  10,
    SINGLE_WHITE   =  11,

    BFO12EX        =  21,   // section 6.1
    BFO12ELLIPSE1  =  22,   // section 6.2
    BFO12ELLIPSE2  =  23,   // section 6.2
    BFO12_1GAUSS   =  24,   // section 6.4
    BFO12_4GAUSS   =  25,   // section 6.4

    DCFCL08EX1     =  31,   // smooth
    DCFCL08EX2     =  32,   // spiral
    DCFCL08EX3     =  33,   // 1/dirac

    SINCOS1        = 211,
    SINCOS3        = 213,
    SINCOS5        = 215,

    BRDF0          = 300,
    BRDF1          = 301,
    BRDF2          = 302,
    BRDF3          = 303

  };
}

/** computes an analytical scalar function on 2D regular grid
  * the function is evaluated at the center of grid cells
  */
void eval_func_to_grid(Eigen::Ref<Eigen::MatrixXd> density, int fn = 1);

/** evaluate given function \a fn at given coordinates \a p */
double func(FuncName::Enum fn, const Eigen::Vector2d& p);
