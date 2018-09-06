
// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2017-2018 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <Eigen/Core>

namespace Eigen {
  // 2D cross product <=>  cross([a 0], [b 0]).z <=> dot(ortho(a),b)
  inline double cross2(const Vector2d& a, const Vector2d& b) { return a.x()*b.y()-a.y()*b.x(); }

  inline double signed_area(const Vector2d& a, const Vector2d& b, const Vector2d& c) {
    return 0.5 * cross2(b-a,c-a);
  }
  inline double signed_area(const Vector2d& a, const Vector2d& b, const Vector2d& c, const Vector2d& d) {
    return signed_area(a,b,c) + signed_area(a,c,d);
  }

  inline Ref<VectorXd> vec(MatrixXd &mat) {
    return Map<VectorXd>(mat.data(), mat.size());
  }
}
