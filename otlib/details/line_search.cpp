// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2017-2018 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "line_search.h"
#include <iostream>
#include <Eigen/LU>

using namespace Eigen;

namespace otmap {

void QuadraticLineSearch::init() {
  m_verbose_level = 0;
  m_width_tolerance = 1e-2;
  m_low_boundary = 0;
  m_high_boundary = 1;
}

double
QuadraticLineSearch::
operator()(Ref<const VectorXd> xk, Ref<const VectorXd> dir,
           Ref<VectorXd> xk1, Ref<VectorXd> rk1,
           double ek, double *palpha) const
{
  int itr = 0;
  const int max_iter = 12;
  m_cache_residual.resize(xk.size());

  double low  = m_low_boundary, rlow;
  if(low==0. && ek>=0.) rlow = ek;
  else                  rlow = m_func(xk1 = xk+low *  dir, rk1);
  double high = 1.0,    rhigh = m_func(xk1 = xk+high*dir, rk1);
  
  double mid = 0.5,     rmid  = m_func(xk1 = xk+mid*dir, rk1);
  double alpha = mid;

  double prev = mid;
  bool do_not_stop = false;
  do {
    do_not_stop = false;
    prev = mid;

    // Compute the quadratic polynomial interpolating f at low, mid, high:
    Matrix3d A;
    A << 1, low,  numext::abs2(low),
         1, mid,  numext::abs2(mid),
         1, high, numext::abs2(high);
    Vector3d b;
    b << rlow, rmid, rhigh;
    Vector3d q = A.inverse()*b;

    if(q(2)<0)
    {
      // the fit is concave!
      if(m_verbose_level>=2)
      {
        std::cerr << "\t - line-search - quadratic_line_search, the quadratic approximation is concave at iteration " << itr << "\n";
        std::cerr << "\t - line-search - with q = " << q.transpose() << " ; values: " <<  rlow << " , " << rmid << " , " << rhigh << "  at " << low << " " << mid << " " << high << "\n";
      }

      // as a heuristic, let's pick the interval whose residual is minimal at its middle position
      // and enforce an additional iteration
      double low_mid = (low+mid)/2.;
      double high_mid = (mid+high)/2.;
      double r_low_mid  = m_func(xk1 = xk+low_mid*dir, m_cache_residual);
      double r_high_mid = m_func(xk1 = xk+high_mid*dir, rk1);
      if(r_low_mid < r_high_mid)
      {
        xk1 = xk+low_mid*dir;
        rk1.swap(m_cache_residual);
        high = mid;
        rhigh = rmid;
        alpha = mid = low_mid;
        rmid = r_low_mid;
      }
      else
      {
        low = mid;
        rlow = rmid;
        alpha = mid = high_mid;
        rmid = r_high_mid;
      }
      
      do_not_stop = true;
      continue;
    }

    // compute the minimum
    alpha = -q(1)/(2.*q(2));

    if(m_verbose_level>=9)
      std::cerr << "\t - line-search - q=" << q.transpose() << " ; values: " <<  rlow << " , " << rmid << " , " << rhigh << "  at " << low << " " << mid << " " << high << " alpha = " << alpha << "\n";

    if(alpha>=high)
    {
      if(m_verbose_level>=6)
        std::cout << "\t - line-search - enlarge search range\n";
      // Then we have rlow>rmid>rhigh

      // The minimum is very likely above high,
      // so let's enlarge the search domain,
      // but only if f(alpha) is indeed better than what we have at hand
      m_cache_residual.swap(rk1);
      double ralpha = m_func(xk1 = xk+alpha*dir, rk1);
      if(ralpha<rhigh)
      {
        // good guess
        low = high;
        rlow = rhigh;
        mid = alpha;
        rmid = ralpha;
        high = high + 2.*(alpha-high);
        rhigh = m_func(xk1 = xk+high*dir, m_cache_residual);
        xk1 = xk+alpha*dir;

        do_not_stop = true;
        continue;
      }
      else
      {
        // we got a bad fit, continue with mid-high-alpha -> low-mid-high
        low = mid;
        rlow = rmid;
        mid = high;
        rmid = mid;
        high = alpha;
        rhigh = ralpha;
      }
    }
    else
    {

      if(alpha<=low) { alpha = (low+mid)/2.; }
      if(alpha>=high) { alpha = (mid+high)/2.; }

      // save previous residual in case we made a bad guess:
      m_cache_residual.swap(rk1);
      double ralpha = m_func(xk1 = xk+alpha*dir, rk1);

      // we now have 4 samples, let's discard the impossible segment
      if(alpha<mid)
      {
        if(ralpha<rmid) // good guess
        {
          high = mid;    rhigh  = rmid;
          mid  = alpha;  rmid   = ralpha;
        }
        else // bad guess
        {
          // restore xk1 and rk1:
          xk1 = xk+mid*dir;
          m_cache_residual.swap(rk1);
          low = alpha;   rlow   = ralpha;
        }
      }
      else // if(alpha>mid)
      {
        if(ralpha<rmid) // good guess
        {
          low = mid;     rlow   = rmid;
          mid = alpha;   rmid   = ralpha;
        }
        else // bad guess
        {
          // restore xk1 and rk1:
          xk1 = xk+mid*dir;
          m_cache_residual.swap(rk1);
          high = alpha;  rhigh  = ralpha;
        }
      }
    }

    ++itr;
  } while((do_not_stop || std::abs(mid-prev)>m_width_tolerance) && itr<max_iter);

  if(m_verbose_level>=8)
    std::cout << "\t - line-search - best residual=" << rmid << ", alpha=" << alpha << ", itr=" << itr << std::endl;

  if(palpha)
    *palpha = alpha;
  return rmid;
}

} // namespace otmap
