// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2018 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <Eigen/Core>
using namespace Eigen;

namespace otmap {

void nestdiss_ordering_impl(int i0, int j0, int rows, int cols, int stride, int* &perm)
{
  int i1 = i0+rows;
  int j1 = j0+cols;
  const int th = 4;
  if(rows<=th || cols<=th)
  {
    for(int i=i0; i<i1; ++i)
      for(int j=j0; j<j1; ++j)
      {
        *perm = i + j*stride;
        ++perm;
      }
  }
  else
  {
    // split in 4
    // insert separators
    int si = i0+rows/2;
    int sj = j0+cols/2;
    for(int i=i0;i<i1;++i)
    {
      *perm = i + stride*sj;
      ++perm;
    }
    for(int j=j0;j<j1;++j)
    {
      if(j!=sj) {
        *perm = si + stride*j;
        ++perm;
      }
    }  
    nestdiss_ordering_impl(i0,   j0,   si-i0,   sj-j0,   stride, perm);
    nestdiss_ordering_impl(si+1, j0,   i1-si-1, sj-j0,   stride, perm);
    nestdiss_ordering_impl(si+1, sj+1, i1-si-1, j1-sj-1, stride, perm);
    nestdiss_ordering_impl(i0,   sj+1, si-i0,   j1-sj-1, stride, perm);
    
  }
}

void nestdiss_ordering(int size, int* perm)
{
  int* p = perm;
  nestdiss_ordering_impl(0, 0, size, size, size, p);
  assert(perm+size*size == p);
  // reverse order (separators are last)
  VectorXi::Map(perm,size*size).reverseInPlace();
}

} // namespace otmap
