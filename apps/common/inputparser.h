// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2017 Georges Nader
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <string>
#include <vector>

class InputParser
{
public:
  InputParser(int& argc, char** argv);

  int getCmdOption(const std::string& option, std::vector< std::string >& value) const;
  bool cmdOptionExists(const std::string& option) const;

private:
  std::vector< std::string > m_tokens;
};

