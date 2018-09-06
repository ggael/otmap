// This file is part of otmap, an optimal transport solver.
//
// Copyright (C) 2017 Georges Nader
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "inputparser.h"

#include <algorithm>
#include <iostream>

InputParser::
InputParser(int& argc, char** argv)
{
  for(int i=1; i<argc; ++i)
    m_tokens.push_back(std::string(argv[i]));
}

int
InputParser::
getCmdOption(const std::string &option, std::vector<std::string> &value) const
{
  std::vector<std::string>::const_iterator itr;
  itr =  std::find(m_tokens.begin(), m_tokens.end(), option);

  if(itr == m_tokens.end() || ++itr== m_tokens.end())
    return 0;

  value.clear();

  do{

    std::string next(*itr);
    if(next.find_first_of("-") == 0)
      break;
    else
      value.push_back(next);

  } while(++itr != m_tokens.end());

  return value.size();
}

bool
InputParser::
cmdOptionExists(const std::string &option) const
{
  return std::find(m_tokens.begin(), m_tokens.end(), option) != m_tokens.end();
}


