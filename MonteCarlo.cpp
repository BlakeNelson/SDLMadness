// The MIT License (MIT)
//
// Copyright (c) 2015 Blake Nelson
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "BracketOdds.hpp"
#include "CudaApproach.cuh"
#include <iostream>
#include <limits>

template<unsigned int n>
void calculateOdds(const BracketOdds<n>& bracket, unsigned int pick)
{
  // Create a random number to represent a specific outcome.
}

int main(int argc, char** argv)
{
  if (argc != 4)
  {
    std::cerr << "Usage: MonteCarlos <odds file> <type> <pick>" << std::endl;
  }

  std::string oddsFile(argv[1]);
  std::string type(argv[2]);
  unsigned int pick = boost::lexical_cast<unsigned int>(argv[3]);

  if (type == "Left" || type == "Right")
  {
    auto fullOdds = createFrom538File(oddsFile, type);
    calculateOdds(fullOdds, pick);
  }
  else
  {
    auto r = toRegion(type);
    auto quarterBracket = createQuarterBracketFrom538File(oddsFile, r);
    calculateOdds(quarterBracket, pick);
  }

  return 0;
}
