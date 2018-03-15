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

int main(int argc, char** argv)
{
  if (argc != 3 && argc != 4)
  {
    std::cerr << "Usage: SDLMadness <odds file> <type>" << std::endl;
  }

  std::string oddsFile(argv[1]);
  std::string type(argv[2]);
  std::cout.precision(std::numeric_limits<Float>::digits10);
  if (argc == 3)
  {
    if (type == "Left" || type == "Right")
    {
      auto fullOdds =
        createFrom538File(oddsFile, type);
      auto bestPick = fullOdds.findBestPickBackwards();
      std::cout << bestPick << std::endl;
      fullOdds.printPick(bestPick);
      std::cout << "Final score is " << fullOdds.chanceOfPickOccurring(bestPick) *
                   fullOdds.pickUnweightedScore(bestPick) << std::endl;
      fullOdds.tracePickUnweightedScore(bestPick);
    }
    else
    {
      auto r = toRegion(argv[2]);
      auto quarterBracket = createQuarterBracketFrom538File(
        oddsFile, r);
      auto bestPick = quarterBracket.findBestPickBackwards();
      std::cout << bestPick << std::endl;
      quarterBracket.printPick(bestPick);
      std::cout << "Final score is " << quarterBracket.chanceOfPickOccurring(bestPick) *
                   quarterBracket.pickUnweightedScore(bestPick) << std::endl;
      quarterBracket.tracePickUnweightedScore(bestPick);
    }
  }
  else
  {
    if (type == "Left" || type == "Right")
    {
      auto fullOdds =
        createFrom538File(oddsFile, type);

      unsigned int pick =boost::lexical_cast<unsigned int>(argv[3]);
      fullOdds.traceOdds(pick);
      std::cout << "Score = " << fullOdds.pickScore(pick) << std::endl;
    }
    else
    {
      auto r = toRegion(argv[2]);
      auto quarterBracket = createQuarterBracketFrom538File(
        oddsFile, r);
      unsigned int pick =boost::lexical_cast<unsigned int>(argv[3]);
      quarterBracket.traceOdds(pick);
      std::cout << "Score = " << quarterBracket.pickScore(pick) << std::endl;
    }
  }
  return 0;
}
