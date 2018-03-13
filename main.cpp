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
  //  {
  //    std::cout << "Left" << std::endl;
  //    std::cout << "------------------------" << std::endl;
  //    Five38BracketOdds bracket("fivethirtyeight_ncaa_forecasts.csv", "Left");
  //    runCudaApproach(bracket);
  //  }
  //  {
  //    std::cout << "Right" << std::endl;
  //    std::cout << "------------------------" << std::endl;
  //    Five38BracketOdds bracket("fivethirtyeight_ncaa_forecasts.csv",
  //    "Right");
  //    runCudaApproach(bracket);
  //  }

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
      auto bestPick = fullOdds.findBestPick();
      std::cout << bestPick << std::endl;
    }
    else
    {
      auto r = toRegion(argv[2]);
      auto quarterBracket = createQuarterBracketFrom538File(
        oddsFile, r);
      auto bestPick = quarterBracket.findBestPick();
      std::cout << bestPick << std::endl;
    }
  }
//  if (argc == 2)
//  {
//    if (type == "Left" || type == "Right")
//    {
//      auto fullOdds =
//        createFrom538File("fivethirtyeight_ncaa_forecasts_2018.csv", type);
//      std::cout << "Left odds" << std::endl;
//      auto stats = fullOdds.getOddsStats();
//      std::cout << "Min: " << stats.min << std::endl;
//      std::cout << "Min Pick: " << stats.minPick << std::endl;
//      fullOdds.printBracket(stats.minPick);
//      std::cout << "Max: " << stats.max << std::endl;
//      std::cout << "Max Pick: " << stats.maxPick << std::endl;
//      fullOdds.printBracket(stats.maxPick);
//      std::cout << "Avg: " << stats.avg << std::endl;
//      std::cout << "Std: " << stats.stddev << std::endl;
//    }
//    else
//    {
//      auto r = toRegion(argv[1]);
//      auto quarterBracket = createQuarterBracketFrom538File(
//        "fivethirtyeight_ncaa_forecasts_2018.csv", r);
//      auto quarterOdds = quarterBracket.getOddsStats();
//      std::cout << "West odds" << std::endl;
//      std::cout << "Min: " << quarterOdds.min << std::endl;
//      std::cout << "Min Pick: " << quarterOdds.minPick << std::endl;
//      quarterBracket.printBracket(quarterOdds.minPick);
//      std::cout << "Max: " << quarterOdds.max << std::endl;
//      std::cout << "Max Pick: " << quarterOdds.maxPick << std::endl;
//      quarterBracket.printBracket(quarterOdds.maxPick);
//      std::cout << "Avg: " << quarterOdds.avg << std::endl;
//      std::cout << "Std: " << quarterOdds.stddev << std::endl;
//    }
//  }
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
