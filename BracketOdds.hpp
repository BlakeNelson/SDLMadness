#ifndef BRACKET_ODDS_HPP
#define BRACKET_ODDS_HPP

#include <set>
#include <map>
#include <fstream>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <algorithm>
#include <iostream>
#include <thread>
#include <future>
#include <fstream>

using Float = double;
// boost::multiprecision::number<boost::multiprecision::cpp_bin_float<1000>>;

namespace
{
  const std::string MIDWEST("Midwest");
  const std::string WEST("West");
  const std::string EAST("East");
  const std::string SOUTH("South");
  const std::string RIGHT("Right");
  const std::string LEFT("Left");
  const std::string TOP("Top");
  const std::string BOTTOM("Bottom");

  std::map<std::string, unsigned int> createHeaderMap(
    const std::string& headerRow)
  {
    std::map<std::string, unsigned int> result;
    boost::char_separator<char> sep(",");
    boost::tokenizer<boost::char_separator<char>> tokens(headerRow, sep);
    unsigned int colIdx = 0;
    for (const auto& token : tokens)
    {
      result[token] = colIdx;
      ++colIdx;
    }
    return result;
  }

  std::vector<std::string> createValueMap(const std::string& row)
  {
    std::vector<std::string> result;
    boost::char_separator<char> sep(",");
    boost::tokenizer<boost::char_separator<char>> tokens(row, sep);
    for (const auto& token : tokens)
    {
      result.push_back(token);
    }
    return result;
  }

  bool getEntryIsValid(const std::string& rowRegion, const std::string& region)
  {

    if (region == LEFT)
    {
      return rowRegion == SOUTH || rowRegion == WEST;
    }
    else if (region == RIGHT)
    {
      return rowRegion == EAST || rowRegion == MIDWEST;
    }
    else if (region == TOP)
    {
      return rowRegion == EAST || rowRegion == MIDWEST;
    }
    else if (region == BOTTOM)
    {
      return rowRegion == SOUTH || rowRegion == WEST;
    }
    else
    {
      return rowRegion == region;
    }
  }

  unsigned int getNumberOfRounds(const std::string& region)
  {
    if (region == LEFT || region == RIGHT) return 5;
    if (region == TOP || region == BOTTOM) return 5;
    return 4;
  }

  unsigned int getTeamNumber(unsigned int seed,
                             const std::string& rowRegion,
                             const std::string& region)
  {
    unsigned int result;
    switch (seed)
    {
    case 1:
      result = 0;
      break;

    case 16:
      result = 1;
      break;

    case 8:
      result = 2;
      break;

    case 9:
      result = 3;
      break;

    case 5:
      result = 4;
      break;

    case 12:
      result = 5;
      break;

    case 4:
      result = 6;
      break;

    case 13:
      result = 7;
      break;

    case 6:
      result = 8;
      break;

    case 11:
      result = 9;
      break;

    case 3:
      result = 10;
      break;

    case 14:
      result = 11;
      break;

    case 7:
      result = 12;
      break;

    case 10:
      result = 13;
      break;

    case 2:
      result = 14;
      break;

    case 15:
      result = 15;
      break;
    }

    if (region == LEFT && rowRegion == WEST)
    {
      result += 16;
    }
    if (region == RIGHT && rowRegion == MIDWEST)
    {
      result += 16;
    }
    if (region == TOP && rowRegion == MIDWEST)
    {
      result += 16;
    }
    if (region == BOTTOM && rowRegion == SOUTH)
    {
      result += 16;
    }
    return result;
  }

  unsigned int getTeamNumber(unsigned int seed)
  {
    unsigned int result;
    switch (seed)
    {
    case 1:
      result = 0;
      break;

    case 16:
      result = 1;
      break;

    case 8:
      result = 2;
      break;

    case 9:
      result = 3;
      break;

    case 5:
      result = 4;
      break;

    case 12:
      result = 5;
      break;

    case 4:
      result = 6;
      break;

    case 13:
      result = 7;
      break;

    case 6:
      result = 8;
      break;

    case 11:
      result = 9;
      break;

    case 3:
      result = 10;
      break;

    case 14:
      result = 11;
      break;

    case 7:
      result = 12;
      break;

    case 10:
      result = 13;
      break;

    case 2:
      result = 14;
      break;

    case 15:
      result = 15;
      break;
    }

    return result;
  }

  unsigned int getGameNumber(unsigned int seed,
                             unsigned int round,
                             const std::string& rowRegion,
                             const std::string& region)
  {
    unsigned int result;
    unsigned int divisor = 0x01 << (round - 1);
    switch (seed)
    {
    case 1:
    case 16:
      result = 0 / divisor;
      break;

    case 8:
    case 9:
      result = 1 / divisor;
      break;

    case 5:
    case 12:
      result = 2 / divisor;
      break;

    case 4:
    case 13:
      result = 3 / divisor;
      break;

    case 6:
    case 11:
      result = 4 / divisor;
      break;

    case 3:
    case 14:
      result = 5 / divisor;
      break;

    case 7:
    case 10:
      result = 6 / divisor;
      break;

    case 2:
    case 15:
      result = 7 / divisor;
      break;
    }

    if (region == LEFT && rowRegion == WEST)
    {
      result += 8 / divisor;
    }
    if (region == RIGHT && rowRegion == MIDWEST)
    {
      result += 8 / divisor;
    }
    if (region == TOP && rowRegion == MIDWEST)
    {
      result += 8 / divisor;
    }
    if (region == BOTTOM && rowRegion == SOUTH)
    {
      result += 8 / divisor;
    }
    return result;
  }
}

unsigned int getGameNumber(unsigned int seed, unsigned int round)
{
  unsigned int result;
  unsigned int divisor = 0x01 << (round - 1);
  switch (seed)
  {
  case 1:
  case 16:
    result = 0 / divisor;
    break;

  case 8:
  case 9:
    result = 1 / divisor;
    break;

  case 5:
  case 12:
    result = 2 / divisor;
    break;

  case 4:
  case 13:
    result = 3 / divisor;
    break;

  case 6:
  case 11:
    result = 4 / divisor;
    break;

  case 3:
  case 14:
    result = 5 / divisor;
    break;

  case 7:
  case 10:
    result = 6 / divisor;
    break;

  case 2:
  case 15:
    result = 7 / divisor;
    break;
  }

  return result;
}

// This class tracks odds by round/game, but doesn't really care about the
// connections.  Connections will all be implicit, all that really matters is
// if the guess is right or not.
class Five38BracketOdds
{
public:
  Five38BracketOdds() {}

  Five38BracketOdds(const std::string& filePath, const std::string& region)
  {
    // The file is a tab-delimited file, with first row header.
    // I will want to slice and dice the bracket construction, so I would like
    // a boolean callback.  Although I really only need single region or left
    // right.
    // Midwest/West are left, East/South are right.
    std::ifstream inFile(filePath.c_str());
    std::string headerLine;
    std::getline(inFile, headerLine);

    // Create a header/column number map.
    auto headerMap = createHeaderMap(headerLine);

    for (std::string line; std::getline(inFile, line);)
    {
      if (!line.empty() && line[0] == '#') continue;
      // Create column number/value map.
      auto values = createValueMap(line);

      // Populate the data.
      auto rowRegion = values[headerMap["team_region"]];
      auto name = values[headerMap["team_name"]];

      if (!getEntryIsValid(rowRegion, region)) continue;

      auto seed =
        boost::lexical_cast<unsigned int>(values[headerMap["team_seed"]]);
      auto teamNumber = getTeamNumber(seed, rowRegion, region);
      m_seeds[teamNumber] = seed;
      m_names[teamNumber] = name;
      double roundOdds[] = {
        boost::lexical_cast<double>(values[headerMap["rd2_win"]]),
        boost::lexical_cast<double>(values[headerMap["rd3_win"]]),
        boost::lexical_cast<double>(values[headerMap["rd4_win"]]),
        boost::lexical_cast<double>(values[headerMap["rd5_win"]]),
        boost::lexical_cast<double>(values[headerMap["rd6_win"]]),
        boost::lexical_cast<double>(values[headerMap["rd7_win"]])};

      unsigned int numRounds = getNumberOfRounds(region);
      for (unsigned int round = 0; round < numRounds; ++round)
      {
        auto game = getGameNumber(seed, round + 1, rowRegion, region);
        m_roundOdds[round][teamNumber] = roundOdds[round];
      }
    }
    inFile.close();
  }

  std::array<std::array<float, 32>, 6> m_roundOdds{};
  std::array<float, 32> m_seeds{};
  std::array<std::string, 32> m_names{};
};

inline std::ostream& operator<<(std::ostream& os, const Five38BracketOdds& odds)
{
  //  for (const auto idxIter : odds.m_odds)
  //  {
  //    os << idxIter.first << std::endl;
  //    for (const auto& gameIter : idxIter.second)
  //    {
  //      os << "  " << gameIter.first.name << ", " << gameIter.second <<
  //      std::endl;
  //    }
  //  }
  return os;
}

// Bitfield representing picks.  A 0 indicates that the first team wins, a
// 1 indicates the second team wins.
struct Pick
{
  Pick() : value(0) {}
  Pick(unsigned int v) : value(v) {}

  unsigned int value;
};

struct BestPick
{
  void updateMinMax(Float score, unsigned int pick)
  {
    if (score < minScore)
    {
      minScore = score;
      minPick = pick;
    }

    if (score > maxScore)
    {
      maxScore = score;
      maxPick = pick;
    }
  }
  unsigned int minPick{0};
  unsigned int maxPick{0};

  Float minScore{std::numeric_limits<Float>::max()};
  Float maxScore{std::numeric_limits<Float>::min()};
};

struct Stats
{
  void updateMinMax(Float score, unsigned int pick)
  {
    if (score < min)
    {
      min = score;
      minPick = pick;
    }

    if (score > max)
    {
      max = score;
      maxPick = pick;
    }
  }
  Float min{std::numeric_limits<Float>::max()};
  Float max{std::numeric_limits<Float>::min()};
  unsigned int minPick{0};
  unsigned int maxPick{0};
  Float avg{0.0};
  Float stddev{0.0};
};

/** Structure to track a bracket and associated odds. */
template <unsigned int TeamCount>
struct BracketOdds
{
  static constexpr unsigned int RoundCount = std::log2(TeamCount);

  void printBracket(Pick pick)
  {
    for (unsigned int round = 0; round < RoundCount; ++round)
    {
      std::cout << "Round " << round << std::endl;
      for (unsigned int game = 0; game < gamesInRound(round); ++game)
      {
        auto winningTeam = getWinningTeamIdx(pick, round, game);
        std::cout << names[winningTeam] << " (" << seeds[winningTeam] << ")"
                  << std::endl;
      }
    }
  }

  void printPick(unsigned int pick)
  {
    for (unsigned int round = 0; round < RoundCount; ++round)
    {
      std::cout << std::endl;
      std::cout << "Round " << round << std::endl;
      std::cout << "====================" << std::endl;
      for (unsigned int game = 0; game < gamesInRound(round); ++game)
      {

        auto winningTeam = getWinningTeamIdx(pick, round, game);
        std::cout << names[winningTeam] << " (" << seeds[winningTeam] << ")"
                  << std::endl;
      }
    }
  }

  void traceOdds(unsigned int pick)
  {
    Float result = 1.0;
    for (unsigned int round = 0; round < RoundCount; ++round)
    {
      for (unsigned int game = 0; game < gamesInRound(round); ++game)
      {
        auto teamIndices = getTeamIndices(pick, round, game);
        auto winningTeam = getWinningTeamIdx(pick, round, game);
        auto loosingTeam = teamIndices.first;
        if (loosingTeam == winningTeam)
        {
          loosingTeam = teamIndices.second;
        }

        auto winningTeamRawOdds = odds[round * TeamCount + winningTeam];
        auto loosingTeamRawOdds = odds[round * TeamCount + loosingTeam];
        result *=
          winningTeamRawOdds / (winningTeamRawOdds + loosingTeamRawOdds);
        std::cout << "Round " << round << " game " << game << " between "
                  << names[teamIndices.first] << " and "
                  << names[teamIndices.second] << std::endl;
        std::cout << "Piced winner is " << names[winningTeam] << std::endl;
        std::cout << "Winner odds = " << winningTeamRawOdds << std::endl;
        std::cout << "Loser odds = " << loosingTeamRawOdds << std::endl;
        std::cout << "Scaled winner odds = "
                  << winningTeamRawOdds /
                       (winningTeamRawOdds + loosingTeamRawOdds)
                  << std::endl;
      }
    }
    std::cout << "Final odds " << result << std::endl;

    for (unsigned int round = 0; round < RoundCount; ++round)
    {
      std::cout << std::endl;
      std::cout << "Round " << round << std::endl;
      std::cout << "====================" << std::endl;
      for (unsigned int game = 0; game < gamesInRound(round); ++game)
      {

        auto winningTeam = getWinningTeamIdx(pick, round, game);
        std::cout << names[winningTeam] << " (" << seeds[winningTeam] << ")"
                  << std::endl;
      }
    }
  }

  Float sumOdds()
  {
    auto numThreads = std::thread::hardware_concurrency();
    Float result;
    unsigned int n = (0x01u << (TeamCount - 1));
    if (n < 8)
    {
      numThreads = 1;
    }

    ///////////////////
    // For debugging
    // numThreads = 1;
    // std::ofstream traceFile("trace.txt");
    ////
    std::vector<std::future<Float>> partialSums;
    for (unsigned int tid = 0; tid < numThreads; ++tid)
    {
      unsigned int picksPerThread = n / numThreads;
      unsigned int startPick = tid * picksPerThread;
      unsigned int endPick = startPick + picksPerThread;
      if (tid == numThreads - 1)
      {
        endPick = n;
      }

      partialSums.push_back(
        // std::async(std::launch::async, [this, startPick, endPick,
        // &traceFile]() {
        std::async(std::launch::async, [this, startPick, endPick]() {
          Float result;
          for (unsigned int pick = startPick; pick < endPick; ++pick)
          {
            if (pick % 100000 == 0)
            {
              std::cout << static_cast<Float>(pick - startPick) /
                             (endPick - startPick)
                        << std::endl;
            }
            auto gameOdds = chanceOfPickOccurring(Pick(pick));
            result += gameOdds;
          }
          return result;
        }));
    }

    for (auto&& f : partialSums)
    {
      result += f.get();
    }

    return result;
  }

  /** The games give a lot of points for correct picks, so much so that I
   * think
   *  highly unlikely scenarios can bias the odds.  This method will evaluate
   *  the odds of all picks, then return a range of odds that are valid.
   */
  Stats getOddsStats()
  {
    auto numThreads = std::thread::hardware_concurrency();
    Stats result;
    unsigned int n = (0x01u << (TeamCount - 1));
    if (n < 8)
    {
      numThreads = 1;
    }

    ///////////////////
    // For debugging
    // numThreads = 1;
    // std::ofstream traceFile("trace.txt");
    ////
    std::vector<std::future<Stats>> partialSums;
    for (unsigned int tid = 0; tid < numThreads; ++tid)
    {
      unsigned int picksPerThread = n / numThreads;
      unsigned int startPick = tid * picksPerThread;
      unsigned int endPick = startPick + picksPerThread;
      if (tid == numThreads - 1)
      {
        endPick = n;
      }

      partialSums.push_back(
        // std::async(std::launch::async, [this, startPick, endPick,
        // &traceFile]() {
        std::async(std::launch::async, [this, startPick, endPick]() {
          Stats result;
          for (unsigned int pick = startPick; pick < endPick; ++pick)
          {
            if (pick % 100000 == 0)
            {
              std::cout << static_cast<Float>(pick - startPick) /
                             (endPick - startPick)
                        << std::endl;
            }
            auto gameOdds = chanceOfPickOccurring(Pick(pick));
            // traceFile << gameOdds << std::endl;
            result.updateMinMax(gameOdds, pick);
            result.avg += gameOdds;
          }
          return result;
        }));
    }

    for (auto&& f : partialSums)
    {
      auto partialStats = f.get();
      result.avg += partialStats.avg;
      if (partialStats.min < result.min)
      {
        result.min = partialStats.min;
        result.minPick = partialStats.minPick;
      }

      if (partialStats.max > result.max)
      {
        result.max = partialStats.max;
        result.maxPick = partialStats.maxPick;
      }
    }
    result.avg = result.avg / static_cast<Float>(n);

    return result;
  }

  unsigned int findBestPickBackwards()
  {
    auto numThreads = std::thread::hardware_concurrency();
    BestPick result;
    unsigned int n = (0x01u << (TeamCount - 1));
    if (n < 8)
    {
      numThreads = 1;
    }

    ///////////////////
    // For debugging
    // numThreads = 1;
    // std::ofstream traceFile("trace.txt");
    ////
    std::vector<std::future<BestPick>> partialPicks;
    for (unsigned int tid = 0; tid < numThreads; ++tid)
    {
      unsigned int picksPerThread = n / numThreads;
      unsigned int startPick = tid * picksPerThread;
      unsigned int endPick = startPick + picksPerThread;
      if (tid == numThreads - 1)
      {
        endPick = n;
      }

      partialPicks.push_back(
        // std::async(std::launch::async, [this, startPick, endPick,
        // &traceFile]() {
        std::async(std::launch::async, [this, startPick, endPick]() {
          BestPick result;
          for (unsigned int pick = startPick; pick < endPick; ++pick)
          {
            if (pick % 100000 == 0)
            {
              std::cout << static_cast<Float>(pick - startPick) /
                             (endPick - startPick)
                        << std::endl;
            }
            auto odds = chanceOfPickOccurringBackward(pick);
            if (odds > 0)
            {
              auto score = odds * pickUnweightedScore(pick);
              result.updateMinMax(score, pick);
            }
          }
          return result;
        }));
    }

    for (auto&& f : partialPicks)
    {
      auto partialStats = f.get();
      if (partialStats.minScore < result.minScore)
      {
        result.minScore = partialStats.minScore;
        result.minPick = partialStats.minPick;
      }

      if (partialStats.maxScore > result.maxScore)
      {
        result.maxScore = partialStats.maxScore;
        result.maxPick = partialStats.maxPick;
      }
    }

    // traceOdds(result.maxPick);
    std::cout << "Best pick = " << result.maxPick << std::endl;
    return result.maxPick;
  }

  unsigned int findBestPick()
  {
    auto numThreads = std::thread::hardware_concurrency();
    Stats result;
    unsigned int n = (0x01u << (TeamCount - 1));
    if (n < 8)
    {
      numThreads = 1;
    }

    ///////////////////
    // For debugging
    // numThreads = 1;
    // std::ofstream traceFile("trace.txt");
    ////
    std::vector<std::future<Stats>> partialSums;
    for (unsigned int tid = 0; tid < numThreads; ++tid)
    {
      unsigned int picksPerThread = n / numThreads;
      unsigned int startPick = tid * picksPerThread;
      unsigned int endPick = startPick + picksPerThread;
      if (tid == numThreads - 1)
      {
        endPick = n;
      }

      partialSums.push_back(
        // std::async(std::launch::async, [this, startPick, endPick,
        // &traceFile]() {
        std::async(std::launch::async, [this, startPick, endPick]() {
          Stats result;
          for (unsigned int pick = startPick; pick < endPick; ++pick)
          {
            if (pick % 100000 == 0)
            {
              std::cout << static_cast<Float>(pick - startPick) /
                             (endPick - startPick)
                        << std::endl;
            }
            auto odds = chanceOfPickOccurring(pick);
            if (odds > 0)
            {
              auto score = pickScore(Pick(pick));
              result.updateMinMax(score, pick);
              result.avg += 1.0;
            }
          }
          return result;
        }));
    }

    for (auto&& f : partialSums)
    {
      auto partialStats = f.get();
      if (partialStats.min < result.min)
      {
        result.min = partialStats.min;
        result.minPick = partialStats.minPick;
      }

      if (partialStats.max > result.max)
      {
        result.max = partialStats.max;
        result.maxPick = partialStats.maxPick;
      }
      result.avg += partialStats.avg;
    }

    traceOdds(result.maxPick);
    std::cout << "Best pick = " << result.maxPick << std::endl;
    std::cout << "Considered picks = " << result.avg << std::endl;
    return result.maxPick;
  }

  Float pickScoreOriginalMethod(Pick pick)
  {
    Float result = 1.0;
    Float weight = 1.0;
    for (unsigned int round = 0; round < RoundCount; ++round)
    {
      for (unsigned int game = 0; game < gamesInRound(round); ++game)
      {
        auto winningTeam = getWinningTeamIdx(pick, round, game);
        auto winningTeamRawOdds = odds[round * TeamCount + winningTeam];
        result += weight * seeds[winningTeam] * winningTeamRawOdds;
      }
      weight *= 2.0;
    }
    return result;
  }

  Float pickScore(Pick pick)
  {
    Float result = 1.0;
    Float weight = 1.0;
    for (unsigned int round = 0; round < RoundCount; ++round)
    {
      for (unsigned int game = 0; game < gamesInRound(round); ++game)
      {
        auto teamIndices = getTeamIndices(pick, round, game);
        auto winningTeam = getWinningTeamIdx(pick, round, game);
        auto loosingTeam = teamIndices.first;
        if (loosingTeam == winningTeam)
        {
          loosingTeam = teamIndices.second;
        }

        auto winningTeamRawOdds = odds[round * TeamCount + winningTeam];
        auto loosingTeamRawOdds = odds[round * TeamCount + loosingTeam];
        result += weight * seeds[winningTeam] * winningTeamRawOdds /
                  (winningTeamRawOdds + loosingTeamRawOdds);
        // result += weight * seeds[winningTeam] * winningTeamRawOdds;
      }
      weight *= 2.0;
    }
    return result;
  }

  void tracePickUnweightedScore(Pick pick)
  {
    Float result = 0.0;
    Float weight = 1.0;
    for (unsigned int round = 0; round < RoundCount; ++round)
    {
      for (unsigned int game = 0; game < gamesInRound(round); ++game)
      {
        auto winningTeam = getWinningTeamIdx(pick, round, game);
        std::cout << names[winningTeam] << "(" << seeds[winningTeam] << ") - "
                  << result << " += " << weight * seeds[winningTeam]
                  << std::endl;
        result += weight * seeds[winningTeam];
      }
      weight *= 2.0;
    }
  }

  Float pickUnweightedScore(Pick pick)
  {
    Float result = 0.0;
    Float weight = 1.0;
    for (unsigned int round = 0; round < RoundCount; ++round)
    {
      for (unsigned int game = 0; game < gamesInRound(round); ++game)
      {
        auto winningTeam = getWinningTeamIdx(pick, round, game);
        result += weight * seeds[winningTeam];
      }
      weight *= 2.0;
    }
    return result;
  }

  Float chanceOfPickOccurringOriginalMethod(Pick pick)
  {
    Float result = 1.0;
    for (unsigned int round = 0; round < RoundCount; ++round)
    {
      for (unsigned int game = 0; game < gamesInRound(round); ++game)
      {
        auto winningTeam = getWinningTeamIdx(pick, round, game);
        auto winningTeamRawOdds = odds[round * TeamCount + winningTeam];
        result *= winningTeamRawOdds;
      }
    }
    return result;
  }

  Float chanceOfPickOccurring(Pick pick)
  {
    Float result = 1.0;
    for (unsigned int round = 0; round < RoundCount; ++round)
    {
      for (unsigned int game = 0; game < gamesInRound(round); ++game)
      {
        auto teamIndices = getTeamIndices(pick, round, game);
        auto winningTeam = getWinningTeamIdx(pick, round, game);
        auto loosingTeam = teamIndices.first;
        if (loosingTeam == winningTeam)
        {
          loosingTeam = teamIndices.second;
        }

        auto winningTeamRawOdds = odds[round * TeamCount + winningTeam];
        auto loosingTeamRawOdds = odds[round * TeamCount + loosingTeam];
        // result *= winningTeamRawOdds;
        result *=
          winningTeamRawOdds / (winningTeamRawOdds + loosingTeamRawOdds);
      }
    }
    return result;
  }

  Float chanceOfPickOccurringBackward(Pick pick)
  {
    auto ultimateWinnerIdx = getWinningTeamIdx(pick, RoundCount - 1, 0);

    Float result = odds[(RoundCount - 1) * TeamCount + ultimateWinnerIdx];

    for (unsigned int round = 0; round < RoundCount; ++round)
    {
      for (unsigned int game = 0; game < gamesInRound(round); ++game)
      {
        auto winningTeam = getWinningTeamIdx(pick, round, game);
        if (winningTeam == ultimateWinnerIdx) continue;

        auto winningTeamRawOdds = odds[round * TeamCount + winningTeam];
        result *= winningTeamRawOdds;
      }
    }
    return result;
  }

  Float chanceOfPickOccurringBackwardTree(Pick pick)
  {
    auto ultimateWinnerIdx = getWinningTeamIdx(pick, RoundCount - 1, 0);
    Float result = odds[(RoundCount - 1) * TeamCount + ultimateWinnerIdx];

    // I have two children.  In one child, the ultimate winner is also the
    // winner of the subtree.  In the other, I don't know.
    Float firstSubtreeOdds = chanceOfPickOccurringBackwardTree(
      pick, RoundCount - 2, 0, ultimateWinnerIdx);
    Float secondSubtreeOdds = chanceOfPickOccurringBackwardTree(
      pick, RoundCount - 2, 1, ultimateWinnerIdx);

    return result * firstSubtreeOdds * secondSubtreeOdds;
  }

  std::pair<unsigned int, unsigned int> getPossibleWinningTeams(
    unsigned int round, unsigned int game)
  {
    unsigned int teamCount = 0x01 << (round+1);
    unsigned int firstTeam = game*teamCount;
    return std::make_pair(firstTeam, firstTeam+teamCount);
  }

  Float chanceOfPickOccurringBackwardTree(Pick pick,
                                          unsigned int round,
                                          unsigned int game,
                                          unsigned int ultimateWinnerIdx)
  {
    if (round == 0)
    {
      auto localWinnerIdx = getWinningTeamIdx(pick, round, game);
      if( localWinnerIdx == ultimateWinnerIdx )
      {
        return 1.0;
      }
      else
      {
        return odds[localWinnerIdx];
      }
    }
    else
    {
      auto localWinnerIdx = getWinningTeamIdx(pick, round, game);

      if (localWinnerIdx == ultimateWinnerIdx)
      {
        Float firstSubtreeOdds = chanceOfPickOccurringBackwardTree(
          pick, round-1, 2*game, ultimateWinnerIdx);
        Float secondSubtreeOdds = chanceOfPickOccurringBackwardTree(
          pick, round-1, 2*game+1, ultimateWinnerIdx);

        return firstSubtreeOdds * secondSubtreeOdds;
      }
      else
      {
        Float result = odds[round * TeamCount + localWinnerIdx];

        // I have two children.  In one child, the ultimate winner is also the
        // winner of the subtree.  In the other, I don't know.
        Float firstSubtreeOdds = chanceOfPickOccurringBackwardTree(
          pick, round-1, 2*game, localWinnerIdx);
        Float secondSubtreeOdds = chanceOfPickOccurringBackwardTree(
          pick, round-1, 2*game+1, localWinnerIdx);

        return result * firstSubtreeOdds * secondSubtreeOdds;
      }
    }
  }
  unsigned int gamesInRound(unsigned int round)
  {
    return TeamCount / (0x01 << (round + 1));
  }

  void setOdds(unsigned int teamIdx, unsigned int round, float pct)
  {
    odds[round * TeamCount + teamIdx] = pct;
  }

  unsigned int getGameMask(unsigned int round, unsigned int game)
  {
    if (round > RoundCount)
    {
      throw std::runtime_error("Invalid round.");
    }

    unsigned int firstRoundGames = TeamCount / 2;
    unsigned int roundShift = 0x01;
    unsigned int divisor = 1;
    while (round != 0)
    {
      roundShift = roundShift << (firstRoundGames / divisor);
      divisor *= 2;
      --round;
    }
    roundShift = roundShift << game;
    return roundShift;
  }

  std::pair<unsigned int, unsigned int> getTeamIndices(const Pick& pick,
                                                       unsigned int round,
                                                       unsigned int game)
  {
    std::pair<unsigned int, unsigned int> result{0, 0};

    if (round == 0)
    {
      result.first = game * 2;
      result.second = game * 2 + 1;
    }
    else
    {
      // round 0          round 1        round 2
      //  game 0            game 0        game 0
      //  game 1
      //
      //  game 2            game 1
      //  game 3

      result.first = getWinningTeamIdx(pick, round - 1, 2 * game);
      result.second = getWinningTeamIdx(pick, round - 1, 2 * game + 1);
    }
    return result;
  }

  // Returns an integer [0,TeamCount) representing which team was picked to
  // win
  // the given round and game.  This is necessary because picks simply
  // indicate
  // if the first team in the game wins or the second, but we don't know which
  // teams are in the game until we back up to the first rounds and see which
  // teams were picked the rounds leading up to this one.
  unsigned int getWinningTeamIdx(const Pick& pick,
                                 unsigned int round,
                                 unsigned int game)
  {
    auto indices = getTeamIndices(pick, round, game);

    unsigned int mask = getGameMask(round, game);
    auto masked = mask & pick.value;
    if (masked == 0)
    {
      return indices.first;
    }
    else
    {
      return indices.second;
    }
  }
  // each round has TeamCount odds.
  float odds[TeamCount * RoundCount];
  float seeds[TeamCount];
  std::string names[TeamCount];
};

/** Populates odds from 538 data.  It will do half the bracket. */
BracketOdds<32> createFrom538File(const std::string& filePath,
                                  const std::string& region)
{
  BracketOdds<32> result;
  // The file is a tab-delimited file, with first row header.
  // I will want to slice and dice the bracket construction, so I would like
  // a boolean callback.  Although I really only need single region or left
  // right.
  // Midwest/West are left, East/South are right.
  std::ifstream inFile(filePath.c_str());
  std::string headerLine;
  std::getline(inFile, headerLine);

  // Create a header/column number map.
  auto headerMap = createHeaderMap(headerLine);

  for (std::string line; std::getline(inFile, line);)
  {
    if (line.empty()) continue;
    if (!line.empty() && line[0] == '#') continue;
    // Create column number/value map.
    auto values = createValueMap(line);

    // Populate the data.
    auto rowRegion = values[headerMap["team_region"]];
    auto name = values[headerMap["team_name"]];

    if (!getEntryIsValid(rowRegion, region)) continue;

    auto seed =
      boost::lexical_cast<unsigned int>(values[headerMap["team_seed"]]);
    auto teamNumber = getTeamNumber(seed, rowRegion, region);
    result.seeds[teamNumber] = seed;
    result.names[teamNumber] = name;

    double roundOdds[] = {
      boost::lexical_cast<double>(values[headerMap["rd2_win"]]),
      boost::lexical_cast<double>(values[headerMap["rd3_win"]]),
      boost::lexical_cast<double>(values[headerMap["rd4_win"]]),
      boost::lexical_cast<double>(values[headerMap["rd5_win"]]),
      boost::lexical_cast<double>(values[headerMap["rd6_win"]]),
      boost::lexical_cast<double>(values[headerMap["rd7_win"]])};

    unsigned int numRounds = getNumberOfRounds(region);
    for (unsigned int round = 0; round < numRounds; ++round)
    {
      auto game = getGameNumber(seed, round + 1, rowRegion, region);
      result.odds[round * 32 + teamNumber] = roundOdds[round];
    }
  }
  inFile.close();

  return result;
}

enum class Region
{
  West,
  East,
  Midwest,
  South
};

std::string toString(Region r)
{
  switch (r)
  {
  case Region::West:
    return WEST;
  case Region::East:
    return EAST;
  case Region::Midwest:
    return MIDWEST;
  case Region::South:
    return SOUTH;
  }
}

Region toRegion(const std::string& regionStr)
{
  if (regionStr == WEST) return Region::West;
  if (regionStr == EAST) return Region::East;
  if (regionStr == MIDWEST) return Region::Midwest;
  if (regionStr == SOUTH) return Region::South;
  throw std::runtime_error("Invalid region string.");
}

BracketOdds<16> createQuarterBracketFrom538File(const std::string& filePath,
                                                Region region)
{
  BracketOdds<16> result;
  // The file is a tab-delimited file, with first row header.
  // I will want to slice and dice the bracket construction, so I would like
  // a boolean callback.  Although I really only need single region or left
  // right.
  // Midwest/West are left, East/South are right.
  std::ifstream inFile(filePath.c_str());
  std::string headerLine;
  std::getline(inFile, headerLine);

  // Create a header/column number map.
  auto headerMap = createHeaderMap(headerLine);

  for (std::string line; std::getline(inFile, line);)
  {
    if (line.empty()) continue;
    if (!line.empty() && line[0] == '#') continue;

    // Create column number/value map.
    auto values = createValueMap(line);

    // Populate the data.
    auto rowRegion = values[headerMap["team_region"]];
    auto name = values[headerMap["team_name"]];

    if (toString(region) != rowRegion) continue;

    auto seed =
      boost::lexical_cast<unsigned int>(values[headerMap["team_seed"]]);
    auto teamNumber = getTeamNumber(seed);
    result.seeds[teamNumber] = seed;
    result.names[teamNumber] = name;

    double roundOdds[] = {
      boost::lexical_cast<double>(values[headerMap["rd2_win"]]),
      boost::lexical_cast<double>(values[headerMap["rd3_win"]]),
      boost::lexical_cast<double>(values[headerMap["rd4_win"]]),
      boost::lexical_cast<double>(values[headerMap["rd5_win"]]),
      boost::lexical_cast<double>(values[headerMap["rd6_win"]]),
      boost::lexical_cast<double>(values[headerMap["rd7_win"]])};

    for (unsigned int round = 0; round < 4; ++round)
    {
      auto game = getGameNumber(seed, round + 1);
      result.odds[round * 16 + teamNumber] = roundOdds[round];
    }
  }
  inFile.close();

  return result;
}

#endif
