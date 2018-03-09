#ifndef BRACKET_ODDS_HPP
#define BRACKET_ODDS_HPP

#include <set>
#include <map>
#include "GameIndex.hpp"
#include <fstream>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <algorithm>

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

    //    if (region == LEFT)
    //    {
    //      return rowRegion == SOUTH || rowRegion == WEST;
    //    }
    //    else if (region == RIGHT)
    //    {
    //      return rowRegion == EAST || rowRegion == MIDWEST;
    //    }
    if (region == LEFT)
    {
      return rowRegion == EAST || rowRegion == WEST;
    }
    else if (region == RIGHT)
    {
      return rowRegion == SOUTH || rowRegion == MIDWEST;
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
// This class tracks odds by round/game, but doesn't really care about the
// connections.  Connections will all be implicit, all that really matters is
// if the guess is right or not.
class Five38BracketOdds
{
public:
  Five38BracketOdds() {}

  Five38BracketOdds(const std::string& filePath, const std::string& region)
    : m_odds()
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
        set(Team(name, seed), round + 1, game, roundOdds[round]);

        // For the GPU approach.
        m_roundOdds[round][teamNumber] = roundOdds[round];
      }
    }
    inFile.close();
  }

  void set(const Team& t, unsigned int round, unsigned int game, double odds)
  {
    GameIndex idx(round, game);
    m_odds[idx][t] = odds;
  }

  // Iterator through all combinations of picks represented by this bracket.
  class PickIterator
  {
  private:
    struct Data
    {
      std::map<Team, double>::const_iterator cur;
      std::map<Team, double>::const_iterator end;
      std::map<Team, double>::const_iterator begin;
      bool done() const { return cur == end; }
      void reset() { cur = begin; }
      void advance() { ++cur; }
      void finish() { cur = end; }
      bool operator==(const Data& rhs) const
      {
        return cur == rhs.cur && end == rhs.end && begin == rhs.begin;
      }
    };

  public:
    PickIterator(const std::map<GameIndex, std::map<Team, double>>& d,
                 bool isBegin)
    {
      for (const auto& iter : d)
      {
        Data data;
        data.cur = iter.second.begin();
        data.end = iter.second.end();
        data.begin = iter.second.begin();
        m_data[iter.first] = data;
      }
      if (!isBegin)
      {
        m_data.rbegin()->second.finish();
      }
    }

    bool operator==(const PickIterator& rhs) const
    {
      bool result = true;
      if (m_data.size() != rhs.m_data.size()) return false;

      // end detection
      if (m_data.rbegin()->second.done() && rhs.m_data.rbegin()->second.done())
        return true;

      for (const auto& iter : m_data)
      {
        auto found = rhs.m_data.find(iter.first);
        if (found != rhs.m_data.end())
        {
          result &= iter.second == found->second;
        }
        else
        {
          throw std::runtime_error("can't find key");
        }
        if (!result) return result;
      }
      return result;
    }

    bool operator!=(const PickIterator& rhs) const { return !(*this == rhs); }

    PickIterator& operator++()
    {
      auto last = m_data.begin();
      for (auto iter = m_data.begin(); iter != m_data.end(); ++iter)
      {
        last = iter;
      }

      // Starting at the first game, we increment the local iterator.
      // If we overflow, we reset the current and move onto the next one.
      // If there is no next iterator, then we do not reset the current
      for (auto iter = m_data.begin(); iter != m_data.end(); ++iter)
      {
        auto& data = iter->second;
        data.advance();
        if (!data.done())
        {
          break;
        }

        // We need to reset and move on if we are not the last.
        if (iter != last)
        {
          data.reset();
        }
      }
      return *this;
    }

  private:
    bool isValid() const
    {
      // Picks are invalid if a pick in a later round is made without picking
      // it to win in earlier rounds.
      for (auto& iter : m_data)
      {
        if (iter.first.round > 1)
        {
          auto idx1 = GameIndex(iter.first.round - 1, iter.first.game * 2);
          auto idx2 = GameIndex(iter.first.round - 1, iter.first.game * 2 + 1);

          auto found1 = m_data.find(idx1);
          auto found2 = m_data.find(idx2);

          if (found1->second.cur->first != iter.second.cur->first &&
              found2->second.cur->first != iter.second.cur->first)
          {
            return false;
          }
        }
      }
      return true;
    }

    std::map<GameIndex, Data> m_data;
  };

  PickIterator begin() const { return PickIterator(m_odds, true); }
  PickIterator end() const { return PickIterator(m_odds, false); }

  // For each games, stores the odds of each team (by name) of winning that
  // game.
  std::map<GameIndex, std::map<Team, double>> m_odds{};

  std::array<std::array<float, 32>, 6> m_roundOdds{};
  std::array<float, 32> m_seeds{};
  std::array<std::string, 32> m_names{};
};

inline std::ostream& operator<<(std::ostream& os, const Five38BracketOdds& odds)
{
  for (const auto idxIter : odds.m_odds)
  {
    os << idxIter.first << std::endl;
    for (const auto& gameIter : idxIter.second)
    {
      os << "  " << gameIter.first.name << ", " << gameIter.second << std::endl;
    }
  }
  return os;
}

/** Structure to track a bracket and associated odds. */
template<unsigned int TeamCount>
struct BracketOdds
{
  static constexpr unsigned int RoundCount = std::log2(TeamCount);

  void setOdds(unsigned int teamIdx, unsigned int round, float pct)
  {
    odds[round*TeamCount + teamIdx] = pct;
  }
  // each round has TeamCount odds.
  float odds[TeamCount*RoundCount];
  float seeds[TeamCount];
};

#endif
