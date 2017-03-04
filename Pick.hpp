#ifndef PICK_HPP
#define PICK_HPP

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include "GameIndex.hpp"

class Pick
{
public:
  Pick() : m_picks(), m_score(0.0)
  {
  }

  // The input is all of the games that will be played, along with the
  // odds for each game.
  // For each game, Data stores the actual pick made for the game.
  // This member is actual state.
  std::map<GameIndex, Team> m_picks;
  double m_score;
};

inline std::ostream& operator<<(std::ostream& os, const Pick& p)
{
  for (const auto& iter : p.m_picks)
  {
    os << iter.first << " - " << iter.second.name << std::endl;
  }
  os << "Score: " << p.m_score;
  return os;
}

#endif
