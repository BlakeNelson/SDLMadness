#ifndef GAME_INDEX_HPP
#define GAME_INDEX_HPP

#include <iostream>

struct GameIndex
{
  GameIndex(unsigned int r, unsigned int g) :
    round(r), game(g) {}

  unsigned int round;
  unsigned int game;
};

inline bool operator<(const GameIndex& lhs, const GameIndex& rhs)
{
  if( lhs.round < rhs.round ) return true;
  if( rhs.round < lhs.round ) return false;
  return lhs.game < rhs.game;
}

inline std::ostream& operator<<(std::ostream& os, const GameIndex& idx)
{
  os << "[" << idx.round << ", " << idx.game << "]";
  return os;
}

// A team with a specific seed.
struct Team
{
  Team(const std::string& n, unsigned int s) : name(n), seed(s) {}
  std::string name;
  unsigned int seed;
};

inline bool operator<(const Team& lhs, const Team& rhs)
{
  return lhs.name < rhs.name;
}

inline bool operator==(const Team& lhs, const Team& rhs)
{
  return lhs.name == rhs.name;
}

inline bool operator!=(const Team& lhs, const Team& rhs)
{
  return lhs.name != rhs.name;
}

inline std::ostream& operator<<(std::ostream& os, const Team& p)
{
  os << p.name << "(" << p.seed << ")";
  return os;
}
#endif
