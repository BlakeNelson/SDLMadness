#ifndef FIND_BEST_PICK_HPP
#define FIND_BEST_PICK_HPP

#include "BracketOdds.hpp"
#include "Pick.hpp"


Pick score(const Five38BracketOdds& odds,
             const Five38BracketOdds::PickIterator& iter)
{
  Pick result;
  return result;
//  double expectedScore = 0.0;
//  for(const auto& iter : rPick.m_picks)
//  {
//    auto idx = iter.first;
//    auto team = iter.second.cur->first;
//    auto seed = iter.second.cur->first.seed;

//    auto found = odds.m_odds.find(idx);
//    auto percent = found->second.find(team);
//    auto extra = (0x01 << idx.round-1);
//    expectedScore += extra *seed* (percent->second);
//    //expectedScore += idx.round*seed* (percent->second);
//  }
//  rPick.m_score = expectedScore;
}

Pick findBestPick(const Five38BracketOdds& odds)
{
  Pick bestPick;

  unsigned int maxCount = 67108864;
  unsigned int curCount = 0;
  for(auto iter = odds.begin(); iter != odds.end(); ++iter)
  {
    ++curCount;
    if( curCount % 1000 == 0 )
    {
      std::cout << curCount / (float)maxCount << std::endl;
    }
    auto curPick = score(odds, iter);
    if( curPick.m_score > bestPick.m_score)
    {
      bestPick = curPick;
    }
  }

  return bestPick;
}

#endif
