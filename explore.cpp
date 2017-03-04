#include "explore.hpp"
#include "BracketOdds.hpp"
#include <iostream>
#include "FindBestPick.hpp"


void run2015Midwest()
{
  Five38BracketOdds bracket("/data/TestDev/SDLMadness2016/src/bracket-06.tsv", "Midwest");
  auto bestPick = findBestPick(bracket);

  std::cout << "Bracket" << std::endl;
  std::cout << "-------" << std::endl;
  std::cout << bracket << std::endl;
  std::cout << "Best Pick" << std::endl;
  std::cout << "---------" << std::endl;
  std::cout << bestPick << std::endl;
}

void run2015West()
{
  Five38BracketOdds bracket("/data/TestDev/SDLMadness2016/src/bracket-06.tsv", "Left");
  auto bestPick = findBestPick(bracket);

  std::cout << "Bracket" << std::endl;
  std::cout << "-------" << std::endl;
  std::cout << bracket << std::endl;
  std::cout << "Best Pick" << std::endl;
  std::cout << "---------" << std::endl;
  std::cout << bestPick << std::endl;
}
