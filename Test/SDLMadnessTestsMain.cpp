
#define BOOST_TEST_MODULE SDLMadnessTest
#include <boost/test/unit_test.hpp>

#include "../BracketOdds.hpp"
#include "../CudaApproach.cuh"
#include "../Util.hpp"
#include <iostream>

BOOST_AUTO_TEST_CASE(Test4TeamGetGameMask)
{
  BracketOdds<4> odds;
  BOOST_CHECK_EQUAL(odds.getGameMask(0, 0), 0x01u);
  BOOST_CHECK_EQUAL(odds.getGameMask(0, 1), 0x02u);

  BOOST_CHECK_EQUAL(odds.getGameMask(1, 0), 0x4u);
}

BOOST_AUTO_TEST_CASE(Test8TeamGetGameMask)
{
  BracketOdds<8> odds;
  BOOST_CHECK_EQUAL(odds.getGameMask(0, 0), 0x01u);
  BOOST_CHECK_EQUAL(odds.getGameMask(0, 1), 0x02u);
  BOOST_CHECK_EQUAL(odds.getGameMask(0, 2), 0x04u);
  BOOST_CHECK_EQUAL(odds.getGameMask(0, 3), 0x08u);

  BOOST_CHECK_EQUAL(odds.getGameMask(1, 0), 0x10u);
  BOOST_CHECK_EQUAL(odds.getGameMask(1, 1), 0x20u);

  BOOST_CHECK_EQUAL(odds.getGameMask(2, 0), 0x40u);
}

BOOST_AUTO_TEST_CASE(Test4TeamGetWinningTeam)
{
  BracketOdds<4> odds;
  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x0u), 0, 0), 0);
  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x0u), 0, 1), 2);
  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x0u), 1, 0), 0);

  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x1u), 0, 0), 1);
  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x1u), 0, 1), 2);
  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x1u), 1, 0), 1);

  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x2u), 0, 0), 0);
  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x2u), 0, 1), 3);
  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x2u), 1, 0), 0);

  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x3u), 0, 0), 1);
  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x3u), 0, 1), 3);
  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x3u), 1, 0), 1);

  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x4u), 0, 0), 0);
  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x4u), 0, 1), 2);
  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x4u), 1, 0), 2);

  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x5u), 0, 0), 1);
  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x5u), 0, 1), 2);
  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x5u), 1, 0), 2);

  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x6u), 0, 0), 0);
  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x6u), 0, 1), 3);
  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x6u), 1, 0), 3);

  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x7u), 0, 0), 1);
  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x7u), 0, 1), 3);
  BOOST_CHECK_EQUAL(odds.getWinningTeamIdx(Pick(0x7u), 1, 0), 3);
}

BOOST_AUTO_TEST_CASE(TestGamesInRound)
{
  BracketOdds<4> fourOdds;
  BOOST_CHECK_EQUAL(fourOdds.gamesInRound(0), 2);
  BOOST_CHECK_EQUAL(fourOdds.gamesInRound(1), 1);

  BracketOdds<8> eightOdds;
  BOOST_CHECK_EQUAL(eightOdds.gamesInRound(0), 4);
  BOOST_CHECK_EQUAL(eightOdds.gamesInRound(1), 2);
  BOOST_CHECK_EQUAL(eightOdds.gamesInRound(2), 1);
}

BOOST_AUTO_TEST_CASE(TestchanceOfPickOccurring)
{
  BracketOdds<4> odds;
  odds.odds[0] = .1;
  odds.odds[1] = .9;
  odds.odds[2] = .2;
  odds.odds[3] = .8;
  odds.odds[4] = .02;
  odds.odds[5] = .4;
  odds.odds[6] = .08;
  odds.odds[7] = .5;

  BOOST_CHECK_CLOSE(odds.chanceOfPickOccurring(Pick(0x0)), .004, 1e-3);
  BOOST_CHECK_CLOSE(odds.chanceOfPickOccurring(Pick(0x5)), .03, 1e-3);

  Float result = 0.0f;
  for (unsigned int i = 0; i < 8; ++i)
  {
    result += odds.chanceOfPickOccurring(Pick(i));
  }
  BOOST_CHECK_CLOSE(result, 1.0, 1e-3);
}

BOOST_AUTO_TEST_CASE(TestChanceOfPickOccurring)
{
  BracketOdds<4> odds;
  odds.odds[0] = .2;
  odds.odds[1] = .8;
  odds.odds[2] = .3;
  odds.odds[3] = .7;
  odds.odds[4] = .1;
  odds.odds[5] = .7;
  odds.odds[6] = .05;
  odds.odds[7] = .15;

  std::cout << "Original" << std::endl;
  Float f = 0.0;
  for(unsigned int i = 0; i < 8; ++i)
  {
    f += odds.chanceOfPickOccurring(i);
    std::cout << odds.chanceOfPickOccurring(i) << std::endl;
  }
  std::cout << "Total: " << f << std::endl;

  f = 0.0;
  for(unsigned int i = 0; i < 8; ++i)
  {
    f += odds.chanceOfPickOccurringBackward(i);
    std::cout << odds.chanceOfPickOccurringBackward(i) << std::endl;
  }
  std::cout << "Total: " << f << std::endl;

}

BOOST_AUTO_TEST_CASE(TestBracketGetOddsStats)
{
  //  BracketOdds<4> odds;
  //  odds.odds[0] = .1;
  //  odds.odds[1] = .9;
  //  odds.odds[2] = .2;
  //  odds.odds[3] = .8;
  //  odds.odds[4] = .02;
  //  odds.odds[5] = .4;
  //  odds.odds[6] = .08;
  //  odds.odds[7] = .5;

  //  auto stats = odds.getOddsStats();
  //  std::cout << stats.avg << std::endl;

  auto quarterBracket = createQuarterBracketFrom538File(
    "fivethirtyeight_ncaa_forecasts_2018.csv", Region::East);
  auto quarterOdds = quarterBracket.getOddsStats();
  std::cout << "West odds" << std::endl;
  std::cout << "Min: " << quarterOdds.min << std::endl;
  std::cout << "Min Pick: " << quarterOdds.minPick << std::endl;
  quarterBracket.printBracket(quarterOdds.minPick);
  std::cout << "Max: " << quarterOdds.max << std::endl;
  std::cout << "Max Pick: " << quarterOdds.maxPick << std::endl;
  quarterBracket.printBracket(quarterOdds.maxPick);
  std::cout << "Avg: " << quarterOdds.avg << std::endl;
  std::cout << "Std: " << quarterOdds.stddev << std::endl;

  //    auto fullOdds =
  //      createFrom538File("fivethirtyeight_ncaa_forecasts_2018.csv", "Left");
  //    std::cout << "Left odds" << std::endl;
  //    auto stats = fullOdds.getOddsStats();
  //    std::cout << "Min: " << stats.min << std::endl;
  //    std::cout << "Min Pick: " << stats.minPick << std::endl;
  //    fullOdds.printBracket(stats.minPick);
  //    std::cout << "Max: " << stats.max << std::endl;
  //    std::cout << "Max Pick: " << stats.maxPick << std::endl;
  //    fullOdds.printBracket(stats.maxPick);
  //    std::cout << "Avg: " << stats.avg << std::endl;
  //    std::cout << "Std: " << stats.stddev << std::endl;
}
// template <unsigned int round, unsigned int game>
// struct ExpectedTreeResult
//{
//  static float Evaluate(unsigned int pick,
//                        const float* __restrict__ pOdds,
//                        const float* __restrict__ pSeeds)
//  {
//    auto a = ExpectedTreeResult<round - 1, 2 * game>::Evaluate(
//             pick, pOdds, pSeeds);
//    auto b =
//           ExpectedTreeResult<round - 1, 2 * game + 1>::Evaluate(
//             pick, pOdds, pSeeds);
//    auto c =
//           ExpectedResult<round, game>::Evaluate(pick, pOdds, pSeeds);
//    return a*b*c;
//  }
//};

// template <unsigned int game>
// struct ExpectedTreeResult<1, game>
//{
//  static float Evaluate(unsigned int pick,
//                        const float* __restrict__ pOdds,
//                        const float* __restrict__ pSeeds)
//  {
//    return ExpectedResult<1, game>::Evaluate(pick, pOdds, pSeeds);
//  }
//};

// template <unsigned int round, unsigned int game>
// struct EvaluateGames
//{
//  static float Evaluate(unsigned int pick,
//                        const float* __restrict__ pOdds,
//                        const float* __restrict__ pSeeds)
//  {
//    return ExpectedTreeResult<round, game>::Evaluate(pick, pOdds, pSeeds) +
//           EvaluateGames<round, game - 1>::Evaluate(pick, pOdds, pSeeds);
//  }
//};

// template <unsigned int round>
// struct EvaluateGames<round, 0>
//{
//  static float Evaluate(unsigned int pick,
//                        const float* __restrict__ pOdds,
//                        const float* __restrict__ pSeeds)
//  {
//    return ExpectedTreeResult<round, 0>::Evaluate(pick, pOdds, pSeeds);
//  }
//};

// BOOST_AUTO_TEST_CASE(RunFourTeamTest)
//{
//  Five38BracketOdds bracket("four_team_test.csv", "Left");
//  runCudaApproach(bracket);
//}

// template <unsigned int round, unsigned int game>
// struct TreeOdds
//{
//  static float Evaluate(unsigned int pick, const float* __restrict__ pOdds)
//  {
//    return TreeOdds<round - 1, 2 * game>::Evaluate(pick, pOdds) +
//           TreeOdds<round - 1, 2 * game + 1>::Evaluate(pick, pOdds) +
//           Odds<round, game>::Evaluate(pick, pOdds);
//  }
//};

// template <unsigned int game>
// struct TreeOdds<1, game>
//{
//  static float Evaluate(unsigned int pick, const float* __restrict__ pOdds)
//  {
//    return Odds<1, game>::Evaluate(pick, pOdds);
//  }
//};

// BOOST_AUTO_TEST_CASE(TestSingleOdds)
//{
//  float odds[] = {.2, .3, .4};

//  auto r = Odds<1, 0>::Evaluate(0x00, odds);
//  BOOST_CHECK_CLOSE(r, .2, .01);
//  r = Odds<1, 0>::Evaluate(0x01, odds);
//  BOOST_CHECK_CLOSE(r, .3, .01);

//  float pOdds[] = {.1,  .9,  .3,  .7,  0.0, 0,   0,   0.0, 0.0, 0.0, 0,   0,
//                   0.0, 0.0, 0.0, 0,   0,   0.0, 0.0, 0.0, 0,   0,   0.0, 0.0,
//                   0.0, 0,   0,   0.0, 0.0, 0.0, 0,   0,
//                   .1,  .4,  .3,  .2,  0.0, 0,   0,   0.0, 0.0,
//                   0.0, 0,   0,   0.0, 0.0, 0.0, 0,   0,   0.0, 0.0, 0.0, 0,
//                   0,   0.0, 0.0, 0.0, 0,   0,   0.0, 0.0, 0.0, 0,   0,   0.0,
//                   0.0, 0.0, 0,   0,   0.0, 0.0};
//  float seeds[]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

//  std::cout << ExpectedTreeResult<2, 0>::Evaluate(0x10003, &pOdds[0],
//  &seeds[0])
//            << std::endl;

//  std::cout << ExpectedTreeResult<2, 0>::Evaluate(0x00000, &pOdds[0],
//  &seeds[0]) +
//      ExpectedTreeResult<2, 0>::Evaluate(0x00001, &pOdds[0], &seeds[0]) +
//      ExpectedTreeResult<2, 0>::Evaluate(0x00002, &pOdds[0], &seeds[0]) +
//      ExpectedTreeResult<2, 0>::Evaluate(0x00003, &pOdds[0], &seeds[0]) +
//      ExpectedTreeResult<2, 0>::Evaluate(0x10000, &pOdds[0], &seeds[0]) +
//            ExpectedTreeResult<2, 0>::Evaluate(0x10001, &pOdds[0], &seeds[0])
//            +
//            ExpectedTreeResult<2, 0>::Evaluate(0x10002, &pOdds[0], &seeds[0])
//            +
//            ExpectedTreeResult<2, 0>::Evaluate(0x10003, &pOdds[0], &seeds[0]);
//}

// BOOST_AUTO_TEST_CASE(TestBracket)
//{
//  std::cout << BracketOdds<4>::RoundCount << std::endl;
//  std::cout << BracketOdds<8>::RoundCount << std::endl;
//  std::cout << BracketOdds<16>::RoundCount << std::endl;
//  std::cout << BracketOdds<32>::RoundCount << std::endl;
//}
