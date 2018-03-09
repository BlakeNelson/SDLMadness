
#define BOOST_TEST_MODULE SDLMadnessTest
#include <boost/test/unit_test.hpp>

#include "../BracketOdds.hpp"
#include "../CudaApproach.cuh"
#include "../Util.hpp"
template <unsigned int round, unsigned int game>
struct ExpectedTreeResult
{
  static float Evaluate(unsigned int pick,
                        const float* __restrict__ pOdds,
                        const float* __restrict__ pSeeds)
  {
    auto a = ExpectedTreeResult<round - 1, 2 * game>::Evaluate(
             pick, pOdds, pSeeds);
    auto b =
           ExpectedTreeResult<round - 1, 2 * game + 1>::Evaluate(
             pick, pOdds, pSeeds);
    auto c =
           ExpectedResult<round, game>::Evaluate(pick, pOdds, pSeeds);
    return a*b*c;
  }
};

template <unsigned int game>
struct ExpectedTreeResult<1, game>
{
  static float Evaluate(unsigned int pick,
                        const float* __restrict__ pOdds,
                        const float* __restrict__ pSeeds)
  {
    return ExpectedResult<1, game>::Evaluate(pick, pOdds, pSeeds);
  }
};

template <unsigned int round, unsigned int game>
struct EvaluateGames
{
  static float Evaluate(unsigned int pick,
                        const float* __restrict__ pOdds,
                        const float* __restrict__ pSeeds)
  {
    return ExpectedTreeResult<round, game>::Evaluate(pick, pOdds, pSeeds) +
           EvaluateGames<round, game - 1>::Evaluate(pick, pOdds, pSeeds);
  }
};

template <unsigned int round>
struct EvaluateGames<round, 0>
{
  static float Evaluate(unsigned int pick,
                        const float* __restrict__ pOdds,
                        const float* __restrict__ pSeeds)
  {
    return ExpectedTreeResult<round, 0>::Evaluate(pick, pOdds, pSeeds);
  }
};

BOOST_AUTO_TEST_CASE(RunFourTeamTest)
{
  Five38BracketOdds bracket("four_team_test.csv", "Left");
  runCudaApproach(bracket);
}

template <unsigned int round, unsigned int game>
struct TreeOdds
{
  static float Evaluate(unsigned int pick, const float* __restrict__ pOdds)
  {
    return TreeOdds<round - 1, 2 * game>::Evaluate(pick, pOdds) +
           TreeOdds<round - 1, 2 * game + 1>::Evaluate(pick, pOdds) +
           Odds<round, game>::Evaluate(pick, pOdds);
  }
};

template <unsigned int game>
struct TreeOdds<1, game>
{
  static float Evaluate(unsigned int pick, const float* __restrict__ pOdds)
  {
    return Odds<1, game>::Evaluate(pick, pOdds);
  }
};

BOOST_AUTO_TEST_CASE(TestSingleOdds)
{
  float odds[] = {.2, .3, .4};

  auto r = Odds<1, 0>::Evaluate(0x00, odds);
  BOOST_CHECK_CLOSE(r, .2, .01);
  r = Odds<1, 0>::Evaluate(0x01, odds);
  BOOST_CHECK_CLOSE(r, .3, .01);

  float pOdds[] = {.1,  .9,  .3,  .7,  0.0, 0,   0,   0.0, 0.0, 0.0, 0,   0,
                   0.0, 0.0, 0.0, 0,   0,   0.0, 0.0, 0.0, 0,   0,   0.0, 0.0,
                   0.0, 0,   0,   0.0, 0.0, 0.0, 0,   0,
                   .1,  .4,  .3,  .2,  0.0, 0,   0,   0.0, 0.0,
                   0.0, 0,   0,   0.0, 0.0, 0.0, 0,   0,   0.0, 0.0, 0.0, 0,
                   0,   0.0, 0.0, 0.0, 0,   0,   0.0, 0.0, 0.0, 0,   0,   0.0,
                   0.0, 0.0, 0,   0,   0.0, 0.0};
  float seeds[]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  std::cout << ExpectedTreeResult<2, 0>::Evaluate(0x10003, &pOdds[0], &seeds[0])
            << std::endl;

  std::cout << ExpectedTreeResult<2, 0>::Evaluate(0x00000, &pOdds[0], &seeds[0]) +
      ExpectedTreeResult<2, 0>::Evaluate(0x00001, &pOdds[0], &seeds[0]) +
      ExpectedTreeResult<2, 0>::Evaluate(0x00002, &pOdds[0], &seeds[0]) +
      ExpectedTreeResult<2, 0>::Evaluate(0x00003, &pOdds[0], &seeds[0]) +
      ExpectedTreeResult<2, 0>::Evaluate(0x10000, &pOdds[0], &seeds[0]) +
            ExpectedTreeResult<2, 0>::Evaluate(0x10001, &pOdds[0], &seeds[0]) +
            ExpectedTreeResult<2, 0>::Evaluate(0x10002, &pOdds[0], &seeds[0]) +
            ExpectedTreeResult<2, 0>::Evaluate(0x10003, &pOdds[0], &seeds[0]);
}


BOOST_AUTO_TEST_CASE(TestBracket)
{
  std::cout << BracketOdds<4>::RoundCount << std::endl;
  std::cout << BracketOdds<8>::RoundCount << std::endl;
  std::cout << BracketOdds<16>::RoundCount << std::endl;
  std::cout << BracketOdds<32>::RoundCount << std::endl;
}
