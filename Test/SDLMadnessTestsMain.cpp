
#define BOOST_TEST_MODULE SDLMadnessTest
#include <boost/test/unit_test.hpp>

#include "../BracketOdds.hpp"
#include "../CudaApproach.cuh"

BOOST_AUTO_TEST_CASE(RunFourTeamTest)
{
  Five38BracketOdds bracket("four_team_test.csv", "Left");
  runCudaApproach(bracket);
}
