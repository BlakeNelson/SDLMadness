#ifndef UTIL_HPP
#define UTIL_HPP

#include <cuda_runtime.h>
#include <stdio.h>

// Half of the tournament reprsents 31 individual games.  All possible picks
// of these games can be represented in 31 bits of a 32 bit integer (note that
// we don't even try to evaluate the entire tournament, which is 63 separate
// games and would involve 2^63 possible choices).

// Returns a mask for a given round and game to determine who is picked to
// win.  Rounds are 1 based, games are 0 based.
template <unsigned int round, unsigned int game>
struct GetMask;

template <>
struct GetMask<1, 0>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00000001; }
};

template <>
struct GetMask<1, 1>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00000002; }
};

template <>
struct GetMask<1, 2>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00000004; }
};

template <>
struct GetMask<1, 3>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00000008; }
};

template <>
struct GetMask<1, 4>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00000010; }
};

template <>
struct GetMask<1, 5>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00000020; }
};

template <>
struct GetMask<1, 6>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00000040; }
};

template <>
struct GetMask<1, 7>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00000080; }
};

template <>
struct GetMask<1, 8>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00000100; }
};

template <>
struct GetMask<1, 9>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00000200; }
};

template <>
struct GetMask<1, 10>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00000400; }
};

template <>
struct GetMask<1, 11>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00000800; }
};

template <>
struct GetMask<1, 12>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00001000; }
};

template <>
struct GetMask<1, 13>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00002000; }
};

template <>
struct GetMask<1, 14>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00004000; }
};

template <>
struct GetMask<1, 15>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00008000; }
};

template <>
struct GetMask<2, 0>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00010000; }
};

template <>
struct GetMask<2, 1>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00020000; }
};

template <>
struct GetMask<2, 2>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00040000; }
};

template <>
struct GetMask<2, 3>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00080000; }
};

template <>
struct GetMask<2, 4>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00100000; }
};

template <>
struct GetMask<2, 5>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00200000; }
};

template <>
struct GetMask<2, 6>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00400000; }
};

template <>
struct GetMask<2, 7>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x00800000; }
};

template <>
struct GetMask<3, 0>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x01000000; }
};

template <>
struct GetMask<3, 1>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x02000000; }
};

template <>
struct GetMask<3, 2>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x04000000; }
};

template <>
struct GetMask<3, 3>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x08000000; }
};

template <>
struct GetMask<4, 0>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x10000000; }
};

template <>
struct GetMask<4, 1>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x20000000; }
};

template <>
struct GetMask<5, 0>
{
  __host__ __device__ static unsigned int Evaluate() { return 0x40000000; }
};

// Returns the index of the chosen winner.
// 0 means the top team wins, 1 means the bottom team wins.

// Returns the team id for the team that won the given game.
template <unsigned int round, unsigned int game>
struct GetWinningTeamId;

template <unsigned int game>
struct GetWinningTeamId<1, game>
{
  __host__ __device__ static unsigned int Evaluate(unsigned int pick)
  {
    auto mask = GetMask<1, game>::Evaluate();
    auto masked = pick & mask;
    unsigned int result = game * 2;
    if (masked > 0) result += 1;
    return result;
  }
};

template <unsigned int game>
struct GetWinningTeamId<2, game>
{
  __host__ __device__ static unsigned int Evaluate(unsigned int pick)
  {
    auto topIdx = GetWinningTeamId<1, 2 * game>::Evaluate(pick);
    auto bottomIdx = GetWinningTeamId<1, 2 * game + 1>::Evaluate(pick);

    auto mask = GetMask<2, game>::Evaluate();
    auto masked = pick & mask;
    unsigned int result = topIdx;
    if (masked > 0) result = bottomIdx;
    return result;
  }
};

template <unsigned int game>
struct GetWinningTeamId<3, game>
{
  __host__ __device__ static unsigned int Evaluate(unsigned int pick)
  {
    auto topIdx = GetWinningTeamId<2, 2 * game>::Evaluate(pick);
    auto bottomIdx = GetWinningTeamId<2, 2 * game + 1>::Evaluate(pick);

    auto mask = GetMask<3, game>::Evaluate();
    auto masked = pick & mask;
    unsigned int result = topIdx;
    if (masked > 0) result = bottomIdx;
    return result;
  }
};

template <unsigned int game>
struct GetWinningTeamId<4, game>
{
  __host__ __device__ static unsigned int Evaluate(unsigned int pick)
  {
    auto topIdx = GetWinningTeamId<3, 2 * game>::Evaluate(pick);
    auto bottomIdx = GetWinningTeamId<3, 2 * game + 1>::Evaluate(pick);

    auto mask = GetMask<4, game>::Evaluate();
    auto masked = pick & mask;
    unsigned int result = topIdx;
    if (masked > 0) result = bottomIdx;
    return result;
  }
};

template <unsigned int game>
struct GetWinningTeamId<5, game>
{
  __host__ __device__ static unsigned int Evaluate(unsigned int pick)
  {
    auto topIdx = GetWinningTeamId<4, 2 * game>::Evaluate(pick);
    auto bottomIdx = GetWinningTeamId<4, 2 * game + 1>::Evaluate(pick);

    auto mask = GetMask<5, game>::Evaluate();
    auto masked = pick & mask;
    unsigned int result = topIdx;
    if (masked > 0) result = bottomIdx;
    return result;
  }
};

// The CBS site awards point as scaleFactor*picksCorrect per round.
// The RoundMultipliers captures this value.
template <unsigned int round>
struct RoundMultiplier;

template <>
struct RoundMultiplier<1>
{
  static __host__ __device__ float Evaluate() { return 1.0f; }
};

template <>
struct RoundMultiplier<2>
{
  static __host__ __device__ float Evaluate() { return 2.0f; }
};

template <>
struct RoundMultiplier<3>
{
  static __host__ __device__ float Evaluate() { return 4.0f; }
};

template <>
struct RoundMultiplier<4>
{
  static __host__ __device__ float Evaluate() { return 8.0f; }
};

template <>
struct RoundMultiplier<5>
{
  static __host__ __device__ float Evaluate() { return 16.0f; }
};

template <unsigned int round>
struct ESPNRoundMultiplier;

template <>
struct ESPNRoundMultiplier<1>
{
  static __host__ __device__ float Evaluate() { return 10.0f; }
};

template <>
struct ESPNRoundMultiplier<2>
{
  static __host__ __device__ float Evaluate() { return 20.0f; }
};

template <>
struct ESPNRoundMultiplier<3>
{
  static __host__ __device__ float Evaluate() { return 40.0f; }
};

template <>
struct ESPNRoundMultiplier<4>
{
  static __host__ __device__ float Evaluate() { return 80.0f; }
};

template <>
struct ESPNRoundMultiplier<5>
{
  static __host__ __device__ float Evaluate() { return 160.0f; }
};


template <unsigned int round, unsigned int game>
struct ExpectedResult
{
  static __host__ __device__ float Evaluate(unsigned int pick,
                                            const float* __restrict__ pOdds,
                                            const float* __restrict__ pSeeds)
  {
    if (pick == 0)
    {
      auto idx = GetWinningTeamId<round, game>::Evaluate(pick);
      auto odds = pOdds[idx + 32 * (round - 1)];
      auto seed = pSeeds[idx];
      auto mult = RoundMultiplier<round>::Evaluate();
      printf("Round %d Game %d Idx %d odds %f Seed %f multiplier %f\n",
             round,
             game,
             idx,
             odds,
             seed,
             mult);
    }
    auto idx = GetWinningTeamId<round, game>::Evaluate(pick);
    // CBS
    //    return pOdds[idx + 32 * (round - 1)] * pSeeds[idx] *
    //           RoundMultiplier<round>::Evaluate();

    // ESPN
//    return pOdds[idx + 32 * (round - 1)] *
//           ESPNRoundMultiplier<round>::Evaluate();

    // Just odds
    return pOdds[idx + 32 * (round - 1)];
  }
};

template <unsigned int round, unsigned int game>
struct Odds
{
  static __host__ __device__ float Evaluate(unsigned int pick,
                                            const float* __restrict__ pOdds)
  {
    auto idx = GetWinningTeamId<round, game>::Evaluate(pick);
    return pOdds[idx + 32 * (round - 1)];
  }
};

#endif
