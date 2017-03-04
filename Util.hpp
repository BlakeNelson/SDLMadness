#ifndef UTIL_HPP
#define UTIL_HPP

#include <cuda_runtime.h>
#include <stdio.h>

// Get the mask for teh given round/game.  Round is 1 based, game is 0 based.
// Games are from the top of the bracket to the bottom on one side only,
// so 16 first round games, 8 second, and so on.

#ifdef __CUDACC__
#define DEVICE __host__ __device__
#else
#define DEVICE
#endif

template <unsigned int round, unsigned int game>
struct GetMask;

template <>
struct GetMask<1, 0>
{
  DEVICE static unsigned int Evaluate() { return 0x00000001; }
};

template <>
struct GetMask<1, 1>
{
  DEVICE static unsigned int Evaluate() { return 0x00000002; }
};

template <>
struct GetMask<1, 2>
{
  DEVICE static unsigned int Evaluate() { return 0x00000004; }
};

template <>
struct GetMask<1, 3>
{
  DEVICE static unsigned int Evaluate() { return 0x00000008; }
};

template <>
struct GetMask<1, 4>
{
  DEVICE static unsigned int Evaluate() { return 0x00000010; }
};

template <>
struct GetMask<1, 5>
{
  DEVICE static unsigned int Evaluate() { return 0x00000020; }
};

template <>
struct GetMask<1, 6>
{
  DEVICE static unsigned int Evaluate() { return 0x00000040; }
};

template <>
struct GetMask<1, 7>
{
  DEVICE static unsigned int Evaluate() { return 0x00000080; }
};

template <>
struct GetMask<1, 8>
{
  DEVICE static unsigned int Evaluate() { return 0x00000100; }
};

template <>
struct GetMask<1, 9>
{
  DEVICE static unsigned int Evaluate() { return 0x00000200; }
};

template <>
struct GetMask<1, 10>
{
  DEVICE static unsigned int Evaluate() { return 0x00000400; }
};

template <>
struct GetMask<1, 11>
{
  DEVICE static unsigned int Evaluate() { return 0x00000800; }
};

template <>
struct GetMask<1, 12>
{
  DEVICE static unsigned int Evaluate() { return 0x00001000; }
};

template <>
struct GetMask<1, 13>
{
  DEVICE static unsigned int Evaluate() { return 0x00002000; }
};

template <>
struct GetMask<1, 14>
{
  DEVICE static unsigned int Evaluate() { return 0x00004000; }
};

template <>
struct GetMask<1, 15>
{
  DEVICE static unsigned int Evaluate() { return 0x00008000; }
};

template <>
struct GetMask<2, 0>
{
  DEVICE static unsigned int Evaluate() { return 0x00010000; }
};

template <>
struct GetMask<2, 1>
{
  DEVICE static unsigned int Evaluate() { return 0x00020000; }
};

template <>
struct GetMask<2, 2>
{
  DEVICE static unsigned int Evaluate() { return 0x00040000; }
};

template <>
struct GetMask<2, 3>
{
  DEVICE static unsigned int Evaluate() { return 0x00080000; }
};

template <>
struct GetMask<2, 4>
{
  DEVICE static unsigned int Evaluate() { return 0x00100000; }
};

template <>
struct GetMask<2, 5>
{
  DEVICE static unsigned int Evaluate() { return 0x00200000; }
};

template <>
struct GetMask<2, 6>
{
  DEVICE static unsigned int Evaluate() { return 0x00400000; }
};

template <>
struct GetMask<2, 7>
{
  DEVICE static unsigned int Evaluate() { return 0x00800000; }
};

template <>
struct GetMask<3, 0>
{
  DEVICE static unsigned int Evaluate() { return 0x01000000; }
};

template <>
struct GetMask<3, 1>
{
  DEVICE static unsigned int Evaluate() { return 0x02000000; }
};

template <>
struct GetMask<3, 2>
{
  DEVICE static unsigned int Evaluate() { return 0x04000000; }
};

template <>
struct GetMask<3, 3>
{
  DEVICE static unsigned int Evaluate() { return 0x08000000; }
};

template <>
struct GetMask<4, 0>
{
  DEVICE static unsigned int Evaluate() { return 0x10000000; }
};

template <>
struct GetMask<4, 1>
{
  DEVICE static unsigned int Evaluate() { return 0x20000000; }
};

template <>
struct GetMask<5, 0>
{
  DEVICE static unsigned int Evaluate() { return 0x40000000; }
};

// Returns the index of the chosen winner.
// 0 means the top team wins, 1 means the bottom team wins.
template <unsigned int round, unsigned int game>
struct GetIndex;

template <unsigned int game>
struct GetIndex<1, game>
{
  DEVICE static unsigned int Evaluate(unsigned int pick)
  {
    auto mask = GetMask<1, game>::Evaluate();
    auto masked = pick & mask;
    unsigned int result = game * 2;
    if (masked > 0) result += 1;
    return result;
  }
};

template <unsigned int game>
struct GetIndex<2, game>
{
  DEVICE static unsigned int Evaluate(unsigned int pick)
  {
    auto topIdx = GetIndex<1, 2 * game>::Evaluate(pick);
    auto bottomIdx = GetIndex<1, 2 * game + 1>::Evaluate(pick);

    auto mask = GetMask<2, game>::Evaluate();
    auto masked = pick & mask;
    unsigned int result = topIdx;
    if (masked > 0) result = bottomIdx;
    return result;
  }
};

template <unsigned int game>
struct GetIndex<3, game>
{
  DEVICE static unsigned int Evaluate(unsigned int pick)
  {
    auto topIdx = GetIndex<2, 2 * game>::Evaluate(pick);
    auto bottomIdx = GetIndex<2, 2 * game + 1>::Evaluate(pick);

    auto mask = GetMask<3, game>::Evaluate();
    auto masked = pick & mask;
    unsigned int result = topIdx;
    if (masked > 0) result = bottomIdx;
    return result;
  }
};

template <unsigned int game>
struct GetIndex<4, game>
{
  DEVICE static unsigned int Evaluate(unsigned int pick)
  {
    auto topIdx = GetIndex<3, 2 * game>::Evaluate(pick);
    auto bottomIdx = GetIndex<3, 2 * game + 1>::Evaluate(pick);

    auto mask = GetMask<4, game>::Evaluate();
    auto masked = pick & mask;
    unsigned int result = topIdx;
    if (masked > 0) result = bottomIdx;
    return result;
  }
};

template <unsigned int game>
struct GetIndex<5, game>
{
  DEVICE static unsigned int Evaluate(unsigned int pick)
  {
    auto topIdx = GetIndex<4, 2 * game>::Evaluate(pick);
    auto bottomIdx = GetIndex<4, 2 * game + 1>::Evaluate(pick);

    auto mask = GetMask<5, game>::Evaluate();
    auto masked = pick & mask;
    unsigned int result = topIdx;
    if (masked > 0) result = bottomIdx;
    return result;
  }
};

template<unsigned int round>
struct RoundMultiplier;

template<>
struct RoundMultiplier<1>
{
  //static DEVICE float Evaluate() { return 1.0f; }
  static DEVICE float Evaluate() { return 10.0f; }
};

template<>
struct RoundMultiplier<2>
{
  //static DEVICE float Evaluate() { return 2.0f; }
  static DEVICE float Evaluate() { return 20.0f; }
};

template<>
struct RoundMultiplier<3>
{
  //static DEVICE float Evaluate() { return 4.0f; }
  static DEVICE float Evaluate() { return 40.0f; }
};


template<>
struct RoundMultiplier<4>
{
  //static DEVICE float Evaluate() { return 8.0f; }
  static DEVICE float Evaluate() { return 80.0f; }
};

template<>
struct RoundMultiplier<5>
{
  //static DEVICE float Evaluate() { return 16.0f; }
  static DEVICE float Evaluate() { return 160.0f; }
};

template <unsigned int round, unsigned int game>
struct ExpectedResult
{
  static DEVICE float Evaluate(unsigned int pick,
                                   const float* __restrict__ pOdds,
                                   const float* __restrict__ pSeeds)
  {
    if( pick == 0 )
    {
      auto idx = GetIndex<round, game>::Evaluate(pick);
      auto odds = pOdds[idx+32*(round-1)];
      auto seed = pSeeds[idx];
      auto mult = RoundMultiplier<round>::Evaluate();
      printf("Round %d Game %d Idx %d odds %f Seed %f multiplier %f\n",
             round, game, idx, odds, seed, mult);
    }
    auto idx = GetIndex<round, game>::Evaluate(pick);
    // CBS
    //return pOdds[idx+32*(round-1)] * pSeeds[idx] * RoundMultiplier<round>::Evaluate();
    //return pOdds[idx+32*(round-1)];

    // 538
     return pOdds[idx+32*(round-1)]  * RoundMultiplier<round>::Evaluate();
  }
};


#endif
