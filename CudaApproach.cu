//#include "CudaApproach.cuh"

//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>
//#include <memory>
//#include <iostream>
//#include <cuda_profiler_api.h>
//#include "BracketOdds.hpp"
//#include "Util.hpp"
//#include <stdio.h>

//namespace impl
//{

//  template <unsigned int round, unsigned int game>
//  struct ExpectedTreeResult
//  {
//    static __device__ float Evaluate(unsigned int pick,
//                                     const float* __restrict__ pOdds,
//                                     const float* __restrict__ pSeeds)
//    {
//      return ExpectedTreeResult<round - 1, 2 * game>::Evaluate(
//               pick, pOdds, pSeeds) +
//             ExpectedTreeResult<round - 1, 2 * game + 1>::Evaluate(
//               pick, pOdds, pSeeds) +
//             ExpectedResult<round, game>::Evaluate(pick, pOdds, pSeeds);
//    }
//  };

//  template <unsigned int game>
//  struct ExpectedTreeResult<1, game>
//  {
//    static __device__ float Evaluate(unsigned int pick,
//                                     const float* __restrict__ pOdds,
//                                     const float* __restrict__ pSeeds)
//    {
//      return ExpectedResult<1, game>::Evaluate(pick, pOdds, pSeeds);
//    }
//  };

//  template <unsigned int round, unsigned int game>
//  struct EvaluateGames
//  {
//    static __device__ float Evaluate(unsigned int pick,
//                                     const float* __restrict__ pOdds,
//                                     const float* __restrict__ pSeeds)
//    {
//      return ExpectedTreeResult<round, game>::Evaluate(pick, pOdds, pSeeds) +
//             EvaluateGames<round, game - 1>::Evaluate(pick, pOdds, pSeeds);
//    }
//  };

//  template <unsigned int round>
//  struct EvaluateGames<round, 0>
//  {
//    static __device__ float Evaluate(unsigned int pick,
//                                     const float* __restrict__ pOdds,
//                                     const float* __restrict__ pSeeds)
//    {
//      return ExpectedTreeResult<round, 0>::Evaluate(pick, pOdds, pSeeds);
//    }
//  };

//  //  template <unsigned int game>
//  //  struct EvaluateGames<1, game>
//  //  {
//  //    static __device__ float Evaluate(unsigned int pick,
//  //                                     const float* __restrict__ pOdds,
//  //                                     const float* __restrict__ pSeeds)
//  //    {
//  //      return EvaluateGames<1, game - 1>::Evaluate(pick, pOdds, pSeeds) +
//  //             ExpectedResult<1, game>::Evaluate(pick, pOdds, pSeeds);
//  //    }
//  //  };

//  //  template <>
//  //  struct EvaluateGames<1, 0>
//  //  {
//  //    static __device__ float Evaluate(unsigned int pick,
//  //                                     const float* __restrict__ pOdds,
//  //                                     const float* __restrict__ pSeeds)
//  //    {
//  //      return ExpectedResult<1, 0>::Evaluate(pick, pOdds, pSeeds);
//  //    }
//  //  };

//  // All of the below assumes 32 team brackets, since we'll easily be able
//  // to run them.  For some spot checking I can compare against CPU calculated
//  // 16 team brackets.
//  // Runs the first round of games and returns the expected score.
//  // pick - The 31 bit number representing all picks.  The first 16 are used for
//  // the first round.  A 1 means the top team wins, a 0 the bottom team.
//  // pOdds - 32 entry array of the odds that each team wins their first round
//  // game.  Ordering is as the standard ordering of brackets.
//  // pSeeds = 32 entry array of the seeds.
//  // Returns - the expected score.
//  __device__ float runFirstRound(unsigned int pick,
//                                 const float* __restrict__ pOdds,
//                                 const float* __restrict__ pSeeds)
//  {
//    return EvaluateGames<1, 15>::Evaluate(pick, pOdds, pSeeds);
//  }

//  __device__ float runSecondRound(unsigned int pick,
//                                  const float* __restrict__ pOdds,
//                                  const float* __restrict__ pSeeds)
//  {
//    return EvaluateGames<2, 7>::Evaluate(pick, pOdds, pSeeds);
//  }

//  __device__ float runThirdRound(unsigned int pick,
//                                 const float* __restrict__ pOdds,
//                                 const float* __restrict__ pSeeds)
//  {
//    return EvaluateGames<3, 3>::Evaluate(pick, pOdds, pSeeds);
//  }

//  __device__ float runFourthRound(unsigned int pick,
//                                  const float* __restrict__ pOdds,
//                                  const float* __restrict__ pSeeds)
//  {
//    return EvaluateGames<4, 1>::Evaluate(pick, pOdds, pSeeds);
//  }

//  __device__ float runFifthRound(unsigned int pick,
//                                 const float* __restrict__ pOdds,
//                                 const float* __restrict__ pSeeds)
//  {
//    return EvaluateGames<5, 0>::Evaluate(pick, pOdds, pSeeds);
//  }

//  __global__ void runArrayedOneRound(float* pBestScore,
//                                     unsigned int* pBestPick,
//                                     float* pOdds,
//                                     float* pSeeds)
//  {
//    float bestScore = 0.0f;
//    unsigned int bestPick = 0;
//    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
//    unsigned int gridSize = gridDim.x * blockDim.x;
//    unsigned int maxInt = 0x0000FFFF;
//    // Assumes a power of 2 grid size.
//    unsigned int numIterations = maxInt / gridSize;
//    for (int i = 0; i <= numIterations; ++i)
//    {

//      unsigned int pick = idx + i * gridSize;
//      auto score = runFirstRound(pick, pOdds, pSeeds);
//      if (score > bestScore)
//      {
//        bestScore = score;
//        bestPick = pick;
//      }
//    }

//    pBestScore[idx] = bestScore;
//    pBestPick[idx] = bestPick;
//  }

//  __global__ void runArrayedTwoRounds(float* pBestScore,
//                                      unsigned int* pBestPick,
//                                      float* pOdds,
//                                      float* pSeeds)
//  {
//    float bestScore = 0.0f;
//    unsigned int bestPick = 0;
//    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
//    unsigned int gridSize = gridDim.x * blockDim.x;
//    unsigned int maxInt = 0x00FFFFFF;
//    // Assumes a power of 2 grid size.
//    unsigned int numIterations = maxInt / gridSize + 2;
//    for (int i = 0; i <= numIterations; ++i)
//    {
//      unsigned int pick = idx + i * gridSize;
//      if (pick > maxInt) continue;

//      auto score = runSecondRound(pick, pOdds, pSeeds);
//      if (score > bestScore)
//      {
//        bestScore = score;
//        bestPick = pick;
//      }
//    }

//    pBestScore[idx] = bestScore;
//    pBestPick[idx] = bestPick;
//  }

//  __global__ void runArrayedThreeRounds(float* pBestScore,
//                                        unsigned int* pBestPick,
//                                        float* pOdds,
//                                        float* pSeeds)
//  {
//    float bestScore = 0.0f;
//    unsigned int bestPick = 0;
//    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
//    unsigned int gridSize = gridDim.x * blockDim.x;
//    unsigned int maxInt = 0x0FFFFFFF;
//    // Assumes a power of 2 grid size.
//    unsigned int numIterations = maxInt / gridSize + 2;
//    for (int i = 0; i <= numIterations; ++i)
//    {
//      unsigned int pick = idx + i * gridSize;
//      if (pick > maxInt) continue;

//      auto score = runThirdRound(pick, pOdds, pSeeds);
//      if (score > bestScore)
//      {
//        bestScore = score;
//        bestPick = pick;
//      }
//    }

//    pBestScore[idx] = bestScore;
//    pBestPick[idx] = bestPick;
//  }

//  __global__ void runArrayedFourRounds(float* pBestScore,
//                                       unsigned int* pBestPick,
//                                       float* pOdds,
//                                       float* pSeeds)
//  {
//    float bestScore = 0.0f;
//    unsigned int bestPick = 0;
//    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
//    unsigned int gridSize = gridDim.x * blockDim.x;
//    unsigned int maxInt = 0x3FFFFFFF;
//    // Assumes a power of 2 grid size.
//    unsigned int numIterations = maxInt / gridSize + 2;
//    for (int i = 0; i <= numIterations; ++i)
//    {
//      if (idx == 0)
//      {
//        printf("%032x\n", i);
//      }
//      unsigned int pick = idx + i * gridSize;
//      if (pick > maxInt) continue;

//      auto score = runFourthRound(pick, pOdds, pSeeds);
//      if (score > bestScore)
//      {
//        bestScore = score;
//        bestPick = pick;
//      }
//    }

//    pBestScore[idx] = bestScore;
//    pBestPick[idx] = bestPick;
//  }

//  __global__ void runArrayedFiveRounds(float* pBestScore,
//                                       unsigned int* pBestPick,
//                                       float* pOdds,
//                                       float* pSeeds)
//  {
//    float bestScore = 0.0f;
//    unsigned int bestPick = 0;
//    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
//    unsigned int gridStride = gridDim.x * blockDim.x;
//    unsigned int maxInt = 0x7FFFFFFF;

//    for (int pick = blockDim.x * blockIdx.x + threadIdx.x; pick < maxInt;
//         pick += gridStride)
//    {
//      auto score = runFifthRound(pick, pOdds, pSeeds);
//      if (idx == 0)
//      {
//         //printf("%032x = %f\n", pick, score);
//         //printf("%d", i);
//      }
//      if (score > bestScore)
//      {
//        bestScore = score;
//        bestPick = pick;
//      }
//    }

//    pBestScore[idx] = bestScore;
//    pBestPick[idx] = bestPick;
//  }
//}

//void runCudaApproach(const Five38BracketOdds& odds)
//{
//  dim3 blockDim {256, 1, 1};
//  dim3 gridDim {1024, 1, 1};

//  // Each thread examines multiple games (using grid striding).  After it
//  // calculates a game, it ccompares the results to all other games it has
//  // evaluated and stores the score and pick in these arrays.  Additional
//  // processing is required to find the global best.
//  float* pdBestScore;
//  unsigned int* pdBestPick;
//  cudaMalloc(&pdBestScore, sizeof(float) * blockDim.x * gridDim.x);
//  cudaMalloc(&pdBestPick, sizeof(unsigned int) * blockDim.x * gridDim.x);

//  float* pdSeeds;
//  cudaMalloc(&pdSeeds, sizeof(float) * 32);
//  float* pdOdds;
//  cudaMalloc(&pdOdds, sizeof(float) * 32 * 6);

//  cudaMemcpy(
//    pdSeeds, odds.m_seeds.data(), sizeof(float) * 32, cudaMemcpyHostToDevice);
//  cudaMemcpy(pdOdds,
//             odds.m_roundOdds[0].data(),
//             sizeof(float) * 32,
//             cudaMemcpyHostToDevice);
//  cudaMemcpy(pdOdds + 1 * 32,
//             odds.m_roundOdds[1].data(),
//             sizeof(float) * 32,
//             cudaMemcpyHostToDevice);
//  cudaMemcpy(pdOdds + 2 * 32,
//             odds.m_roundOdds[2].data(),
//             sizeof(float) * 32,
//             cudaMemcpyHostToDevice);
//  cudaMemcpy(pdOdds + 3 * 32,
//             odds.m_roundOdds[3].data(),
//             sizeof(float) * 32,
//             cudaMemcpyHostToDevice);
//  cudaMemcpy(pdOdds + 4 * 32,
//             odds.m_roundOdds[4].data(),
//             sizeof(float) * 32,
//             cudaMemcpyHostToDevice);
//  cudaMemcpy(pdOdds + 5 * 32,
//             odds.m_roundOdds[5].data(),
//             sizeof(float) * 32,
//             cudaMemcpyHostToDevice);

//  // impl::runArrayedOneRound << <blockDim, gridDim>>> (pdBestScore, pdBestPick,
//  // pdOdds, pdSeeds);
//  // impl::runArrayedTwoRounds << <blockDim, gridDim>>> (pdBestScore,
//  // pdBestPick, pdOdds, pdSeeds);
//  // impl::runArrayedThreeRounds << <blockDim, gridDim>>> (pdBestScore,
//  // pdBestPick, pdOdds, pdSeeds);
//  // impl::runArrayedFourRounds << <blockDim, gridDim>>> (pdBestScore,
//  // pdBestPick, pdOdds, pdSeeds);
//  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 2000000000);
//  impl::runArrayedFiveRounds<<<blockDim, gridDim>>>(
//    pdBestScore, pdBestPick, pdOdds, pdSeeds);
//  auto result = cudaDeviceSynchronize();


//  float* pBestScore = new float[blockDim.x * gridDim.x];
//  unsigned int* pBestPick = new unsigned int[blockDim.x * gridDim.x];
//  cudaMemcpy(pBestScore,
//             pdBestScore,
//             sizeof(float) * blockDim.x * gridDim.x,
//             cudaMemcpyDeviceToHost);
//  cudaMemcpy(pBestPick,
//             pdBestPick,
//             sizeof(unsigned int) * blockDim.x * gridDim.x,
//             cudaMemcpyDeviceToHost);

//  float bestScore = 0.0f;
//  unsigned int bestPick = 0;
//  for (unsigned int i = 0; i < blockDim.x * gridDim.x; ++i)
//  {
//    if (pBestScore[i] > bestScore)
//    {
//      bestScore = pBestScore[i];
//      bestPick = pBestPick[i];
//    }
//  }

//  std::cout << "Expected Score = " << bestScore << std::endl;
//  std::cout << "Pick: " << bestPick << std::endl;
//  std::cout << std::endl;
//  std::cout << "First Round" << std::endl;

//  std::cout << odds.m_names[GetWinningTeamId<1, 0>::Evaluate(bestPick)]
//            << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<1, 1>::Evaluate(bestPick)]
//            << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<1, 2>::Evaluate(bestPick)]
//            << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<1, 3>::Evaluate(bestPick)]
//            << std::endl;

//  std::cout << odds.m_names[GetWinningTeamId<1, 4>::Evaluate(bestPick)]
//            << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<1, 5>::Evaluate(bestPick)]
//            << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<1, 6>::Evaluate(bestPick)]
//            << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<1, 7>::Evaluate(bestPick)]
//            << std::endl;

//  std::cout << odds.m_names[GetWinningTeamId<1, 8>::Evaluate(bestPick)]
//            << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<1, 9>::Evaluate(bestPick)]
//            << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<1, 10>::Evaluate(bestPick)]
//            << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<1, 11>::Evaluate(bestPick)]
//            << std::endl;

//  std::cout << odds.m_names[GetWinningTeamId<1, 12>::Evaluate(bestPick)]
//            << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<1, 13>::Evaluate(bestPick)]
//            << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<1, 14>::Evaluate(bestPick)]
//            << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<1, 15>::Evaluate(bestPick)]
//            << std::endl;

//  std::cout << std::endl;
//  std::cout << "Second Round" << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<2, 0>::Evaluate(bestPick)]
//            << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<2, 1>::Evaluate(bestPick)]
//            << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<2, 2>::Evaluate(bestPick)]
//            << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<2, 3>::Evaluate(bestPick)]
//            << std::endl;

//  std::cout << odds.m_names[GetWinningTeamId<2, 4>::Evaluate(bestPick)]
//            << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<2, 5>::Evaluate(bestPick)]
//            << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<2, 6>::Evaluate(bestPick)]
//            << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<2, 7>::Evaluate(bestPick)]
//            << std::endl;

//  std::cout << std::endl;
//  std::cout << "Third round" << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<3, 0>::Evaluate(bestPick)]
//            << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<3, 1>::Evaluate(bestPick)]
//            << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<3, 2>::Evaluate(bestPick)]
//            << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<3, 3>::Evaluate(bestPick)]
//            << std::endl;

//  std::cout << std::endl;
//  std::cout << "Fourth round" << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<4, 0>::Evaluate(bestPick)]
//            << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<4, 1>::Evaluate(bestPick)]
//            << std::endl;

//  std::cout << std::endl;
//  std::cout << "Fifth round" << std::endl;
//  std::cout << odds.m_names[GetWinningTeamId<5, 0>::Evaluate(bestPick)]
//            << std::endl;
//  if (GetWinningTeamId<5, 0>::Evaluate(bestPick) > 15)
//  {
//    std::cout << "Lower" << std::endl;
//  }
//  else
//  {
//    std::cout << "Upper" << std::endl;
//  }


////  std::cout << "Best picks for each block" << std::endl;
////  for (unsigned int i = 0; i < blockDim.x * gridDim.x; ++i)
////  {
////    auto bestPick = pBestPick[i];
////    auto bestScore = pBestScore[i];
////    std::cout << "Expected Score = " << bestScore << std::endl;
////    std::cout << "Pick: " << bestPick << std::endl;
////    std::cout << std::endl;
////    std::cout << "First Round" << std::endl;

////    std::cout << odds.m_names[GetWinningTeamId<1, 0>::Evaluate(bestPick)]
////              << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<1, 1>::Evaluate(bestPick)]
////              << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<1, 2>::Evaluate(bestPick)]
////              << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<1, 3>::Evaluate(bestPick)]
////              << std::endl;

////    std::cout << odds.m_names[GetWinningTeamId<1, 4>::Evaluate(bestPick)]
////              << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<1, 5>::Evaluate(bestPick)]
////              << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<1, 6>::Evaluate(bestPick)]
////              << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<1, 7>::Evaluate(bestPick)]
////              << std::endl;

////    std::cout << odds.m_names[GetWinningTeamId<1, 8>::Evaluate(bestPick)]
////              << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<1, 9>::Evaluate(bestPick)]
////              << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<1, 10>::Evaluate(bestPick)]
////              << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<1, 11>::Evaluate(bestPick)]
////              << std::endl;

////    std::cout << odds.m_names[GetWinningTeamId<1, 12>::Evaluate(bestPick)]
////              << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<1, 13>::Evaluate(bestPick)]
////              << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<1, 14>::Evaluate(bestPick)]
////              << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<1, 15>::Evaluate(bestPick)]
////              << std::endl;

////    std::cout << std::endl;
////    std::cout << "Second Round" << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<2, 0>::Evaluate(bestPick)]
////              << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<2, 1>::Evaluate(bestPick)]
////              << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<2, 2>::Evaluate(bestPick)]
////              << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<2, 3>::Evaluate(bestPick)]
////              << std::endl;

////    std::cout << odds.m_names[GetWinningTeamId<2, 4>::Evaluate(bestPick)]
////              << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<2, 5>::Evaluate(bestPick)]
////              << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<2, 6>::Evaluate(bestPick)]
////              << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<2, 7>::Evaluate(bestPick)]
////              << std::endl;

////    std::cout << std::endl;
////    std::cout << "Third round" << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<3, 0>::Evaluate(bestPick)]
////              << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<3, 1>::Evaluate(bestPick)]
////              << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<3, 2>::Evaluate(bestPick)]
////              << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<3, 3>::Evaluate(bestPick)]
////              << std::endl;

////    std::cout << std::endl;
////    std::cout << "Fourth round" << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<4, 0>::Evaluate(bestPick)]
////              << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<4, 1>::Evaluate(bestPick)]
////              << std::endl;

////    std::cout << std::endl;
////    std::cout << "Fifth round" << std::endl;
////    std::cout << odds.m_names[GetWinningTeamId<5, 0>::Evaluate(bestPick)]
////              << std::endl;
////  }


//}
