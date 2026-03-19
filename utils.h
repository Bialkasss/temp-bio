#pragma once
// ============================================================
//  utils.h — shared utilities
//  Implementations live in utils.cpp
//
//  Provides:
//    generatePermutation(n)   — Fisher-Yates random permutation
//    randomPair(n)            — two distinct indices in [0,n)
//    measureRuntime(f, T, c)  — average wall-clock time per call
// ============================================================
#include <vector>
#include <utility>   // std::pair
#include <functional>

// Returns a uniformly random permutation of [0, n-1]
std::vector<int> generatePermutation(int n);

// Returns a pair of DISTINCT indices (x, y) both in [0, n)
// Uses:  y = rand(n);  x = (y + rand(n-1) + 1) % n
std::pair<int,int> randomPair(int n);

// Runs f() repeatedly for at least T seconds AND at least minRuns times.
// Returns average single-call duration in seconds.
double measureRuntime(
    std::function<void()> f,
    double T       = 1.0,
    int    minRuns = 10
);