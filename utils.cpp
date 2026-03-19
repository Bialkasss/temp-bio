#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <functional>
#include <cassert>

// ─────────────────────────────────────────────
//  RNG setup
// ─────────────────────────────────────────────
static std::mt19937 rng(std::random_device{}());

// Returns a uniform random integer in [0, range)
static int randInt(int range) {
    return std::uniform_int_distribution<int>(0, range - 1)(rng);
}


// ─────────────────────────────────────────────
//  1. RANDOM PERMUTATION
//  Pseudocode:
//    have list [0,1,...,n-1]
//    for i in range n:
//        x = random(n - i)          <- draw from remaining
//        put element at position x at the end of stable part
//  Result: Fisher-Yates (Knuth) shuffle
// ─────────────────────────────────────────────
std::vector<int> generatePermutation(int n) {

    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i)
        perm[i] = i;                  // initialise [0, 1, 2, ..., n-1]

    for (int i = 0; i < n; ++i) {
        // Draw uniformly from the NOT-yet-drawn part: indices [i, n)
        int x = i + randInt(n - i);   // random index in [i, n-1]

        // Swap chosen element to position i (stable / drawn part grows left)
        std::swap(perm[i], perm[x]);
    }

    return perm;
}


// ─────────────────────────────────────────────
//  2. RANDOM PAIR OF DISTINCT INDICES in [0, n)
//  Pseudocode:
//    y = rand(n)
//    x = (y + rand(n-1) + 1) % n    <- guarantees x != y
// ─────────────────────────────────────────────
std::pair<int,int> randomPair(int n) {
    int y = randInt(n);
    int x = (y + randInt(n - 1) + 1) % n;  // shift by [1, n-1] mod n

    return {x, y};
}


// ─────────────────────────────────────────────
//  3. MEASURE ALGORITHM RUNTIME
//  Pseudocode:
//    t0 = time()
//    counter = 0
//    do {
//        f()
//        counter++
//    } while (time() - t0 < T  OR  counter < minRuns)
//    runtime = (time() - t0) / counter
//
//  T should be >> minimal clock resolution (e.g. 100× larger).
//  Returns average single-call duration in seconds.
// ─────────────────────────────────────────────
double measureRuntime(
    std::function<void()> f,
    double T       = 1.0,   // minimum wall-clock budget (seconds)
    int    minRuns = 10     // minimum number of calls regardless of time
) {
    using clock     = std::chrono::high_resolution_clock;
    using dseconds  = std::chrono::duration<double>;

    auto t0      = clock::now();
    int  counter = 0;

    do {
        f();
        ++counter;
    } while (
        dseconds(clock::now() - t0).count() < T   // haven't spent T seconds yet
        || counter < minRuns                       // haven't hit minimum runs yet
    );

    double elapsed = dseconds(clock::now() - t0).count();
    return elapsed / counter;   // average time per call
}
