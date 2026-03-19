// ============================================================
//  QAP Solver — Batch folder mode
//  Heuristics implemented:
//    1. Greedy construction (best-fit row-by-row assignment)
//    2. 2-OPT Steepest Descent  (SD)  — scan all pairs, take best improving swap
//    3. 2-OPT First Improvement (FI)  — scan pairs, take FIRST improving swap
//
//  All heuristics start from a random permutation (generatePermutation).
//  Greedy construction builds a better starting point; both 2-opt variants
//  then refine it.  The solver runs each heuristic independently N times,
//  collects min/avg/max time and objective, and writes one CSV row per
//  (instance × heuristic).
//
//  Compile:  g++ -O2 -std=c++17 utils.cpp qap.cpp -o qap
//  Usage:    ./qap <folder_path> [runs_per_instance=10]
//  Output:   <folder_path>/results.csv
// ============================================================
#include "utils.h" // generatePermutation, randomPair, measureRuntime

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <climits>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;

// ─────────────────────────────────────────
//  Data structures
// ─────────────────────────────────────────
struct QAP
{
    int n = 0;
    std::vector<std::vector<int>> F; // flow matrix     a_ij
    std::vector<std::vector<int>> D; // distance matrix d_rs
};

struct Solution
{
    int n = 0;
    long long knownOpt = LLONG_MIN;
    std::vector<int> perm; // 0-based
    bool loaded = false;
};

struct RunMetrics
{
    std::vector<int> perm;
    long long objValue = LLONG_MAX;
    double runtimeSec = 0.0;
    long long numSwaps = 0;  // accepted moves
    long long numDeltas = 0; // delta evaluations
};

struct InstanceResult
{
    std::string instance;
    std::string heuristic;
    int n = 0;
    int runs = 0;
    long long bestObj = LLONG_MAX;
    std::vector<int> bestPerm;
    double minTime = 1e18, avgTime = 0.0, maxTime = 0.0;
    double avgObj = 0.0;
    long long minObj = LLONG_MAX, maxObj = LLONG_MIN;
    double avgSwaps = 0.0, avgDeltas = 0.0;
    long long knownOpt = LLONG_MIN;
    bool hasSln = false;
};

// ─────────────────────────────────────────
//  File I/O
// ─────────────────────────────────────────
QAP loadDAT(const fs::path& p) {
    std::ifstream f(p);
    if (!f) { std::cerr << "[ERROR] Cannot open " << p << "\n"; exit(1); }
    QAP q;
    if (!(f >> q.n) || q.n <= 0 || q.n > 10000) {  // ← guard
        std::cerr << "[ERROR] Bad n in " << p << "\n"; exit(1);
    }
    int n = q.n;
    q.F.assign(n, std::vector<int>(n));
    q.D.assign(n, std::vector<int>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (!(f >> q.F[i][j])) {   // ← guard
                std::cerr << "[ERROR] Truncated F matrix in " << p
                          << " at (" << i << "," << j << ")\n"; exit(1);
            }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (!(f >> q.D[i][j])) {   // ← guard
                std::cerr << "[ERROR] Truncated D matrix in " << p
                          << " at (" << i << "," << j << ")\n"; exit(1);
            }
    return q;
}

Solution loadSLN(const fs::path &p)
{
    Solution s;
    std::ifstream f(p);
    if (!f)
        return s;
    f >> s.n >> s.knownOpt;
    s.perm.resize(s.n);
    for (int i = 0; i < s.n; ++i)
    {
        f >> s.perm[i];
        s.perm[i] -= 1;
    }
    s.loaded = true;
    return s;
}

// ─────────────────────────────────────────
//  Objective  f(π) = Σ_ij a[i][j] * d[π[i]][π[j]]
// ─────────────────────────────────────────
long long objective(const QAP &q, const std::vector<int> &perm)
{
    long long v = 0;
    for (int i = 0; i < q.n; ++i)
        for (int j = 0; j < q.n; ++j)
            v += (long long)q.F[i][j] * q.D[perm[i]][perm[j]];
    return v;
}

// ─────────────────────────────────────────
//  Delta evaluation — O(n) per swap
//  Full asymmetric formula (slides 13-15):
//
//  Δ(π,r,s) =
//    a_rr*(d[πs][πs]-d[πr][πr]) + a_rs*(d[πs][πr]-d[πr][πs])
//  + a_sr*(d[πr][πs]-d[πs][πr]) + a_ss*(d[πr][πr]-d[πs][πs])
//  + Σ_{k≠r,s} [ a_kr*(d[πk][πs]-d[πk][πr]) + a_ks*(d[πk][πr]-d[πk][πs])
//              + a_rk*(d[πs][πk]-d[πr][πk]) + a_sk*(d[πr][πk]-d[πs][πk]) ]
// ─────────────────────────────────────────
long long deltaSwap(const QAP &q, const std::vector<int> &perm, int r, int s)
{
    const auto &a = q.F;
    const auto &d = q.D;
    int pr = perm[r], ps = perm[s];

    long long delta =
        (long long)a[r][r] * (d[ps][ps] - d[pr][pr]) + (long long)a[r][s] * (d[ps][pr] - d[pr][ps]) + (long long)a[s][r] * (d[pr][ps] - d[ps][pr]) + (long long)a[s][s] * (d[pr][pr] - d[ps][ps]);

    for (int k = 0; k < q.n; ++k)
    {
        if (k == r || k == s)
            continue;
        int pk = perm[k];
        delta += (long long)a[k][r] * (d[pk][ps] - d[pk][pr]) + (long long)a[k][s] * (d[pk][pr] - d[pk][ps]) + (long long)a[r][k] * (d[ps][pk] - d[pr][pk]) + (long long)a[s][k] * (d[pr][pk] - d[ps][pk]);
    }
    return delta;
}

// ═══════════════════════════════════════════════════════════
//  HEURISTIC 1 — GREEDY CONSTRUCTION
//
//  Builds an initial permutation by greedily assigning objects
//  to locations one at a time.
//
//  Algorithm:
//    - Compute total flow out of each object:  flowSum[i] = Σ_j a[i][j]
//    - Compute total distance from each location: distSum[r] = Σ_s d[r][s]
//    - Sort objects descending by flowSum  (most "connected" first)
//    - Sort locations descending by distSum (most "central" first)  [WRONG direction]
//      Actually sort locations ASCENDING by distSum (least costly first)
//    - Assign object with rank k → location with rank k
//      i.e. highest-flow object gets the lowest-distance location
//
//  This is a classic O(n² log n) greedy that gives a good starting point.
//  After construction we apply a random perturbation so that repeated
//  runs from the same greedy solution are not all identical.
// ═══════════════════════════════════════════════════════════
std::vector<int> greedyPermutation(const QAP &q)
{
    int n = q.n;

    // total flow out of each object i
    std::vector<long long> flowSum(n, 0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            flowSum[i] += q.F[i][j];

    // total distance from each location r
    std::vector<long long> distSum(n, 0);
    for (int r = 0; r < n; ++r)
        for (int s = 0; s < n; ++s)
            distSum[r] += q.D[r][s];

    // sort objects: highest flow first
    std::vector<int> objOrder(n);
    std::iota(objOrder.begin(), objOrder.end(), 0);
    std::sort(objOrder.begin(), objOrder.end(),
              [&](int a, int b)
              { return flowSum[a] > flowSum[b]; });

    // sort locations: lowest total distance first (cheapest to be at)
    std::vector<int> locOrder(n);
    std::iota(locOrder.begin(), locOrder.end(), 0);
    std::sort(locOrder.begin(), locOrder.end(),
              [&](int a, int b)
              { return distSum[a] < distSum[b]; });

    // assign: object objOrder[k] → location locOrder[k]
    // perm[i] = location of object i
    std::vector<int> perm(n);
    for (int k = 0; k < n; ++k)
        perm[objOrder[k]] = locOrder[k];

    return perm;
}

RunMetrics runGreedy(const QAP &q)
{
    RunMetrics m;
    // Start from greedy construction — deterministic, so we add a small
    // random perturbation (one random swap) so repeated runs differ.
    m.perm = greedyPermutation(q);
    if (q.n > 1)
    {
        auto [pi, pj] = randomPair(q.n);   // random distinct pair from utils.cpp
        std::swap(m.perm[pi], m.perm[pj]); // single perturbation
    }
    m.objValue = objective(q, m.perm);
    m.numSwaps = 0; // no search phase — pure construction
    m.numDeltas = 0;
    return m;
}

// ═══════════════════════════════════════════════════════════
//  HEURISTIC 2 — 2-OPT STEEPEST DESCENT (SD)
//
//  Each iteration scans ALL unique (i,j) pairs using deltaSwap.
//  The best improving swap (most negative delta) is applied.
//  Stops when no improving swap exists → local optimum.
//
//  Pair iteration (avoids diagonal, covers all n*(n-1)/2 unique pairs):
//    r  = random start index (randomises which row we begin from)
//    for ii = 0 .. n-2:          ← n-1 outer steps
//      i  = (r + ii) % n
//      for jj = 1 .. n-1-ii:     ← shrinking inner range avoids repeats
//        j  = (i + jj) % n
//        evaluate delta(i, j)     ← always use sorted indices inside deltaSwap
//
//  This guarantees every pair is visited exactly once regardless of r.
//
//  Complexity per iteration: O(n³)  (n*(n-1)/2 pairs × O(n) delta each)
// ═══════════════════════════════════════════════════════════
RunMetrics run2OptSD(const QAP &q)
{
    RunMetrics m;
    m.perm = generatePermutation(q.n); // random start (utils.cpp)
    long long fCur = objective(q, m.perm);

    bool improved = true;
    while (improved)
    {
        improved = false;
        long long bestDelta = 0;
        int bestI = -1, bestJ = -1;

        // randomPair gives a uniform random start index r in [0,n)
        auto [r, dummy] = randomPair(q.n);

        for (int ii = 0; ii < q.n - 1; ++ii)
        {
            int i = (r + ii) % q.n;
            for (int jj = ii + 1; jj < q.n; ++jj)
            { // ← was: jj=1..n-1-ii
                int j = (r + jj) % q.n;
                long long d = deltaSwap(q, m.perm, std::min(i, j), std::max(i, j));
                ++m.numDeltas;
                if (d < bestDelta)
                {
                    bestDelta = d;
                    bestI = i;
                    bestJ = j;
                }
            }
        }

        if (bestI != -1)
        {
            std::swap(m.perm[bestI], m.perm[bestJ]);
            fCur += bestDelta;
            ++m.numSwaps;
            improved = true;
        }
    }

    m.objValue = fCur;
    return m;
}

// ═══════════════════════════════════════════════════════════
//  HEURISTIC 3 — 2-OPT FIRST IMPROVEMENT (FI)
//
//  Same neighbourhood and delta formula as SD, but accepts the
//  FIRST improving swap found and immediately restarts the scan.
//
//  Same correct pair-iteration structure as SD:
//    r  = random start
//    ii = 0..n-2,  i = (r+ii)%n
//    jj = 1..n-1-ii,  j = (i+jj)%n
//  → all n*(n-1)/2 unique pairs, no repeats, no skips.
//
//  Stops when a full scan (no early exit triggered) finds nothing.
// ═══════════════════════════════════════════════════════════
RunMetrics run2OptFI(const QAP &q)
{
    RunMetrics m;
    m.perm = generatePermutation(q.n); // random start (utils.cpp)
    long long fCur = objective(q, m.perm);

    bool improved = true;
    while (improved)
    {
        improved = false;

        auto [r, dummy] = randomPair(q.n);

        for (int ii = 0; ii < q.n - 1; ++ii)
        {
            int i = (r + ii) % q.n;
            for (int jj = ii + 1; jj < q.n; ++jj)
            { // ← was: jj=1..n-1-ii
                int j = (r + jj) % q.n;
                long long d = deltaSwap(q, m.perm, std::min(i, j), std::max(i, j));
                ++m.numDeltas;
                if (d < 0)
                { // first improvement
                    std::swap(m.perm[i], m.perm[j]);
                    fCur += d;
                    ++m.numSwaps;
                    improved = true;
                    break; // restart scan from new random r
                }
            }
        }
    }

    m.objValue = fCur;
    return m;
}

// ─────────────────────────────────────────
//  Generic runner — times N independent runs of any heuristic
// ─────────────────────────────────────────
using HeuristicFn = RunMetrics (*)(const QAP &);

InstanceResult runHeuristic(const std::string &instName,
                            const std::string &heurName,
                            HeuristicFn heurFn,
                            const QAP &q,
                            const Solution &sln,
                            int runs)
{
    InstanceResult res;
    res.instance = instName;
    res.heuristic = heurName;
    res.n = q.n;
    res.runs = runs;
    res.hasSln = sln.loaded;
    res.knownOpt = sln.knownOpt;

    double sumTime = 0, sumObj = 0, sumSwaps = 0, sumDeltas = 0;

    for (int r = 0; r < runs; ++r)
    {
        RunMetrics m;

        // measureRuntime (utils.cpp): T=0, minRuns=1 → single timed call
        double elapsed = measureRuntime(
            [&]()
            { m = heurFn(q); },
            0.0, // no time budget — just measure one call
            1);
        m.runtimeSec = elapsed;

        sumTime += m.runtimeSec;
        sumObj += (double)m.objValue;
        sumSwaps += (double)m.numSwaps;
        sumDeltas += (double)m.numDeltas;

        if (m.runtimeSec < res.minTime)
            res.minTime = m.runtimeSec;
        if (m.runtimeSec > res.maxTime)
            res.maxTime = m.runtimeSec;
        if (m.objValue < res.minObj)
            res.minObj = m.objValue;
        if (m.objValue > res.maxObj)
            res.maxObj = m.objValue;
        if (m.objValue < res.bestObj)
        {
            res.bestObj = m.objValue;
            res.bestPerm = m.perm;
        }

        std::cout << "    [" << std::setw(3) << r + 1 << "/" << runs << "] "
                  << std::left << std::setw(6) << heurName << std::right
                  << "  obj=" << std::setw(14) << m.objValue
                  << "  t=" << std::fixed << std::setprecision(6) << m.runtimeSec << "s"
                  << "  swaps=" << std::setw(5) << m.numSwaps
                  << "  Δ-evals=" << m.numDeltas << "\n";
    }

    res.avgTime = sumTime / runs;
    res.avgObj = sumObj / runs;
    res.avgSwaps = sumSwaps / runs;
    res.avgDeltas = sumDeltas / runs;
    return res;
}

// ─────────────────────────────────────────
//  CSV
// ─────────────────────────────────────────
void writeCSVHeader(std::ofstream &f)
{
    f << "instance,heuristic,n,runs,"
         "best_obj,min_obj,avg_obj,max_obj,"
         "known_opt,gap_best_pct,"
         "min_time_s,avg_time_s,max_time_s,"
         "avg_swaps,avg_deltas,"
         "best_permutation\n";
}

void writeCSVRow(std::ofstream &f, const InstanceResult &r)
{
    std::string gapStr = "N/A";
    if (r.hasSln && r.knownOpt != 0)
        gapStr = std::to_string(
            100.0 * (r.bestObj - r.knownOpt) / std::abs((double)r.knownOpt));

    f << r.instance << ","
      << r.heuristic << ","
      << r.n << "," << r.runs << ","
      << r.bestObj << "," << r.minObj << ","
      << std::fixed << std::setprecision(2) << r.avgObj << ","
      << r.maxObj << ","
      << (r.hasSln ? std::to_string(r.knownOpt) : "N/A") << ","
      << gapStr << ","
      << std::setprecision(6) << r.minTime << ","
      << r.avgTime << "," << r.maxTime << ","
      << std::setprecision(1) << r.avgSwaps << "," << r.avgDeltas << ","
      << "\"";
    for (int i = 0; i < (int)r.bestPerm.size(); ++i)
    {
        if (i)
            f << " ";
        f << (r.bestPerm[i] + 1); // 0-based → 1-based (matches .sln)
    }
    f << "\"\n";
}

// ─────────────────────────────────────────
//  Main
// ─────────────────────────────────────────
int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <folder_path> [runs_per_instance=10]\n";
        return 1;
    }

    fs::path folder(argv[1]);
    int runs = (argc >= 3) ? std::stoi(argv[2]) : 10;

    if (!fs::is_directory(folder))
    {
        std::cerr << "[ERROR] Not a directory: " << folder << "\n";
        return 1;
    }

    std::vector<fs::path> datFiles;
    for (auto &e : fs::directory_iterator(folder))
        if (e.is_regular_file() && e.path().extension() == ".dat")
            datFiles.push_back(e.path());
    std::sort(datFiles.begin(), datFiles.end());

    if (datFiles.empty())
    {
        std::cerr << "[ERROR] No .dat files in " << folder << "\n";
        return 1;
    }

    // Write results.csv to current working directory (not data folder, which may be read-only)
    fs::path csvPath = fs::current_path() / "results.csv";
    std::ofstream csv(csvPath);
    if (!csv)
    {
        std::cerr << "[ERROR] Cannot write " << csvPath << "\n";
        return 1;
    }
    writeCSVHeader(csv);

    // heuristics to run on every instance
    struct Heuristic
    {
        std::string name;
        HeuristicFn fn;
    };
    std::vector<Heuristic> heuristics = {
        {"Greedy", runGreedy},
        {"2opt-SD", run2OptSD},
        {"2opt-FI", run2OptFI},
    };

    std::cout << "QAP Batch Solver\n"
              << "Folder   : " << fs::absolute(folder) << "\n"
              << "Instances: " << datFiles.size() << "\n"
              << "Runs/inst: " << runs << "\n"
              << "Heuristics: Greedy, 2opt-SD (steepest descent), 2opt-FI (first improvement)\n"
              << "Output   : " << csvPath << "\n\n";

    for (size_t idx = 0; idx < datFiles.size(); ++idx)
    {

        auto &datPath = datFiles[idx];
        std::string name = datPath.stem().string();
        fs::path slnPath = datPath.parent_path() / (name + ".sln");
        if (idx == 80) {
            std::cout << name << std::endl;
        }

        std::cout << "══════════════════════════════════════════\n"
                << "[" << idx+1 << "/" << datFiles.size() << "] "
                << name << " loading...\n";
        std::cout.flush();

        Solution sln = loadSLN(slnPath);
        QAP q        = loadDAT(datPath);   // crash is in here

        std::cout << "  n=" << q.n;


        if (sln.loaded)
        {
            if (sln.n != q.n || (int)sln.perm.size() != q.n)
            {
                std::cerr << "[WARN] .sln n=" << sln.n
                          << " mismatches .dat n=" << q.n
                          << " — ignoring solution file\n";
                sln.loaded = false;
            }
            else
            {
                long long check = objective(q, sln.perm);
                std::cout << "  sln_opt=" << sln.knownOpt
                          << "  verify=" << check
                          << (check == sln.knownOpt ? " ✓" : " ✗ MISMATCH");
            }
        }
        std::cout << "\n";

        for (auto &h : heuristics)
        {
            std::cout << "  ── " << h.name << " ──\n";
            InstanceResult res = runHeuristic(name, h.name, h.fn, q, sln, runs);

            std::cout << "  Best obj     : " << res.bestObj << "\n"
                      << "  Obj [min,max]: [" << res.minObj << ", " << res.maxObj << "]\n"
                      << "  Time (s)     : min=" << std::fixed << std::setprecision(6)
                      << res.minTime << "  avg=" << res.avgTime
                      << "  max=" << res.maxTime << "\n";
            if (res.hasSln)
                std::cout << "  Gap to opt   : " << std::setprecision(4)
                          << 100.0 * (res.bestObj - res.knownOpt) /
                                 std::abs((double)res.knownOpt)
                          << " %\n";
            std::cout << "\n";

            writeCSVRow(csv, res);
        }
    }

    csv.flush();
    csv.close();
    if (!csv)
    {
        std::cerr << "[ERROR] Failed writing results.csv\n";
        return 1;
    }
    std::cout << "All done.\n"
              << "Results → " << csvPath << "\n"
              << "PDF:     python3 generate_report.py " << csvPath << "\n";
    return 0;
}