#include "omp/shkurinskaya_e_convex_hull_components/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

using namespace shkurinskaya_e_convex_hull_components_omp;

namespace {

// Удвоенная ориентированная площадь (левый поворот > 0)
inline long long TwiceOrientedArea(const Point& a, const Point& b, const Point& c) noexcept {
  const long long abx = static_cast<long long>(b.x) - static_cast<long long>(a.x);
  const long long aby = static_cast<long long>(b.y) - static_cast<long long>(a.y);
  const long long acx = static_cast<long long>(c.x) - static_cast<long long>(a.x);
  const long long acy = static_cast<long long>(c.y) - static_cast<long long>(a.y);
  return (abx * acy) - (aby * acx);
}

inline bool LexiLess(const Point& a, const Point& b) noexcept { return (a.x < b.x) || (a.x == b.x && a.y < b.y); }

inline void DedupSortedInPlace(std::vector<Point>& pts) {
  const auto sub = std::ranges::unique(pts, [](const Point& l, const Point& r) { return l.x == r.x && l.y == r.y; });
  pts.erase(sub.begin(), sub.end());
}

inline void AppendWithLeftTurn(std::vector<Point>& chain, const Point& nxt) {
  while (chain.size() >= 2 && TwiceOrientedArea(chain[chain.size() - 2], chain.back(), nxt) <= 0) {
    chain.pop_back();
  }
  chain.push_back(nxt);
}

inline std::vector<Point> BuildHullMonotone(std::vector<Point> pts) {
  if (pts.size() <= 1) {
    return pts;
  }

  std::ranges::sort(pts, LexiLess);
  DedupSortedInPlace(pts);
  if (pts.size() <= 1) {
    return pts;
  }

  std::vector<Point> lower_chain;
  std::vector<Point> upper_chain;
  lower_chain.reserve(pts.size());
  upper_chain.reserve(pts.size());

  for (const auto& p : pts) {
    AppendWithLeftTurn(lower_chain, p);
  }
  for (auto it = pts.rbegin(); it != pts.rend(); ++it) {
    AppendWithLeftTurn(upper_chain, *it);
  }

  if (!lower_chain.empty()) {
    lower_chain.pop_back();
  }
  if (!upper_chain.empty()) {
    upper_chain.pop_back();
  }

  std::vector<Point> hull;
  hull.reserve(lower_chain.size() + upper_chain.size());
  hull.insert(hull.end(), lower_chain.begin(), lower_chain.end());
  hull.insert(hull.end(), upper_chain.begin(), upper_chain.end());
  return hull;
}

}  // namespace

namespace shkurinskaya_e_convex_hull_components_omp {

int ConvexHullOmp::Cross(const Point& o, const Point& a, const Point& b) noexcept {
  const long long v = TwiceOrientedArea(o, a, b);
  if (v > 0) {
    return 1;
  }
  if (v < 0) {
    return -1;
  }
  return 0;
}

bool ConvexHullOmp::ValidationImpl() {
  if (task_data == nullptr) {
    return false;
  }

  if (task_data->inputs.size() < 3 || task_data->inputs_count.size() < 3) {
    return false;
  }

  const unsigned int n = task_data->inputs_count[0];
  if (n > 0 && task_data->inputs[0] == nullptr) {
    return false;
  }

  if (task_data->inputs_count[1] != 1 || task_data->inputs_count[2] != 1) {
    return false;
  }
  if (task_data->inputs[1] == nullptr || task_data->inputs[2] == nullptr) {
    return false;
  }

  const int w = *reinterpret_cast<const int*>(task_data->inputs[1]);
  const int h = *reinterpret_cast<const int*>(task_data->inputs[2]);
  if (w <= 0 || h <= 0) {
    return false;
  }
  if (static_cast<unsigned long long>(w) * static_cast<unsigned long long>(h) != n) {
    return false;
  }
  if (task_data->outputs.empty() || task_data->outputs_count.empty()) {
    return false;
  }
  const unsigned int cap = task_data->outputs_count[0];
  return (cap == 0) || (task_data->outputs[0] != nullptr);
}

bool ConvexHullOmp::PreProcessingImpl() {
  input_points_.clear();

  const auto* img = reinterpret_cast<const unsigned char*>(task_data->inputs[0]);
  const int w = *reinterpret_cast<const int*>(task_data->inputs[1]);
  const int h = *reinterpret_cast<const int*>(task_data->inputs[2]);

  const int nthreads = omp_get_max_threads();
  std::vector<std::vector<Point>> bins(static_cast<std::size_t>(nthreads));

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    std::vector<Point> local;
    const unsigned nt = (nthreads > 0) ? static_cast<unsigned>(nthreads) : 1U;
    local.reserve(((static_cast<std::size_t>(w) * static_cast<std::size_t>(h)) / (8U * nt)) + 64U);

#pragma omp for schedule(static)
    for (int y = 0; y < h; ++y) {
      const std::size_t off = static_cast<std::size_t>(y) * static_cast<std::size_t>(w);
      for (int x = 0; x < w; ++x) {
        if (img[off + static_cast<std::size_t>(x)] != 0U) {
          local.push_back(Point{.x = x, .y = y});
        }
      }
    }

    bins[static_cast<std::size_t>(tid)] = std::move(local);
  }  // implicit barrier

  std::size_t total = 0;
  for (const auto& b : bins) {
    total += b.size();
  }
  input_points_.reserve(total);
  for (auto& b : bins) {
    input_points_.insert(input_points_.end(), b.begin(), b.end());
  }
  return true;
}

bool ConvexHullOmp::RunImpl() {
  output_hull_ = BuildHullMonotone(input_points_);
  return true;
}

bool ConvexHullOmp::PostProcessingImpl() {
  auto* out = reinterpret_cast<Point*>(task_data->outputs[0]);
  const unsigned int cap = task_data->outputs_count[0];

  if (out == nullptr || cap == 0) {
    task_data->outputs_count[0] = 0;
    return true;
  }

  const std::size_t n = std::min<std::size_t>(output_hull_.size(), cap);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = output_hull_[i];
  }

  task_data->outputs_count[0] = static_cast<unsigned int>(n);
  return true;
}

}  // namespace shkurinskaya_e_convex_hull_components_omp
