#include "seq/shkurinskaya_e_convex_hull_components/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

using namespace shkurinskaya_e_convex_hull_components_seq;

namespace {

inline long long TwiceOrientedArea(const Point& a, const Point& b, const Point& c) noexcept {
  const long long abx = static_cast<long long>(b.x) - static_cast<long long>(a.x);
  const long long aby = static_cast<long long>(b.y) - static_cast<long long>(a.y);
  const long long acx = static_cast<long long>(c.x) - static_cast<long long>(a.x);
  const long long acy = static_cast<long long>(c.y) - static_cast<long long>(a.y);
  return (abx * acy) - (aby * acx);
}

inline bool LexiLess(const Point& a, const Point& b) noexcept { return (a.x < b.x) || (a.x == b.x && a.y < b.y); }

inline void DedupSortedInPlace(std::vector<Point>& pts) {
  pts.erase(
      std::unique(pts.begin(), pts.end(), [](const Point& l, const Point& r) { return l.x == r.x && l.y == r.y; }),
      pts.end());
}

inline void AppendWithLeftTurn(std::vector<Point>& chain, const Point& nxt) {
  while (chain.size() >= 2 && TwiceOrientedArea(chain[chain.size() - 2], chain.back(), nxt) <= 0) {
    chain.pop_back();
  }
  chain.push_back(nxt);
}

inline std::vector<Point> BuildHullMonotone(std::vector<Point> pts) {
  if (pts.size() <= 1) return pts;

  std::sort(pts.begin(), pts.end(), LexiLess);
  DedupSortedInPlace(pts);
  if (pts.size() <= 1) return pts;

  std::vector<Point> lower, upper;
  lower.reserve(pts.size());
  upper.reserve(pts.size());

  for (const auto& p : pts) AppendWithLeftTurn(lower, p);
  for (auto it = pts.rbegin(); it != pts.rend(); ++it) AppendWithLeftTurn(upper, *it);

  if (!lower.empty()) lower.pop_back();
  if (!upper.empty()) upper.pop_back();

  std::vector<Point> hull;
  hull.reserve(lower.size() + upper.size());
  hull.insert(hull.end(), lower.begin(), lower.end());
  hull.insert(hull.end(), upper.begin(), upper.end());
  return hull;
}

}  // namespace

namespace shkurinskaya_e_convex_hull_components_seq {

int ConvexHullSequential::Cross(const Point& o, const Point& a, const Point& b) noexcept {
  const long long v = TwiceOrientedArea(o, a, b);
  return (v > 0) - (v < 0);
}

bool ConvexHullSequential::ValidationImpl() {
  if (!task_data) return false;

  if (task_data->inputs.size() < 3 || task_data->inputs_count.size() < 3) return false;

  const unsigned int n = task_data->inputs_count[0];
  if (n > 0 && task_data->inputs[0] == nullptr) return false;

  if (task_data->inputs_count[1] != 1 || task_data->inputs_count[2] != 1) return false;
  if (task_data->inputs[1] == nullptr || task_data->inputs[2] == nullptr) return false;

  const int w = *reinterpret_cast<const int*>(task_data->inputs[1]);
  const int h = *reinterpret_cast<const int*>(task_data->inputs[2]);
  if (w <= 0 || h <= 0) return false;
  if (static_cast<unsigned long long>(w) * static_cast<unsigned long long>(h) != n) return false;

  if (task_data->outputs.size() < 1 || task_data->outputs_count.size() < 1) return false;
  const unsigned int cap = task_data->outputs_count[0];
  if (cap > 0 && task_data->outputs[0] == nullptr) return false;

  return true;
}

bool ConvexHullSequential::PreProcessingImpl() {
  input_points_.clear();

  const auto* img = reinterpret_cast<const unsigned char*>(task_data->inputs[0]);
  const int w = *reinterpret_cast<const int*>(task_data->inputs[1]);
  const int h = *reinterpret_cast<const int*>(task_data->inputs[2]);

  input_points_.reserve(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) / 8 + 64);

  for (int y = 0; y < h; ++y) {
    const std::size_t off = static_cast<std::size_t>(y) * static_cast<std::size_t>(w);
    for (int x = 0; x < w; ++x) {
      if (img[off + static_cast<std::size_t>(x)] != 0) {
        input_points_.push_back({x, y});
      }
    }
  }
  return true;
}

bool ConvexHullSequential::RunImpl() {
  output_hull_ = BuildHullMonotone(input_points_);
  return true;
}

bool ConvexHullSequential::PostProcessingImpl() {
  auto* out = reinterpret_cast<Point*>(task_data->outputs[0]);
  const unsigned int cap = task_data->outputs_count[0];

  if (!out || cap == 0) {
    task_data->outputs_count[0] = 0;
    return true;
  }

  const std::size_t n = std::min<std::size_t>(output_hull_.size(), cap);
  for (std::size_t i = 0; i < n; ++i) out[i] = output_hull_[i];

  task_data->outputs_count[0] = static_cast<unsigned int>(n);
  return true;
}

}  // namespace shkurinskaya_e_convex_hull_components_seq
