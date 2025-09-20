#include "seq/shkurinskaya_e_convex_hull_components/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/task/include/task.hpp"

using namespace shkurinskaya_e_convex_hull_components_seq;

namespace {

static inline long long twice_oriented_area(const Point& a, const Point& b, const Point& c) noexcept {
  return 1LL * (b.x - a.x) * (c.y - a.y) - 1LL * (b.y - a.y) * (c.x - a.x);
}

static inline bool lexi_less(const Point& A, const Point& B) noexcept {
  return (A.x < B.x) || (A.x == B.x && A.y < B.y);
}

static inline void dedup_sorted_inplace(std::vector<Point>& pts) {
  pts.erase(
      std::unique(pts.begin(), pts.end(), [](const Point& L, const Point& R) { return L.x == R.x && L.y == R.y; }),
      pts.end());
}

static inline void append_with_left_turn(std::vector<Point>& chain, const Point& nxt) {
  while (chain.size() >= 2 && twice_oriented_area(chain[chain.size() - 2], chain.back(), nxt) <= 0) {
    chain.pop_back();
  }
  chain.push_back(nxt);
}

static std::vector<Point> build_hull_monotone(std::vector<Point> pts) {
  if (pts.size() <= 1) return pts;

  std::sort(pts.begin(), pts.end(), lexi_less);
  dedup_sorted_inplace(pts);
  if (pts.size() <= 1) return pts;

  std::vector<Point> lower_chain, upper_chain;
  lower_chain.reserve(pts.size());
  upper_chain.reserve(pts.size());

  for (const auto& p : pts) append_with_left_turn(lower_chain, p);
  for (auto it = pts.rbegin(); it != pts.rend(); ++it) append_with_left_turn(upper_chain, *it);

  if (!lower_chain.empty()) lower_chain.pop_back();
  if (!upper_chain.empty()) upper_chain.pop_back();

  std::vector<Point> hull;
  hull.reserve(lower_chain.size() + upper_chain.size());
  hull.insert(hull.end(), lower_chain.begin(), lower_chain.end());
  hull.insert(hull.end(), upper_chain.begin(), upper_chain.end());
  return hull;
}

}  // namespace

namespace shkurinskaya_e_convex_hull_components_seq {

int ConvexHullSequential::Cross(const Point& o, const Point& a, const Point& b) noexcept {
  const long long v = twice_oriented_area(o, a, b);
  return (v > 0) - (v < 0);
}

bool ConvexHullSequential::ValidationImpl() {
  if (!task_data) return false;

  if (task_data->inputs.size() < 3 || task_data->inputs_count.size() < 3) return false;

  const unsigned int n = task_data->inputs_count[0];
  if (n > 0 && task_data->inputs[0] == nullptr) return false;

  if (task_data->inputs_count[1] != 1 || task_data->inputs_count[2] != 1) return false;
  if (task_data->inputs[1] == nullptr || task_data->inputs[2] == nullptr) return false;

  const int W = *reinterpret_cast<const int*>(task_data->inputs[1]);
  const int H = *reinterpret_cast<const int*>(task_data->inputs[2]);
  if (W <= 0 || H <= 0) return false;
  if (static_cast<unsigned long long>(W) * static_cast<unsigned long long>(H) != n) return false;

  if (task_data->outputs.size() < 1 || task_data->outputs_count.size() < 1) return false;
  const unsigned int cap = task_data->outputs_count[0];
  if (cap > 0 && task_data->outputs[0] == nullptr) return false;

  return true;
}

bool ConvexHullSequential::PreProcessingImpl() {
  input_points_.clear();

  const auto* img = reinterpret_cast<const unsigned char*>(task_data->inputs[0]);
  const int W = *reinterpret_cast<const int*>(task_data->inputs[1]);
  const int H = *reinterpret_cast<const int*>(task_data->inputs[2]);

  input_points_.reserve(static_cast<size_t>(W) * static_cast<size_t>(H) / 8 + 64);

  for (int y = 0; y < H; ++y) {
    const size_t off = static_cast<size_t>(y) * static_cast<size_t>(W);
    for (int x = 0; x < W; ++x) {
      if (img[off + static_cast<size_t>(x)] != 0) input_points_.push_back({x, y});
    }
  }
  return true;
}

bool ConvexHullSequential::RunImpl() {
  output_hull_ = build_hull_monotone(input_points_);
  return true;
}

bool ConvexHullSequential::PostProcessingImpl() {
  auto* out = reinterpret_cast<Point*>(task_data->outputs[0]);
  const unsigned int cap = task_data->outputs_count[0];
  if (!out || cap == 0) {
    task_data->outputs_count[0] = 0;
    return true;
  }

  const size_t n = std::min<size_t>(output_hull_.size(), cap);
  for (size_t i = 0; i < n; ++i) out[i] = output_hull_[i];

  task_data->outputs_count[0] = static_cast<unsigned int>(n);
  // for (size_t i = n; i < cap; ++i) out[i] = Point{0,0};

  return true;
}

}  // namespace shkurinskaya_e_convex_hull_components_seq
