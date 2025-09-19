#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/shkurinskaya_e_convex_hull_components/include/ops_seq.hpp"

using shkurinskaya_e_convex_hull_components_seq::ConvexHullSequential;
using shkurinskaya_e_convex_hull_components_seq::Point;

namespace {
static std::vector<Point> RunHull(const std::vector<uint8_t>& img, int W, int H) {
  std::vector<Point> out(static_cast<size_t>(W) * static_cast<size_t>(H));

  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<uint8_t*>(img.data())));
  td->inputs_count.emplace_back(static_cast<unsigned int>(img.size()));
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&W)));
  td->inputs_count.emplace_back(1);
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&H)));
  td->inputs_count.emplace_back(1);
  td->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  td->outputs_count.emplace_back(static_cast<unsigned int>(out.size()));

  auto task = std::make_shared<ConvexHullSequential>(td);

  EXPECT_TRUE(task->ValidationImpl());
  ASSERT_TRUE(task->PreProcessingImpl());
  ASSERT_TRUE(task->RunImpl());
  ASSERT_TRUE(task->PostProcessingImpl());

  const unsigned int out_len = td->outputs_count[0];
  EXPECT_LE(out_len, out.size());

  return std::vector<Point>(out.begin(), out.begin() + out_len);
}
}  // namespace

TEST(shkurinskaya_e_convex_hull_components_seq, hull_on_solid_square_5x5) {
  const int W = 5, H = 5;
  std::vector<uint8_t> img(static_cast<size_t>(W) * static_cast<size_t>(H), 0);
  auto set1 = [&](int x, int y) { img[static_cast<size_t>(y) * W + x] = 1; };
  for (int y = 1; y <= 3; ++y)
    for (int x = 1; x <= 3; ++x) set1(x, y);

  auto hull = RunHull(img, W, H);

  auto contains = [&](Point q) {
    return std::find_if(hull.begin(), hull.end(), [&](const Point& p) { return p.x == q.x && p.y == q.y; }) !=
           hull.end();
  };
  EXPECT_TRUE(contains({1, 1}));
  EXPECT_TRUE(contains({3, 1}));
  EXPECT_TRUE(contains({3, 3}));
  EXPECT_TRUE(contains({1, 3}));
  EXPECT_EQ(hull.size(), 4u);
}

TEST(shkurinskaya_e_convex_hull_components_seq, hull_on_perfect_diagonal_collinear) {
  const int W = 7, H = 7;
  std::vector<uint8_t> img(static_cast<size_t>(W) * static_cast<size_t>(H), 0);
  for (int i = 0; i < std::min(W, H); ++i) img[static_cast<size_t>(i) * W + i] = 1;

  auto hull = RunHull(img, W, H);

  auto contains = [&](Point q) {
    return std::find_if(hull.begin(), hull.end(), [&](const Point& p) { return p.x == q.x && p.y == q.y; }) !=
           hull.end();
  };
  EXPECT_TRUE(contains({0, 0}));
  EXPECT_TRUE(contains({W - 1, H - 1}));

  auto is_inner_diag = [&](const Point& p) { return p.x == p.y && p.x > 0 && p.x < W - 1; };
  EXPECT_TRUE(std::none_of(hull.begin(), hull.end(), is_inner_diag));

  EXPECT_EQ(hull.size(), 2u);
}
