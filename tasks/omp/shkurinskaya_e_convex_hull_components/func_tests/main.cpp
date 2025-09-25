#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/shkurinskaya_e_convex_hull_components/include/ops_omp.hpp"

using shkurinskaya_e_convex_hull_components_omp::ConvexHullOmp;
using shkurinskaya_e_convex_hull_components_omp::Point;

namespace {
std::vector<Point> RunHull(const std::vector<uint8_t>& img, int w, int h) {
  std::vector<Point> out(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));

  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.emplace_back(const_cast<uint8_t*>(img.data()));
  td->inputs_count.emplace_back(static_cast<unsigned int>(img.size()));
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(&w));
  td->inputs_count.emplace_back(1);
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(&h));
  td->inputs_count.emplace_back(1);
  td->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  td->outputs_count.emplace_back(static_cast<unsigned int>(out.size()));

  auto task = std::make_shared<ConvexHullOmp>(td);

  if (!task->ValidationImpl()) {
    ADD_FAILURE() << "ValidationImpl() returned false";
    return {};
  }
  if (!task->PreProcessingImpl()) {
    ADD_FAILURE() << "PreProcessingImpl() returned false";
    return {};
  }
  if (!task->RunImpl()) {
    ADD_FAILURE() << "RunImpl() returned false";
    return {};
  }
  if (!task->PostProcessingImpl()) {
    ADD_FAILURE() << "PostProcessingImpl() returned false";
    return {};
  }

  const unsigned int n = td->outputs_count[0];
  EXPECT_LE(n, out.size());

  return {out.begin(), out.begin() + n};
}
}  // namespace

TEST(shkurinskaya_e_convex_hull_components_omp, hull_on_solid_square_5x5) {
  const int w = 5;
  const int h = 5;
  std::vector<uint8_t> img(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
  auto set1 = [&](int x, int y) {
    const std::size_t idx = (static_cast<std::size_t>(y) * static_cast<std::size_t>(w)) + static_cast<std::size_t>(x);
    img[idx] = 1;
  };
  for (int y = 1; y <= 3; ++y) {
    for (int x = 1; x <= 3; ++x) {
      set1(x, y);
    }
  }

  auto hull = RunHull(img, w, h);

  const auto contains = [&](Point q) {
    return std::ranges::find_if(hull, [&](const Point& p) { return p.x == q.x && p.y == q.y; }) != hull.end();
  };
  EXPECT_TRUE(contains({1, 1}));
  EXPECT_TRUE(contains({3, 1}));
  EXPECT_TRUE(contains({3, 3}));
  EXPECT_TRUE(contains({1, 3}));
  EXPECT_EQ(hull.size(), 4U);
}

TEST(shkurinskaya_e_convex_hull_components_omp, hull_on_perfect_diagonal_collinear) {
  const int w = 7;
  const int h = 7;
  std::vector<uint8_t> img(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
  for (int i = 0; i < std::min(w, h); ++i) {
    const std::size_t idx = (static_cast<std::size_t>(i) * static_cast<std::size_t>(w)) + static_cast<std::size_t>(i);
    img[idx] = 1;
  }

  auto hull = RunHull(img, w, h);

  const auto contains = [&](Point q) {
    return std::ranges::find_if(hull, [&](const Point& p) { return p.x == q.x && p.y == q.y; }) != hull.end();
  };
  EXPECT_TRUE(contains({0, 0}));
  EXPECT_TRUE(contains({w - 1, h - 1}));

  const auto is_inner_diag = [&](const Point& p) { return p.x == p.y && p.x > 0 && p.x < (w - 1); };
  EXPECT_TRUE(std::ranges::none_of(hull, is_inner_diag));
  EXPECT_EQ(hull.size(), 2U);
}
