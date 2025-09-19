// perf_tests/main.cpp 
#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/shkurinskaya_e_convex_hull_components/include/ops_seq.hpp"

using shkurinskaya_e_convex_hull_components_seq::ConvexHullSequential;
using shkurinskaya_e_convex_hull_components_seq::Point;

namespace {
constexpr int kW = 4096;
constexpr int kH = 4096;
const std::vector<uint8_t>& SharedImage() {
  static std::vector<uint8_t> img;
  static bool inited = false;
  if (!inited) {
    img.assign(static_cast<size_t>(kW) * static_cast<size_t>(kH), 0);
    // верх/низ
    for (int x = 0; x < kW; ++x) {
      img[static_cast<size_t>(0) * kW + x] = 1;
      img[static_cast<size_t>(kH - 1) * kW + x] = 1;
    }
    // лево/право
    for (int y = 0; y < kH; ++y) {
      img[static_cast<size_t>(y) * kW + 0] = 1;
      img[static_cast<size_t>(y) * kW + (kW - 1)] = 1;
    }
    inited = true;
  }
  return img;
}

inline double NowSec() {
  using clock = std::chrono::high_resolution_clock;
  static const auto t0 = clock::now();
  return std::chrono::duration<double>(clock::now() - t0).count();
}

std::shared_ptr<ppc::core::PerfAttr> MakePerfAttr(int runs) {
  auto a = std::make_shared<ppc::core::PerfAttr>();
  a->num_running = runs;
  a->current_timer = [] { return NowSec(); };
  return a;
}

// Собираем TaskData под (img, W, H) + выходной буфер
std::shared_ptr<ppc::core::TaskData> MakeTaskData(const std::vector<uint8_t>& img, int W, int H,
                                                  std::vector<Point>& out) {
  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<uint8_t*>(img.data())));
  td->inputs_count.emplace_back(static_cast<unsigned int>(img.size()));
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&W)));
  td->inputs_count.emplace_back(1);
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&H)));
  td->inputs_count.emplace_back(1);

  td->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  td->outputs_count.emplace_back(static_cast<unsigned int>(out.size()));
  return td;
}

bool HasPoint(const Point* out, unsigned n, Point q) {
  for (unsigned i = 0; i < n; ++i)
    if (out[i].x == q.x && out[i].y == q.y) return true;
  return false;
}
}  // namespace

TEST(shkurinskaya_e_convex_hull_components_seq, perf_pipeline_shared_frame) {
  const auto& img = SharedImage();
  std::vector<Point> out(img.size());
  auto td = MakeTaskData(img, kW, kH, out);
  auto task = std::make_shared<ConvexHullSequential>(td);

  auto perf_attr = MakePerfAttr(10);
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);

  perf->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  const unsigned n = td->outputs_count[0];
  ASSERT_LE(n, out.size());
  EXPECT_GE(n, 4u);

  EXPECT_TRUE(HasPoint(out.data(), n, Point{0, 0}));
  EXPECT_TRUE(HasPoint(out.data(), n, Point{kW - 1, 0}));
  EXPECT_TRUE(HasPoint(out.data(), n, Point{kW - 1, kH - 1}));
  EXPECT_TRUE(HasPoint(out.data(), n, Point{0, kH - 1}));
}

TEST(shkurinskaya_e_convex_hull_components_seq, perf_taskrun_shared_frame) {
  const auto& img = SharedImage();
  std::vector<Point> out(img.size());

  auto td = MakeTaskData(img, kW, kH, out);
  auto task = std::make_shared<ConvexHullSequential>(td);

  auto perf_attr = MakePerfAttr(10);
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);

  ASSERT_TRUE(task->ValidationImpl());
  ASSERT_TRUE(task->PreProcessingImpl());

  perf->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  const unsigned n = td->outputs_count[0];
  ASSERT_LE(n, out.size());
  EXPECT_GE(n, 4u);
  EXPECT_TRUE(HasPoint(out.data(), n, Point{0, 0}));
  EXPECT_TRUE(HasPoint(out.data(), n, Point{kW - 1, kH - 1}));
}
