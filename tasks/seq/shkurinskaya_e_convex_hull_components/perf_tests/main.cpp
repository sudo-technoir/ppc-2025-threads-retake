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
constexpr int kW = 65536;
constexpr int kH = 65536;

const std::vector<uint8_t>& SharedImage() {
  static std::vector<uint8_t> img;
  static bool inited = false;
  if (!inited) {
    img.assign(static_cast<std::size_t>(kW) * static_cast<std::size_t>(kH), 0);
    // верх/низ
    for (int x = 0; x < kW; ++x) {
      img[(static_cast<std::size_t>(0) * static_cast<std::size_t>(kW)) + static_cast<std::size_t>(x)] = 1;
      img[(static_cast<std::size_t>(kH - 1) * static_cast<std::size_t>(kW)) + static_cast<std::size_t>(x)] = 1;
    }
    // лево/право
    for (int y = 0; y < kH; ++y) {
      img[(static_cast<std::size_t>(y) * static_cast<std::size_t>(kW)) + static_cast<std::size_t>(0)] = 1;
      img[(static_cast<std::size_t>(y) * static_cast<std::size_t>(kW)) + static_cast<std::size_t>(kW - 1)] = 1;
    }
    inited = true;
  }
  return img;
}

inline double NowSec() {
  using Clock = std::chrono::high_resolution_clock;
  static const auto kT0 = Clock::now();
  return std::chrono::duration<double>(Clock::now() - kT0).count();
}

struct ImgSpec {
  int w;
  int h;
};

std::shared_ptr<ppc::core::PerfAttr> MakePerfAttr(int runs) {
  auto a = std::make_shared<ppc::core::PerfAttr>();
  a->num_running = runs;
  a->current_timer = [] { return NowSec(); };
  return a;
}

std::shared_ptr<ppc::core::TaskData> MakeTaskData(const std::vector<uint8_t>& img,
                                                  ImgSpec& spec,  // NOLINT(modernize-pass-by-value)
                                                  std::vector<Point>& out) {
  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.emplace_back(const_cast<uint8_t*>(img.data()));
  td->inputs_count.emplace_back(static_cast<unsigned int>(img.size()));
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(&spec.w));
  td->inputs_count.emplace_back(1);
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(&spec.h));
  td->inputs_count.emplace_back(1);

  td->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  td->outputs_count.emplace_back(static_cast<unsigned int>(out.size()));
  return td;
}

bool HasPoint(const Point* out, unsigned n, Point q) {
  for (unsigned i = 0; i < n; ++i) {
    if (out[i].x == q.x && out[i].y == q.y) {
      return true;
    }
  }
  return false;
}
}  // namespace

TEST(shkurinskaya_e_convex_hull_components_seq, perf_pipeline_shared_frame) {
  const auto& img = SharedImage();
  std::vector<Point> out(img.size());
  ImgSpec spec{.w = kW, .h = kH};

  auto td = MakeTaskData(img, spec, out);
  auto task = std::make_shared<ConvexHullSequential>(td);

  auto perf_attr = MakePerfAttr(10);
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);

  perf->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  const unsigned n = td->outputs_count[0];
  ASSERT_LE(n, out.size());
  EXPECT_GE(n, 4U);

  EXPECT_TRUE(HasPoint(out.data(), n, Point{0, 0}));
  EXPECT_TRUE(HasPoint(out.data(), n, Point{kW - 1, 0}));
  EXPECT_TRUE(HasPoint(out.data(), n, Point{kW - 1, kH - 1}));
  EXPECT_TRUE(HasPoint(out.data(), n, Point{0, kH - 1}));
}

TEST(shkurinskaya_e_convex_hull_components_seq, perf_taskrun_shared_frame) {
  const auto& img = SharedImage();
  std::vector<Point> out(img.size());
  ImgSpec spec{.w = kW, .h = kH};

  auto td = MakeTaskData(img, spec, out);
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
  EXPECT_GE(n, 4U);
  EXPECT_TRUE(HasPoint(out.data(), n, Point{0, 0}));
  EXPECT_TRUE(HasPoint(out.data(), n, Point{kW - 1, kH - 1}));
}
