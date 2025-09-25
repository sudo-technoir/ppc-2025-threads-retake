#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/concurrent_vector.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_sort.h"

namespace shkurinskaya_e_convex_hull_components_tbb {

struct Point {
  int x = 0;
  int y = 0;
};

class ConvexHullTbb : public ppc::core::Task {
 public:
  explicit ConvexHullTbb(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<Point> input_points_;
  std::vector<Point> output_hull_;

  static int Cross(const Point& o, const Point& a, const Point& b) noexcept;
};

}  // namespace shkurinskaya_e_convex_hull_components_tbb
