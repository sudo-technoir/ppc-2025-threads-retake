#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shkurinskaya_e_convex_hull_components_seq {

struct Point {
  int x = 0;
  int y = 0;
};

class ConvexHullSequential : public ppc::core::Task {
 public:
  explicit ConvexHullSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<Point> input_points_;
  std::vector<Point> output_hull_;

  static int Cross(const Point& o, const Point& a, const Point& b) noexcept;
};

}  // namespace shkurinskaya_e_convex_hull_components_seq
