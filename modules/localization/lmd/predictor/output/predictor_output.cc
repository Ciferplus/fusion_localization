/******************************************************************************
 * Copyright 2018 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#include "modules/localization/lmd/predictor/output/predictor_output.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "modules/common/proto/geometry.pb.h"

#include "modules/common/log.h"
#include "modules/common/math/euler_angles_zxy.h"
#include "modules/common/math/linear_interpolation.h"
#include "modules/common/math/math_utils.h"
#include "modules/common/math/quaternion.h"
#include "modules/localization/common/localization_gflags.h"

namespace apollo {
namespace localization {

using apollo::common::PathPoint;
using apollo::common::Point3D;
using apollo::common::PointENU;
using apollo::common::Quaternion;
using apollo::common::Status;
using apollo::common::math::EulerAnglesZXYd;
using apollo::common::math::HeadingToQuaternion;
using apollo::common::math::InterpolateUsingLinearApproximation;
using apollo::common::math::NormalizeAngle;
using apollo::common::math::QuaternionRotate;
using apollo::common::math::QuaternionToHeading;
using apollo::common::math::RotateAxis;

namespace {
template <class T>
T QuaternionRotateXYZ(const T v, const Quaternion& orientation) {
  auto vec =
      QuaternionRotate(orientation, Eigen::Vector3d(v.x(), v.y(), v.z()));
  T ret;
  ret.set_x(vec[0]);
  ret.set_y(vec[1]);
  ret.set_z(vec[2]);
  return ret;
}

PathPoint FindProjectionPoint(const PathPoint& p0, const PathPoint& p1,
                              const double x, const double y) {
  double v0x = x - p0.x();
  double v0y = y - p0.y();

  double v1x = p1.x() - p0.x();
  double v1y = p1.y() - p0.y();

  double v1_norm = std::sqrt(v1x * v1x + v1y * v1y);
  double dot = v0x * v1x + v0y * v1y;

  double delta_s = dot / v1_norm;
  return InterpolateUsingLinearApproximation(p0, p1, p0.s() + delta_s);
}

void FillPoseFromImu(const Pose& imu_pose, Pose* pose) {
  // linear acceleration
  if (imu_pose.has_linear_acceleration()) {
    if (FLAGS_enable_map_reference_unify) {
      if (pose->has_orientation()) {
        pose->mutable_linear_acceleration()->CopyFrom(QuaternionRotateXYZ(
            imu_pose.linear_acceleration(), pose->orientation()));
        pose->mutable_linear_acceleration_vrf()->CopyFrom(
            imu_pose.linear_acceleration());
      } else {
        AERROR << "Fail to convert linear_acceleration";
      }
    } else {
      pose->mutable_linear_acceleration()->CopyFrom(
          imu_pose.linear_acceleration());
    }
  }

  // angular velocity
  if (imu_pose.has_angular_velocity()) {
    if (FLAGS_enable_map_reference_unify) {
      if (FLAGS_enable_map_reference_unify) {
        if (pose->has_orientation()) {
          pose->mutable_angular_velocity()->CopyFrom(QuaternionRotateXYZ(
              imu_pose.angular_velocity(), pose->orientation()));
          pose->mutable_angular_velocity_vrf()->CopyFrom(
              imu_pose.angular_velocity());
        } else {
          AERROR << "Fail to convert angular_velocity";
        }
      } else {
        pose->mutable_angular_velocity()->CopyFrom(imu_pose.angular_velocity());
      }
    } else {
      pose->mutable_angular_velocity()->CopyFrom(imu_pose.angular_velocity());
    }
  }

  // euler angle
  if (imu_pose.has_euler_angles()) {
    pose->mutable_euler_angles()->CopyFrom(imu_pose.euler_angles());
  }
}
}  // namespace

PredictorOutput::PredictorOutput(
    double memory_cycle_sec,
    const std::function<Status(double, const Pose&)>& publish_loc_func)
    : Predictor(memory_cycle_sec),
      publish_loc_func_(publish_loc_func),
      pc_map_(FLAGS_enable_lmd_premapping ? &lm_provider_ : nullptr) {
  name_ = kPredictorOutputName;
  dep_predicteds_.emplace(kPredictorGpsName, PoseList(memory_cycle_sec));
  dep_predicteds_.emplace(kPredictorFilteredImuName,
                          PoseList(memory_cycle_sec));
  dep_predicteds_.emplace(kPredictorPerceptionName, PoseList(memory_cycle_sec));
  on_adapter_thread_ = true;

  constexpr double kTs = 0.01;
  constexpr double kCutoffFreq = 0.5;
  InitializeFilter(kTs, kCutoffFreq);
}

PredictorOutput::~PredictorOutput() {}

bool PredictorOutput::Updateable() const {
  const auto& imu = dep_predicteds_.find(kPredictorFilteredImuName)->second;
  if (predicted_.empty()) {
    const auto& gps = dep_predicteds_.find(kPredictorGpsName)->second;
    return !gps.empty() && !imu.empty();
  } else {
    return !imu.empty() && predicted_.Older(imu);
  }
}

Status PredictorOutput::Update() {
  if (predicted_.empty()) {
    const auto& gps = dep_predicteds_[kPredictorGpsName];
    auto gps_latest = gps.Latest();
    auto timestamp_sec = gps_latest->first;
    auto pose = gps_latest->second;

    if (!pose.has_heading() && pose.has_orientation()) {
      pose.set_heading(QuaternionToHeading(
          pose.orientation().qw(), pose.orientation().qx(),
          pose.orientation().qy(), pose.orientation().qz()));
    }

    const auto& imu = dep_predicteds_[kPredictorFilteredImuName];
    Pose imu_pose;
    imu.FindNearestPose(timestamp_sec, &imu_pose);

    // fill pose from imu
    FillPoseFromImu(imu_pose, &pose);

    // push pose to list
    predicted_.Push(timestamp_sec, pose);

    // publish
    return publish_loc_func_(timestamp_sec, pose);
  } else {
    // get timestamp from imu
    const auto& imu = dep_predicteds_[kPredictorFilteredImuName];
    auto timestamp_sec = imu.Latest()->first;

    // base pose for prediction
    double base_timestamp_sec;
    Pose base_pose;
    const auto& perception = dep_predicteds_[kPredictorPerceptionName];
    auto perception_pose_it = perception.RangeOf(timestamp_sec).first;
    if (perception_pose_it != perception.end()) {
      base_timestamp_sec = perception_pose_it->first;
      if (!predicted_.Older(base_timestamp_sec)) {
        predicted_.FindNearestPose(base_timestamp_sec, &base_pose);
        // assign position and heading from perception
        const auto& perception_pose = perception_pose_it->second;
        base_pose.mutable_position()->CopyFrom(perception_pose.position());
        base_pose.mutable_orientation()->CopyFrom(
            perception_pose.orientation());
        base_pose.set_heading(perception_pose.heading());

      } else {
        base_timestamp_sec = predicted_.Latest()->first;
        base_pose = predicted_.Latest()->second;
      }
    } else {
      base_timestamp_sec = predicted_.Latest()->first;
      base_pose = predicted_.Latest()->second;
    }

    // predict
    Pose pose;
    PredictByImu(base_timestamp_sec, base_pose, timestamp_sec, &pose);

    // push pose to list
    predicted_.Push(timestamp_sec, pose);

    if (FLAGS_enable_position_adjust_by_lane) {
      AdjustPositionInLane();
      predicted_.FindNearestPose(timestamp_sec, &pose);
    }
    // publish
    return publish_loc_func_(timestamp_sec, pose);
  }
}

bool PredictorOutput::UpdateLaneMarkers(
    double timestamp_sec, const apollo::perception::LaneMarkers& lane_markers) {
  lane_markers_.CopyFrom(lane_markers);
  lane_markers_time_ = timestamp_sec;
  return true;
}

bool PredictorOutput::PredictByImu(double old_timestamp_sec,
                                   const Pose& old_pose,
                                   double new_timestamp_sec, Pose* new_pose) {
  if (!old_pose.has_position() || !old_pose.has_orientation() ||
      !old_pose.has_linear_velocity()) {
    AERROR << "Pose has no some fields";
    return false;
  }

  const auto& imu = dep_predicteds_[kPredictorFilteredImuName];
  auto p = imu.RangeOf(old_timestamp_sec);
  auto it = p.first;
  auto it_1 = p.second;
  if (it == imu.end() && it_1 == imu.end()) {
    AERROR << std::setprecision(15)
           << "Cannot get the lower of range from imu with timestamp["
           << old_timestamp_sec << "]";
    return false;
  }

  auto timestamp_sec = old_timestamp_sec;
  new_pose->CopyFrom(old_pose);
  bool finished = false;
  while (!finished) {
    Pose imu_pose;
    double timestamp_sec_1;
    Pose imu_pose_1;

    if (it == imu.end()) {
      imu_pose = imu_pose_1 = it_1->second;
      timestamp_sec_1 = std::min(it_1->first, new_timestamp_sec);
    } else if (it_1 == imu.end()) {
      imu_pose = imu_pose_1 = it->second;
      timestamp_sec_1 = new_timestamp_sec;
    } else {
      PoseList::InterpolatePose(it->first, it->second, it_1->first,
                                it_1->second, timestamp_sec, &imu_pose);
      timestamp_sec_1 = std::min(it_1->first, new_timestamp_sec);
      PoseList::InterpolatePose(it->first, it->second, it_1->first,
                                it_1->second, timestamp_sec_1, &imu_pose_1);
    }

    if (new_timestamp_sec <= timestamp_sec_1) {
      finished = true;
    }

    if (!finished && timestamp_sec_1 <= timestamp_sec) {
      it = it_1;
      it_1++;
      continue;
    }

    if (!imu_pose.has_linear_acceleration() ||
        !imu_pose_1.has_linear_acceleration() ||
        !imu_pose.has_angular_velocity() ||
        !imu_pose_1.has_angular_velocity()) {
      AERROR << "Imu_pose or imu_pose_1 has no some fields";
      return false;
    }

    auto dt = timestamp_sec_1 - timestamp_sec;
    auto orientation = new_pose->orientation();

    Point3D angular_velocity;
    if (FLAGS_enable_map_reference_unify) {
      angular_velocity.CopyFrom(
          QuaternionRotateXYZ(imu_pose.angular_velocity(), orientation));
    } else {
      angular_velocity.CopyFrom(imu_pose.angular_velocity());
    }
    Point3D angular_velocity_1;
    if (FLAGS_enable_map_reference_unify) {
      angular_velocity_1.CopyFrom(
          QuaternionRotateXYZ(imu_pose_1.angular_velocity(), orientation));
    } else {
      angular_velocity_1.CopyFrom(imu_pose_1.angular_velocity());
    }

    Point3D angular_vel;
    angular_vel.set_x((angular_velocity.x() + angular_velocity_1.x()) / 2.0);
    angular_vel.set_y((angular_velocity.y() + angular_velocity_1.y()) / 2.0);
    angular_vel.set_z((angular_velocity.z() + angular_velocity_1.z()) / 2.0);

    EulerAnglesZXYd euler_c;
    if (FLAGS_enable_gps_heading) {
      const auto& gps = dep_predicteds_[kPredictorGpsName];
      auto gps_latest = gps.Latest();
      auto gps_pose = gps_latest->second;

      if (!gps_pose.has_orientation()) {
        gps_pose.CopyFrom(old_pose);
      }
      EulerAnglesZXYd euler_b(
          gps_pose.orientation().qw(), gps_pose.orientation().qx(),
          gps_pose.orientation().qy(), gps_pose.orientation().qz());
      euler_c = euler_b;

    } else {
      EulerAnglesZXYd euler_a(orientation.qw(), orientation.qx(),
                              orientation.qy(), orientation.qz());

      auto derivation_roll = angular_vel.x() +
                             std::sin(euler_a.roll()) *
                                 std::tan(euler_a.pitch()) * angular_vel.y() +
                             std::cos(euler_a.roll()) *
                                 std::tan(euler_a.pitch()) * angular_vel.z();

      auto derivation_pitch = std::cos(euler_a.roll()) * angular_vel.y() -
                              std::sin(euler_a.roll()) * angular_vel.z();

      auto derivation_yaw = std::sin(euler_a.roll()) /
                                std::cos(euler_a.pitch()) * angular_vel.y() +
                            std::cos(euler_a.roll()) /
                                std::cos(euler_a.pitch()) * angular_vel.z();

      EulerAnglesZXYd euler_b(euler_a.roll() + derivation_roll * dt,
                              euler_a.pitch() + derivation_pitch * dt,
                              euler_a.yaw() + derivation_yaw * dt);

      euler_c = euler_b;
    }

    auto q = euler_c.ToQuaternion();
    Quaternion orientation_1;
    orientation_1.set_qw(q.w());
    orientation_1.set_qx(q.x());
    orientation_1.set_qy(q.y());
    orientation_1.set_qz(q.z());

    if (FLAGS_enable_heading_filter) {
      auto heading = NormalizeAngle(
          QuaternionToHeading(orientation_1.qw(), orientation_1.qx(),
                              orientation_1.qy(), orientation_1.qz()));

      auto filtered_heading = NormalizeAngle(digital_filter_.Filter(heading));

      auto orientation_2 = HeadingToQuaternion(filtered_heading);
      orientation_1.set_qw(orientation_2.w());
      orientation_1.set_qx(orientation_2.x());
      orientation_1.set_qy(orientation_2.y());
      orientation_1.set_qz(orientation_2.z());
    }

    Point3D linear_acceleration;
    if (FLAGS_enable_map_reference_unify) {
      linear_acceleration.CopyFrom(
          QuaternionRotateXYZ(imu_pose.linear_acceleration(), orientation));
    } else {
      linear_acceleration.CopyFrom(imu_pose.linear_acceleration());
    }
    Point3D linear_acceleration_1;
    if (FLAGS_enable_map_reference_unify) {
      linear_acceleration_1.CopyFrom(
          QuaternionRotateXYZ(imu_pose_1.linear_acceleration(), orientation_1));
    } else {
      linear_acceleration_1.CopyFrom(imu_pose_1.linear_acceleration());
    }

    auto linear_velocity = new_pose->linear_velocity();
    Point3D linear_velocity_1;
    linear_velocity_1.set_x(
        (linear_acceleration.x() + linear_acceleration_1.x()) / 2.0 * dt +
        linear_velocity.x());
    linear_velocity_1.set_y(
        (linear_acceleration.y() + linear_acceleration_1.y()) / 2.0 * dt +
        linear_velocity.y());
    linear_velocity_1.set_z(
        (linear_acceleration.z() + linear_acceleration_1.z()) / 2.0 * dt +
        linear_velocity.z());

    auto position = new_pose->position();
    PointENU position_1;
    position_1.set_x(
        (linear_acceleration.x() / 3.0 + linear_acceleration_1.x() / 6.0) * dt *
            dt +
        linear_velocity.x() * dt + position.x());
    position_1.set_y(
        (linear_acceleration.y() / 3.0 + linear_acceleration_1.y() / 6.0) * dt *
            dt +
        linear_velocity.y() * dt + position.y());
    position_1.set_z(
        (linear_acceleration.z() / 3.0 + linear_acceleration_1.z() / 6.0) * dt *
            dt +
        linear_velocity.z() * dt + position.z());

    new_pose->mutable_position()->CopyFrom(position_1);
    new_pose->mutable_orientation()->CopyFrom(orientation_1);

    new_pose->set_heading(NormalizeAngle(QuaternionToHeading(
        new_pose->orientation().qw(), new_pose->orientation().qx(),
        new_pose->orientation().qy(), new_pose->orientation().qz())));

    new_pose->mutable_linear_velocity()->CopyFrom(linear_velocity_1);
    FillPoseFromImu(imu_pose_1, new_pose);

    if (!finished) {
      timestamp_sec = timestamp_sec_1;
      it = it_1;
      it_1++;
    }
  }

  return true;
}
void PredictorOutput::InitializeFilter(const double ts,
                                       const double cutoff_freq) {
  // Low pass filter
  std::vector<double> den(3, 0.0);
  std::vector<double> num(3, 0.0);
  common::LpfCoefficients(ts, cutoff_freq, &den, &num);
  digital_filter_.set_coefficients(den, num);
}

bool PredictorOutput::AdjustPositionInLane() {
  auto latest_it = predicted_.Latest();

  double timestamp_sec;
  if (latest_it != predicted_.end()) {
    Pose pose;
    timestamp_sec = latest_it->first;
    pose.CopyFrom(latest_it->second);
    predicted_.Pop();

    PointENU enu_position;
    enu_position.CopyFrom(pose.position());

    double nearest_d2 = 0.0;
    PCMapIndex near_ni = 0, nearest_pi = 0;
    std::tie(near_ni, nearest_pi) =
        pc_map_.GetNearestPointOpt(near_ni, enu_position, &nearest_d2);

    if (nearest_pi != (PCMapIndex)-1) {
      const auto& nearest_p = pc_map_.Point(nearest_pi);

      if (nearest_p.prev == (PCMapIndex)-1 ||
          nearest_p.next == (PCMapIndex)-1) {
        ADEBUG << std::setprecision(15) << "not process position, x["
               << nearest_p.position.x() << "], y[" << nearest_p.position.y()
               << "], z[" << nearest_p.position.z() << "]";
      }

      const auto& prev_p = pc_map_.Point(nearest_p.prev);
      const auto& next_p = pc_map_.Point(nearest_p.next);

      PathPoint point0, point1, otherside_point0, otherside_point1;
      if (nearest_p.prev == (PCMapIndex)-1 &&
          nearest_p.next != (PCMapIndex)-1) {
        point0.set_x(2.0 * nearest_p.position.x() - next_p.position.x());
        point0.set_y(2.0 * nearest_p.position.y() - next_p.position.y());
        point0.set_z(0.0);
        point0.set_s(0.0);
        point1.set_x(next_p.position.x());
        point1.set_y(next_p.position.y());
        point1.set_z(0.0);
        point1.set_s(
            std::sqrt(std::pow(next_p.position.x() - point0.x(), 2.0) +
                      std::pow(next_p.position.y() - point0.y(), 2.0)));

        otherside_point0.set_x(2.0 * nearest_p.direction.x() -
                               next_p.direction.x());
        otherside_point0.set_y(2.0 * nearest_p.direction.y() -
                               next_p.direction.y());
        otherside_point0.set_z(0.0);
        otherside_point0.set_s(0.0);
        otherside_point1.set_x(next_p.direction.x());
        otherside_point1.set_y(next_p.direction.y());
        otherside_point1.set_z(0.0);
        otherside_point1.set_s(std::sqrt(
            std::pow(next_p.direction.x() - otherside_point0.x(), 2.0) +
            std::pow(next_p.direction.y() - otherside_point0.y(), 2.0)));

      } else if (nearest_p.prev != (PCMapIndex)-1 &&
                 nearest_p.next == (PCMapIndex)-1) {
        point0.set_x(prev_p.position.x());
        point0.set_y(prev_p.position.y());
        point0.set_z(0.0);
        point0.set_s(0.0);
        point1.set_x(2.0 * nearest_p.position.x() - prev_p.position.x());
        point1.set_y(2.0 * nearest_p.position.y() - prev_p.position.y());
        point1.set_z(0.0);
        point1.set_s(std::sqrt(std::pow(point1.x() - point0.x(), 2.0) +
                               std::pow(point1.y() - point0.y(), 2.0)));

        otherside_point0.set_x(prev_p.direction.x());
        otherside_point0.set_y(prev_p.direction.y());
        otherside_point0.set_z(0.0);
        otherside_point0.set_s(0.0);
        otherside_point1.set_x(2.0 * nearest_p.direction.x() -
                               prev_p.direction.x());
        otherside_point1.set_y(2.0 * nearest_p.direction.y() -
                               prev_p.direction.y());
        otherside_point1.set_z(0.0);
        otherside_point1.set_s(std::sqrt(
            std::pow(otherside_point1.x() - otherside_point0.x(), 2.0) +
            std::pow(otherside_point1.y() - otherside_point0.y(), 2.0)));
      } else if (nearest_p.prev == (PCMapIndex)-1 &&
                 nearest_p.next == (PCMapIndex)-1) {
        predicted_.Push(timestamp_sec, pose);
        return false;
      } else {
        point0.set_x(prev_p.position.x());
        point0.set_y(prev_p.position.y());
        point0.set_z(0.0);
        point0.set_s(0.0);
        point1.set_x(next_p.position.x());
        point1.set_y(next_p.position.y());
        point1.set_z(0.0);
        point1.set_s(std::sqrt(
            std::pow(next_p.position.x() - prev_p.position.x(), 2.0) +
            std::pow(next_p.position.y() - prev_p.position.y(), 2.0)));

        otherside_point0.set_x(prev_p.direction.x());
        otherside_point0.set_y(prev_p.direction.y());
        otherside_point0.set_z(0.0);
        otherside_point0.set_s(0.0);
        otherside_point1.set_x(next_p.direction.x());
        otherside_point1.set_y(next_p.direction.y());
        otherside_point1.set_z(0.0);
        otherside_point1.set_s(std::sqrt(
            std::pow(next_p.direction.x() - prev_p.direction.x(), 2.0) +
            std::pow(next_p.direction.y() - prev_p.direction.y(), 2.0)));
      }

      PointENU enu_nearest_position, otherside_enu_nearest_position;

      auto project_path_point = FindProjectionPoint(
          point0, point1, enu_position.x(), enu_position.y());

      auto otherside_project_path_point =
          FindProjectionPoint(otherside_point0, otherside_point1,
                              enu_position.x(), enu_position.y());

      enu_nearest_position.set_x(project_path_point.x());
      enu_nearest_position.set_y(project_path_point.y());
      enu_nearest_position.set_z(project_path_point.z());

      otherside_enu_nearest_position.set_x(otherside_project_path_point.x());
      otherside_enu_nearest_position.set_y(otherside_project_path_point.y());
      otherside_enu_nearest_position.set_z(otherside_project_path_point.z());

      double c0 = 0.0;
      double road_width = 3.5;
      if (nearest_p.direction.z() < 0.0) {  // the nearest point is on right
                                            // lanemark.
        if (lane_markers_.has_right_lane_marker()) {
          const auto& lanemarker = lane_markers_.right_lane_marker();
          c0 = lanemarker.c0_position();

          auto ratio = std::abs(c0) / road_width;

          ADEBUG << std::setprecision(15) << "before position, x["
                 << pose.position().x() << "], y[" << pose.position().y()
                 << "], z[" << pose.position().z() << "]";

          pose.mutable_position()->set_x(
              enu_nearest_position.x() +
              ratio * (otherside_enu_nearest_position.x() -
                       enu_nearest_position.x()));

          pose.mutable_position()->set_y(
              enu_nearest_position.y() +
              ratio * (otherside_enu_nearest_position.y() -
                       enu_nearest_position.y()));

          ADEBUG << std::setprecision(15) << "after position, x["
                 << pose.position().x() << "], y[" << pose.position().y()
                 << "], z[" << pose.position().z() << "]";

          if (!FLAGS_enable_gps_heading) {
            double heading = 0.0;
            PointENU point_flu;
            point_flu.set_x(0.0);
            point_flu.set_y(c0);
            point_flu.set_z(0.0);
            find_near_heading(pose, enu_nearest_position, point_flu, &heading);
            pose.set_heading(heading);
          }
        }
      } else {  // the nearest point is on left lanemark.
        if (lane_markers_.has_left_lane_marker()) {
          const auto& lanemarker = lane_markers_.left_lane_marker();
          c0 = lanemarker.c0_position();

          auto ratio = std::abs(c0) / road_width;

          ADEBUG << std::setprecision(15) << "before position, x["
                 << pose.position().x() << "], y[" << pose.position().y()
                 << "], z[" << pose.position().z() << "]";

          pose.mutable_position()->set_x(
              enu_nearest_position.x() +
              ratio * (otherside_enu_nearest_position.x() -
                       enu_nearest_position.x()));

          pose.mutable_position()->set_y(
              enu_nearest_position.y() +
              ratio * (otherside_enu_nearest_position.y() -
                       enu_nearest_position.y()));

          ADEBUG << std::setprecision(15) << "after position, x["
                 << pose.position().x() << "], y[" << pose.position().y()
                 << "], z[" << pose.position().z() << "]";

          if (!FLAGS_enable_gps_heading) {
            double heading = 0.0;
            PointENU point_flu;
            point_flu.set_x(0.0);
            point_flu.set_y(c0);
            point_flu.set_z(0.0);
            find_near_heading(pose, enu_nearest_position, point_flu, &heading);
            pose.set_heading(heading);
          }
        }
      }
    }

    if (!FLAGS_enable_gps_heading) {
      latest_it = predicted_.Latest();
      if (latest_it != predicted_.end()) {
        pose.set_heading((pose.heading() + latest_it->second.heading()) / 2.0);
      }
    }

    predicted_.Push(timestamp_sec, pose);
  }
  return true;
}

void PredictorOutput::find_near_heading(
    const Pose& car_pose,
    const apollo::common::PointENU& nearest_lanemarkpoint_pose_enu,
    const apollo::common::PointENU& nearest_lanemarkpoint_pose_flu,
    double* heading_corrected) {
  auto current_error = std::numeric_limits<double>::max();
  double error = 0.0;
  double range = 0.5;
  for (auto heading = car_pose.heading() - range;
       heading <= car_pose.heading() + range; heading += 0.005) {
    double enu_x, enu_y;
    RotateAxis(-heading, nearest_lanemarkpoint_pose_flu.x(),
               nearest_lanemarkpoint_pose_flu.y(), &enu_x, &enu_y);

    error = std::sqrt(std::pow(
        car_pose.position().y() + enu_y - nearest_lanemarkpoint_pose_enu.y(),
        2.0));

    if (error < current_error) {
      current_error = error;
      *heading_corrected = heading;
    }
  }
}
}  // namespace localization
}  // namespace apollo
