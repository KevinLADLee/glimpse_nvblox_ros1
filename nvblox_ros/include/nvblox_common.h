#ifndef RAMLAB_TOOLS_NVBLOX_COMMON_H_
#define RAMLAB_TOOLS_NVBLOX_COMMON_H_

#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "ramlab_sensor/pcl/point_xyzirrat.hpp"
#include "ramlab_tools/dataset/kitti_common.h"

namespace ramlab_tools {

template <typename PointType>
void saveFPDataToNvbloxFolder(
    const std::string data_path, const pcl::PointCloud<PointType> &input_cloud,
    const size_t &raw_point_size, const Eigen::Matrix4d &T_world_pts,
    const int frame_id, const cv::Mat img = cv::Mat(),
    const Eigen::Matrix3d K = Eigen::Matrix3d::Zero()) {
  const float default_scale_factor = 1000.0f;
  // assume that all points are higher than -10.0m
  // points at height within [-10m, 55m] can be stored
  const float default_scale_offset = 10.0f;
  const int raw_elevation_divisions = 128;
  const int raw_azimuth_divisions = 2048;
  float ratio =
      1.0f * raw_point_size / raw_elevation_divisions / raw_azimuth_divisions;
  int num_elevation_divisions = raw_elevation_divisions;
  int num_azimuth_divisions = round(raw_azimuth_divisions * ratio);

  float vertical_fov = 45.0f / 180.0f * M_PI;
  float horizontal_fov = 2 * M_PI;
  float start_azimuth_rad = 0.0f;
  float end_azimuth_rad = 2 * M_PI;
  float start_elevation_rad = 0.0f;
  float end_elevation_rad = 0.0f;
  float rads_per_pixel_elevation;
  float rads_per_pixel_azimuth;

  int min_ring = 1000, max_ring = -1000;
  for (const auto &pt : input_cloud) {
    float r = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
    max_ring = std::max(max_ring, int(pt.ring));
    min_ring = std::min(min_ring, int(pt.ring));
  }
  // LOG(INFO) << "min_ring: " << min_ring << ", max_ring: " << max_ring;

  start_elevation_rad = 0.0f;
  end_elevation_rad = 0.0f;
  int cnt_start = 0;
  int cnt_end = 0;
  for (const auto &pt : input_cloud) {
    float r = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
    float elevation_angle_rad = acos(pt.z / r);
    if (int(pt.ring) == min_ring) {
      start_elevation_rad += elevation_angle_rad;
      cnt_start++;
    }
    if (int(pt.ring) == max_ring) {
      end_elevation_rad += elevation_angle_rad;
      cnt_end++;
    }
  }
  start_elevation_rad /= cnt_start;
  end_elevation_rad /= cnt_end;

  ////////////////
  int num_rings = max_ring - min_ring + 1;
  // NOTE(gogojjh): if all points are removed, output the default values
  if (num_rings > num_elevation_divisions) {
    start_elevation_rad = M_PI / 2 - vertical_fov / 2;
    end_elevation_rad = M_PI / 2 + vertical_fov / 2;
    rads_per_pixel_elevation = (end_elevation_rad - start_elevation_rad) /
                               (num_elevation_divisions - 1);
    rads_per_pixel_azimuth = horizontal_fov / (num_azimuth_divisions - 1);
  }
  // NOTE(gogojjh): if all points at the start ring and end ring are removed, we
  // need to extend the elevation angle and assume that points are distributed
  // evently at the vertical direction
  else {
    rads_per_pixel_elevation =
        (end_elevation_rad - start_elevation_rad) / (num_rings - 1);
    rads_per_pixel_azimuth = horizontal_fov / (num_azimuth_divisions - 1);
    start_elevation_rad -= rads_per_pixel_elevation * min_ring;
    end_elevation_rad +=
        rads_per_pixel_elevation * (num_elevation_divisions - 1 - max_ring);
    vertical_fov = end_elevation_rad - start_elevation_rad;
  }
  LOG(INFO) << "frame_id: " << frame_id << "; " << num_azimuth_divisions << " "
            << num_elevation_divisions << " " << horizontal_fov << " "
            << vertical_fov << " " << start_azimuth_rad << " "
            << end_azimuth_rad << " " << start_elevation_rad << " "
            << end_elevation_rad;

  std::stringstream ss;
  ss.str("");
  ss << data_path << "/frame-" << std::setfill('0') << std::setw(6) << frame_id
     << ".lidar-intrinsics.txt";
  std::ofstream int_file(ss.str());
  int_file << std::fixed << std::setprecision(5);
  int_file << num_azimuth_divisions << " " << num_elevation_divisions << " "
           << horizontal_fov << " " << vertical_fov << " " << start_azimuth_rad
           << " " << end_azimuth_rad << " " << start_elevation_rad << " "
           << end_elevation_rad << std::endl;
  int_file.close();

  // NOTE(gogojjh): convert as mm
  ss.str("");
  ss << data_path << "/frame-" << std::setfill('0') << std::setw(6) << frame_id
     << ".depth.png";
  cv::Mat depth_img(num_elevation_divisions, num_azimuth_divisions, CV_16UC1,
                    cv::Scalar(0));
  for (const auto &pt : input_cloud) {
    float r = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
    float elevation_angle_rad = acos(pt.z / r);
    float azimuth_angle_rad = M_PI - atan2(pt.y, pt.x);
    int row_id = round((elevation_angle_rad - start_elevation_rad) /
                       rads_per_pixel_elevation);
    if (row_id < 0 || row_id > num_elevation_divisions - 1) continue;

    int col_id = round(azimuth_angle_rad / rads_per_pixel_azimuth);
    if (col_id >= num_azimuth_divisions) col_id -= num_azimuth_divisions;

    float dep = r * default_scale_factor;
    if (dep > std::numeric_limits<uint16_t>::max()) continue;
    if (dep < 0.0f) continue;
    depth_img.at<uint16_t>(row_id, col_id) = uint16_t(dep);
  }
  cv::imwrite(ss.str(), depth_img);

  // NOTE(gogojjh): convert as mm
  ss.str("");
  ss << data_path << "/frame-" << std::setfill('0') << std::setw(6) << frame_id
     << ".height.png";
  cv::Mat height_image(num_elevation_divisions, num_azimuth_divisions, CV_16UC1,
                       cv::Scalar(0));
  for (const auto &pt : input_cloud) {
    float r = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
    float elevation_angle_rad = acos(pt.z / r);
    float azimuth_angle_rad = M_PI - atan2(pt.y, pt.x);
    int row_id = round((elevation_angle_rad - start_elevation_rad) /
                       rads_per_pixel_elevation);
    if (row_id < 0 || row_id > num_elevation_divisions - 1) continue;

    int col_id = round(azimuth_angle_rad / rads_per_pixel_azimuth);
    if (col_id >= num_azimuth_divisions) col_id -= num_azimuth_divisions;

    float z = (pt.z + default_scale_offset) * default_scale_factor;
    if (z > std::numeric_limits<uint16_t>::max()) continue;
    if (z < 0.0f) continue;
    height_image.at<uint16_t>(row_id, col_id) = uint16_t(z);
  }
  cv::imwrite(ss.str(), height_image);

  ss.str("");
  ss << data_path << "/frame-" << std::setfill('0') << std::setw(6) << frame_id
     << ".pose.txt";
  std::ofstream output_file(ss.str());
  output_file << std::fixed << std::setprecision(9);
  output_file << T_world_pts << std::endl;
  output_file.close();

  if (!img.empty()) {
    ss.str("");
    ss << data_path << "/frame-" << std::setfill('0') << std::setw(6)
       << frame_id << ".color.png";
    cv::imwrite(ss.str(), img);

    ss.str("");
    ss << data_path << "/../camera-intrinsics.txt";
    output_file.open(ss.str());
    output_file << std::fixed << std::setprecision(5);
    output_file << K << std::endl;
    output_file.close();
  }
}

template <typename PointType>
void saveKITTIDataToNvbloxFolder(
    const std::string data_path, const pcl::PointCloud<PointType> &input_cloud,
    const size_t &raw_point_size, const Eigen::Matrix4d &T_world_pts,
    const int frame_id, const cv::Mat img,
    const ramlab_tools::CameraCalibration &cam0_calibration,
    const ramlab_tools::CameraCalibration &cam_calibration,
    const bool use_semantics = false) {
  const float default_scale_factor = 1000.0f;
  // assume that all points are higher than -10.0m
  // points at height within [-10m, 55m] can be stored
  const float default_scale_offset = 10.0f;
  const int raw_elevation_divisions = 64;
  const int raw_azimuth_divisions = 2048;
  int num_elevation_divisions = raw_elevation_divisions;
  int num_azimuth_divisions = raw_azimuth_divisions;

  float fov_up = 87.0f;
  float fov_down = 115.0f;
  float vertical_fov = (fov_down - fov_up) / 180.0f * M_PI;
  float horizontal_fov = 2 * M_PI;
  float start_azimuth_rad = 0.0f;
  float end_azimuth_rad = 2 * M_PI;
  float start_elevation_rad = 0.0f;
  float end_elevation_rad = 0.0f;
  float rads_per_pixel_elevation;
  float rads_per_pixel_azimuth;

  start_elevation_rad = 2 * M_PI;
  end_elevation_rad = 0.0f;
  int cnt_start = 0;
  int cnt_end = 0;
  for (const auto &pt : input_cloud) {
    float r = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
    float elevation_angle_rad = acos(pt.z / r);
    start_elevation_rad = start_elevation_rad < elevation_angle_rad
                              ? start_elevation_rad
                              : elevation_angle_rad;
    end_elevation_rad = end_elevation_rad > elevation_angle_rad
                            ? end_elevation_rad
                            : elevation_angle_rad;
  }

  ////////////////
  {
    rads_per_pixel_elevation = (end_elevation_rad - start_elevation_rad) /
                               (num_elevation_divisions - 1);
    rads_per_pixel_azimuth = horizontal_fov / (num_azimuth_divisions - 1);
    vertical_fov = end_elevation_rad - start_elevation_rad;
  }
  LOG(INFO) << "frame_id: " << frame_id << "; " << num_azimuth_divisions << " "
            << num_elevation_divisions << " " << horizontal_fov << " "
            << vertical_fov << " " << start_azimuth_rad << " "
            << end_azimuth_rad << " " << start_elevation_rad << " "
            << end_elevation_rad;

  std::stringstream ss;
  ss.str("");
  ss << data_path << "/frame-" << std::setfill('0') << std::setw(6) << frame_id
     << ".lidar-intrinsics.txt";
  std::ofstream int_file(ss.str());
  int_file << std::fixed << std::setprecision(5);
  int_file << num_azimuth_divisions << " " << num_elevation_divisions << " "
           << horizontal_fov << " " << vertical_fov << " " << start_azimuth_rad
           << " " << end_azimuth_rad << " " << start_elevation_rad << " "
           << end_elevation_rad << std::endl;
  int_file.close();

  // NOTE(gogojjh): convert as mm
  ss.str("");
  ss << data_path << "/frame-" << std::setfill('0') << std::setw(6) << frame_id
     << ".depth.png";
  cv::Mat depth_img(num_elevation_divisions, num_azimuth_divisions, CV_16UC1,
                    cv::Scalar(0));
  for (const auto &pt : input_cloud) {
    float r = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
    float elevation_angle_rad = acos(pt.z / r);
    float azimuth_angle_rad = M_PI - atan2(pt.y, pt.x);
    int row_id = round((elevation_angle_rad - start_elevation_rad) /
                       rads_per_pixel_elevation);
    if (row_id < 0 || row_id > num_elevation_divisions - 1) continue;

    int col_id = round(azimuth_angle_rad / rads_per_pixel_azimuth);
    if (col_id >= num_azimuth_divisions) col_id -= num_azimuth_divisions;

    float dep = r * default_scale_factor;
    if (dep > std::numeric_limits<uint16_t>::max()) continue;
    if (dep < 0.0f) continue;
    depth_img.at<uint16_t>(row_id, col_id) = uint16_t(dep);
  }
  cv::imwrite(ss.str(), depth_img);

  // NOTE(gogojjh): convert as mm
  ss.str("");
  ss << data_path << "/frame-" << std::setfill('0') << std::setw(6) << frame_id
     << ".height.png";
  cv::Mat height_image(num_elevation_divisions, num_azimuth_divisions, CV_16UC1,
                       cv::Scalar(0));
  for (const auto &pt : input_cloud) {
    float r = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
    float elevation_angle_rad = acos(pt.z / r);
    float azimuth_angle_rad = M_PI - atan2(pt.y, pt.x);
    int row_id = round((elevation_angle_rad - start_elevation_rad) /
                       rads_per_pixel_elevation);
    if (row_id < 0 || row_id > num_elevation_divisions - 1) continue;

    int col_id = round(azimuth_angle_rad / rads_per_pixel_azimuth);
    if (col_id >= num_azimuth_divisions) col_id -= num_azimuth_divisions;

    float z = (pt.z + default_scale_offset) * default_scale_factor;
    if (z > std::numeric_limits<uint16_t>::max()) continue;
    if (z < 0.0f) continue;
    height_image.at<uint16_t>(row_id, col_id) = uint16_t(z);
  }
  cv::imwrite(ss.str(), height_image);

  ss.str("");
  ss << data_path << "/frame-" << std::setfill('0') << std::setw(6) << frame_id
     << ".pose.txt";
  std::ofstream output_file(ss.str());
  output_file << std::fixed << std::setprecision(9);
  output_file << T_world_pts << std::endl;
  output_file.close();

  if (!img.empty()) {
    ss.str("");
    ss << data_path << "/frame-" << std::setfill('0') << std::setw(6)
       << frame_id << ".color.png";
    cv::imwrite(ss.str(), img);

    ss.str("");
    ///////////////// save camera intrinsics for each frame
    // ss << data_path << "/frame-" << std::setfill('0') << std::setw(6)
    //    << frame_id << ".camera-intrinsics.txt";
    ///////////////// save only one camera intrinsics for all frames
    // intrinsics 1:
    // P_rect_xx: 3x4 matrix
    // R_rect_00: 3x3 matrix -> augmented as a 4x4 matrix
    // camera_intrinsics = P_rect_xx * R_rect_00
    // Eigen::Matrix4d R_rect_00 = Eigen::Matrix4d::Identity();
    // R_rect_00.block<3, 3>(0, 0) = cam0_calibration.rect_mat;
    // Eigen::MatrixXd cam_intrinsics = cam_calibration.projection_mat *
    // R_rect_00;

    // intrinsics 2:
    ss << data_path << "/../camera-intrinsics.txt";
    std::cout << ss.str() << std::endl;
    output_file.open(ss.str());
    output_file << img.rows << " " << img.cols << std::endl;
    output_file << std::fixed << std::setprecision(5);
    output_file << cam0_calibration.projection_mat << std::endl;
    output_file << cam_calibration.rect_mat << std::endl;
    output_file.close();
  }
}

template <typename PointType>
void saveSemanticKITTIDataToNvbloxFolder(
    const std::string data_path, const pcl::PointCloud<PointType> &input_cloud,
    const size_t &raw_point_size, const Eigen::Matrix4d &T_world_pts,
    const int frame_id) {
  const float default_scale_factor = 1000.0f;
  // assume that all points are higher than -10.0m
  // points at height within [-10m, 55m] can be stored
  const float default_scale_offset = 10.0f;
  const int raw_elevation_divisions = 64;
  const int raw_azimuth_divisions = 2048;
  int num_elevation_divisions = raw_elevation_divisions;
  int num_azimuth_divisions = raw_azimuth_divisions;

  float fov_up = 87.0f;
  float fov_down = 115.0f;
  float vertical_fov = (fov_down - fov_up) / 180.0f * M_PI;
  float horizontal_fov = 2 * M_PI;
  float start_azimuth_rad = 0.0f;
  float end_azimuth_rad = 2 * M_PI;
  float start_elevation_rad = 0.0f;
  float end_elevation_rad = 0.0f;
  float rads_per_pixel_elevation;
  float rads_per_pixel_azimuth;

  start_elevation_rad = 2 * M_PI;
  end_elevation_rad = 0.0f;
  int cnt_start = 0;
  int cnt_end = 0;
  for (const auto &pt : input_cloud) {
    float r = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
    float elevation_angle_rad = acos(pt.z / r);
    start_elevation_rad = start_elevation_rad < elevation_angle_rad
                              ? start_elevation_rad
                              : elevation_angle_rad;
    end_elevation_rad = end_elevation_rad > elevation_angle_rad
                            ? end_elevation_rad
                            : elevation_angle_rad;
  }

  ////////////////
  {
    rads_per_pixel_elevation = (end_elevation_rad - start_elevation_rad) /
                               (num_elevation_divisions - 1);
    rads_per_pixel_azimuth = horizontal_fov / (num_azimuth_divisions - 1);
    vertical_fov = end_elevation_rad - start_elevation_rad;
  }
  LOG(INFO) << "frame_id: " << frame_id << "; " << num_azimuth_divisions << " "
            << num_elevation_divisions << " " << horizontal_fov << " "
            << vertical_fov << " " << start_azimuth_rad << " "
            << end_azimuth_rad << " " << start_elevation_rad << " "
            << end_elevation_rad;

  std::stringstream ss;
  ss.str("");
  ss << data_path << "/frame-" << std::setfill('0') << std::setw(6) << frame_id
     << ".lidar-intrinsics.txt";
  std::ofstream int_file(ss.str());
  int_file << std::fixed << std::setprecision(5);
  int_file << num_azimuth_divisions << " " << num_elevation_divisions << " "
           << horizontal_fov << " " << vertical_fov << " " << start_azimuth_rad
           << " " << end_azimuth_rad << " " << start_elevation_rad << " "
           << end_elevation_rad << std::endl;
  int_file.close();

  // NOTE(gogojjh): convert as mm
  ss.str("");
  ss << data_path << "/frame-" << std::setfill('0') << std::setw(6) << frame_id
     << ".depth.png";
  cv::Mat depth_img(num_elevation_divisions, num_azimuth_divisions, CV_16UC1,
                    cv::Scalar(0));
  for (const auto &pt : input_cloud) {
    float r = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
    float elevation_angle_rad = acos(pt.z / r);
    float azimuth_angle_rad = M_PI - atan2(pt.y, pt.x);
    int row_id = round((elevation_angle_rad - start_elevation_rad) /
                       rads_per_pixel_elevation);
    if (row_id < 0 || row_id > num_elevation_divisions - 1) continue;

    int col_id = round(azimuth_angle_rad / rads_per_pixel_azimuth);
    if (col_id >= num_azimuth_divisions) col_id -= num_azimuth_divisions;

    float dep = r * default_scale_factor;
    if (dep > std::numeric_limits<uint16_t>::max()) continue;
    if (dep < 0.0f) continue;
    depth_img.at<uint16_t>(row_id, col_id) = uint16_t(dep);
  }
  cv::imwrite(ss.str(), depth_img);

  // NOTE(gogojjh): convert as mm
  ss.str("");
  ss << data_path << "/frame-" << std::setfill('0') << std::setw(6) << frame_id
     << ".height.png";
  cv::Mat height_image(num_elevation_divisions, num_azimuth_divisions, CV_16UC1,
                       cv::Scalar(0));
  for (const auto &pt : input_cloud) {
    float r = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
    float elevation_angle_rad = acos(pt.z / r);
    float azimuth_angle_rad = M_PI - atan2(pt.y, pt.x);
    int row_id = round((elevation_angle_rad - start_elevation_rad) /
                       rads_per_pixel_elevation);
    if (row_id < 0 || row_id > num_elevation_divisions - 1) continue;

    int col_id = round(azimuth_angle_rad / rads_per_pixel_azimuth);
    if (col_id >= num_azimuth_divisions) col_id -= num_azimuth_divisions;

    float z = (pt.z + default_scale_offset) * default_scale_factor;
    if (z > std::numeric_limits<uint16_t>::max()) continue;
    if (z < 0.0f) continue;
    height_image.at<uint16_t>(row_id, col_id) = uint16_t(z);
  }
  cv::imwrite(ss.str(), height_image);

  // NOTE(gogojjh): semantic label (1-255)
  ss.str("");
  ss << data_path << "/frame-" << std::setfill('0') << std::setw(6) << frame_id
     << ".label.png";
  cv::Mat label_image(num_elevation_divisions, num_azimuth_divisions, CV_16UC1,
                      cv::Scalar(0));
  for (const auto &pt : input_cloud) {
    float r = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
    float elevation_angle_rad = acos(pt.z / r);
    float azimuth_angle_rad = M_PI - atan2(pt.y, pt.x);
    int row_id = round((elevation_angle_rad - start_elevation_rad) /
                       rads_per_pixel_elevation);
    if (row_id < 0 || row_id > num_elevation_divisions - 1) continue;

    int col_id = round(azimuth_angle_rad / rads_per_pixel_azimuth);
    if (col_id >= num_azimuth_divisions) col_id -= num_azimuth_divisions;
    label_image.at<uint16_t>(row_id, col_id) = uint16_t(pt.label);
  }
  cv::imwrite(ss.str(), label_image);

  ss.str("");
  ss << data_path << "/frame-" << std::setfill('0') << std::setw(6) << frame_id
     << ".pose.txt";
  std::ofstream output_file(ss.str());
  output_file << std::fixed << std::setprecision(9);
  output_file << T_world_pts << std::endl;
  output_file.close();
}
}  // namespace ramlab_tools
// namespace ramlab_tools

#endif  // RAMLAB_TOOLS_NVBLOX_COMMON_H_
