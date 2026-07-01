/*
 * Minimal ROS-free EuRoC (ASL) driver for OpenVINS.
 * Reads imu0 + cam0/cam1 (ASL format) directly, feeds the VioManager, and
 * writes the estimated IMU/body trajectory in TUM format (world<-IMU, Hamilton).
 *
 * Usage: run_euroc_asl <estimator_config.yaml> <mav0_dir> <out.tum>
 */
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "types/IMU.h"
#include "utils/dataset_reader.h"
#include "utils/sensor_data.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"

using namespace ov_msckf;

struct Event {
  double t;        // seconds
  int kind;        // 0 = imu, 1 = camera
  size_t idx;      // index into imu or cam vectors
  bool operator<(const Event &o) const { return t < o.t; }
};

static std::vector<std::array<double, 7>> read_imu(const std::string &csv) {
  std::vector<std::array<double, 7>> out;
  std::ifstream f(csv);
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::stringstream ss(line);
    double ts, wx, wy, wz, ax, ay, az;
    if (ss >> ts >> wx >> wy >> wz >> ax >> ay >> az)
      out.push_back({ts * 1e-9, wx, wy, wz, ax, ay, az});
  }
  return out;
}

static std::vector<std::pair<double, std::string>> read_cam(const std::string &csv) {
  std::vector<std::pair<double, std::string>> out;
  std::ifstream f(csv);
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    auto comma = line.find(',');
    if (comma == std::string::npos)
      continue;
    double ts = std::stod(line.substr(0, comma)) * 1e-9;
    std::string name = line.substr(comma + 1);
    // trim whitespace/CR
    while (!name.empty() && (name.back() == '\r' || name.back() == '\n' || name.back() == ' '))
      name.pop_back();
    out.emplace_back(ts, name);
  }
  return out;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: run_euroc_asl <config.yaml> <mav0_dir> <out.tum>\n";
    return 1;
  }
  std::string config_path = argv[1];
  std::string mav0 = argv[2];
  std::string out_tum = argv[3];
  if (mav0.back() != '/')
    mav0 += '/';

  auto parser = std::make_shared<ov_core::YamlParser>(config_path);
  std::string verbosity = "INFO";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  VioManagerOptions params;
  params.print_and_load(parser);
  params.num_opencv_threads = 4;
  params.use_multi_threading_pubs = false;
  params.use_multi_threading_subs = false;
  if (!parser->successful()) {
    PRINT_ERROR(RED "unable to parse all parameters\n" RESET);
    return 1;
  }
  auto sys = std::make_shared<VioManager>(params);

  auto imu = read_imu(mav0 + "imu0/data.csv");
  auto cam0 = read_cam(mav0 + "cam0/data.csv");
  auto cam1 = read_cam(mav0 + "cam1/data.csv");
  std::cout << "[euroc] imu=" << imu.size() << " cam0=" << cam0.size()
            << " cam1=" << cam1.size() << std::endl;

  // pair cam0/cam1 by index (EuRoC cam0/cam1 are hardware-synced, identical ts)
  size_t ncam = std::min(cam0.size(), cam1.size());

  std::vector<Event> events;
  events.reserve(imu.size() + ncam);
  for (size_t i = 0; i < imu.size(); i++)
    events.push_back({imu[i][0], 0, i});
  for (size_t i = 0; i < ncam; i++)
    events.push_back({cam0[i].first, 1, i});
  std::stable_sort(events.begin(), events.end());

  std::ofstream fout(out_tum);
  fout << "# timestamp tx ty tz qx qy qz qw\n";
  fout.setf(std::ios::fixed);
  fout.precision(9);

  size_t nwritten = 0;
  for (const auto &e : events) {
    if (e.kind == 0) {
      ov_core::ImuData m;
      const auto &r = imu[e.idx];
      m.timestamp = r[0];
      m.wm << r[1], r[2], r[3];
      m.am << r[4], r[5], r[6];
      sys->feed_measurement_imu(m);
    } else {
      cv::Mat i0 = cv::imread(mav0 + "cam0/data/" + cam0[e.idx].second, cv::IMREAD_GRAYSCALE);
      cv::Mat i1 = cv::imread(mav0 + "cam1/data/" + cam1[e.idx].second, cv::IMREAD_GRAYSCALE);
      if (i0.empty() || i1.empty())
        continue;
      ov_core::CameraData cam;
      cam.timestamp = cam0[e.idx].first;
      cam.sensor_ids = {0, 1};
      cam.images = {i0, i1};
      cam.masks = {cv::Mat::zeros(i0.size(), CV_8UC1), cv::Mat::zeros(i1.size(), CV_8UC1)};
      sys->feed_measurement_camera(cam);

      if (sys->initialized()) {
        auto state = sys->get_state();
        Eigen::Vector3d p = state->_imu->pos();             // p_IinG
        Eigen::Matrix3d R_ItoG = state->_imu->Rot().transpose();
        Eigen::Quaterniond q(R_ItoG);
        q.normalize();
        fout << state->_timestamp << " " << p.x() << " " << p.y() << " " << p.z()
             << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        nwritten++;
      }
    }
  }
  fout.close();
  std::cout << "[euroc] wrote " << nwritten << " poses -> " << out_tum << std::endl;
  return 0;
}
