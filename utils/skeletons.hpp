#pragma once

#include <is/msgs/image.pb.h>
#include <memory>
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>
#include "skeleton.pb.h"

using namespace is::vision;

struct OpenPoseOptions {
  std::string model_pose;
  std::string model_folder;
  std::string net_resolution;
  std::string output_resolution;
  int32_t num_gpu_start;
  int32_t scale_number;
  double scale_gap;
};

struct OpenPoseExtractor {
  OpenPoseOptions options;
  std::unique_ptr<op::ScaleAndSizeExtractor> scale_and_size_extractor;
  std::unique_ptr<op::PoseExtractorCaffe> pose_extractor;
  op::CvMatToOpInput cvmat_to_input;
  op::PoseModel pose_model;
  op::Point<int> net_input_size;
  op::Point<int> output_size;

  OpenPoseExtractor(OpenPoseOptions const& options);
  SkeletonType get_skeleton_type(std::string const& name);
  void set_skeleton_links(Skeletons* sk, op::PoseModel model);
  Skeletons make_skeletons(op::Array<float> const& keypoints, op::PoseModel const pose_model);
  Skeletons extract(Image const& image);
};