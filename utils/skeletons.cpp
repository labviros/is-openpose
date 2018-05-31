#include "skeletons.hpp"

SkeletonType OpenPoseExtractor::get_skeleton_type(std::string const& name) {
  const static std::unordered_map<std::string, SkeletonType> to_skeleton_type{
      {"Head", SkeletonType::HEAD},
      {"Nose", SkeletonType::NOSE},
      {"Neck", SkeletonType::NECK},
      {"RShoulder", SkeletonType::RIGHT_SHOULDER},
      {"RElbow", SkeletonType::RIGHT_ELBOW},
      {"RWrist", SkeletonType::RIGHT_WRIST},
      {"LShoulder", SkeletonType::LEFT_SHOULDER},
      {"LElbow", SkeletonType::LEFT_ELBOW},
      {"LWrist", SkeletonType::LEFT_WRIST},
      {"RHip", SkeletonType::RIGHT_HIP},
      {"RKnee", SkeletonType::RIGHT_KNEE},
      {"RAnkle", SkeletonType::RIGHT_ANKLE},
      {"LHip", SkeletonType::LEFT_HIP},
      {"LKnee", SkeletonType::LEFT_KNEE},
      {"LAnkle", SkeletonType::LEFT_ANKLE},
      {"REye", SkeletonType::RIGHT_EYE},
      {"LEye", SkeletonType::LEFT_EYE},
      {"REar", SkeletonType::RIGHT_EAR},
      {"LEar", SkeletonType::LEFT_EAR},
      {"Chest", SkeletonType::CHEST},
      {"Background", SkeletonType::BACKGROUND}};

  auto pos = to_skeleton_type.find(name);
  return pos != to_skeleton_type.end() ? pos->second : SkeletonType::UNKNOWN;
}

void OpenPoseExtractor::set_skeleton_links(Skeletons* sk, op::PoseModel model) {
  auto parts = getPoseBodyPartMapping(model);
  auto pairs = getPosePartPairs(model);
  if (pairs.size() % 2 == 0) {
    for (auto it = pairs.begin(); it != pairs.end();) {
      auto link = sk->add_links();
      link->set_begin(get_skeleton_type(parts[*it++]));
      link->set_end(get_skeleton_type(parts[*it++]));
    }
  }
}

Skeletons OpenPoseExtractor::make_skeletons(op::Array<float> const& keypoints, op::PoseModel const pose_model) {
  auto body_part = getPoseBodyPartMapping(pose_model);
  Skeletons skeletons;
  auto n_people = keypoints.getSize(0);
  auto n_parts = keypoints.getSize(1);
  for (auto n = 0; n < n_people; ++n) {
    auto sk = skeletons.add_skeletons();
    for (auto p = 0; p < n_parts; ++p) {
      auto base_index = keypoints.getSize(2) * (n * n_parts + p);
      auto x = keypoints[base_index];
      auto y = keypoints[base_index + 1];
      auto score = keypoints[base_index + 2];
      if ((x + y + score) > 0.0) {
        auto sk_part = sk->add_parts();
        sk_part->set_type(get_skeleton_type(body_part[p]));
        sk_part->set_x(x);
        sk_part->set_y(y);
        sk_part->set_score(score);
      }
    }
  }
  set_skeleton_links(&skeletons, pose_model);
  return skeletons;
}

OpenPoseExtractor::OpenPoseExtractor(OpenPoseOptions const& options) : options(options) {
  output_size = op::flagsToPoint(options.output_resolution, "-1x-1");
  net_input_size = op::flagsToPoint(options.net_resolution, "-1x368");
  pose_model = op::flagsToPoseModel(options.model_pose);
  if (options.scale_gap <= 0. && options.scale_number > 1)
    throw std::runtime_error("\'scale_gap\' must be greater than 0 or scale_number = 1.");

  scale_and_size_extractor =
      std::make_unique<op::ScaleAndSizeExtractor>(net_input_size, output_size, options.scale_number, options.scale_gap);
  pose_extractor = std::make_unique<op::PoseExtractorCaffe>(pose_model, options.model_folder, options.num_gpu_start);

  pose_extractor->initializationOnThread();
}

Skeletons OpenPoseExtractor::extract(Image const& image) {
  std::vector<char> coded(image.data().begin(), image.data().end());
  auto input_image = cv::imdecode(coded, CV_LOAD_IMAGE_COLOR);
  op::Point<int> image_size{input_image.cols, input_image.rows};
  // get desired scale sizes
  std::vector<double> scale_input_to_netinputs;
  std::vector<op::Point<int>> net_inputs_size;
  double scale_input_to_output;
  op::Point<int> output_res;
  std::tie(scale_input_to_netinputs, net_inputs_size, scale_input_to_output, output_res) =
      scale_and_size_extractor->extract(image_size);
  // format input image to OpenPose input and output formats
  auto net_input_array = cvmat_to_input.createArray(input_image, scale_input_to_netinputs, net_inputs_size);
  // estimate pose keypoints and return skeletons
  pose_extractor->forwardPass(net_input_array, image_size, scale_input_to_netinputs);
  auto pose_keypoints = pose_extractor->getPoseKeypoints();
  return make_skeletons(pose_keypoints, pose_model);
}