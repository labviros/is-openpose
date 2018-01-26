#include <is/log.hpp>
#include <is/cli.hpp>
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>
#include "skeleton.pb.h"

using namespace is::vision;

SkeletonPoint* get_skeleton_point(Skeleton* sk, std::string part) {
  if (part == "Head") return sk->mutable_head();
  if (part == "Nose") return sk->mutable_nose();
  if (part == "Neck") return sk->mutable_neck();
  if (part == "RShoulder") return sk->mutable_right_shoulder();
  if (part == "RElbow") return sk->mutable_right_elbow();
  if (part == "RWrist") return sk->mutable_right_wrist();
  if (part == "LShoulder") return sk->mutable_left_shoulder();
  if (part == "LElbow") return sk->mutable_left_elbow();
  if (part == "LWrist") return sk->mutable_left_wrist();
  if (part == "RHip") return sk->mutable_right_hip();
  if (part == "RKnee") return sk->mutable_right_knee();
  if (part == "RAnkle") return sk->mutable_right_ankle();
  if (part == "LHip") return sk->mutable_left_hip();
  if (part == "LKnee") return sk->mutable_left_knee();
  if (part == "LAnkle") return sk->mutable_left_ankle();
  if (part == "REye") return sk->mutable_right_eye();
  if (part == "LEye") return sk->mutable_left_eye();
  if (part == "REar") return sk->mutable_right_ear();
  if (part == "LEar") return sk->mutable_left_ear();
  if (part == "Chest") return sk->mutable_chest();
  if (part == "Background") return sk->mutable_background();
  return nullptr;
}

Skeletons make_skeletons(op::Array<float> const& keypoints, op::PoseModel const pose_model) {
  auto model = pose_model == op::PoseModel::COCO_18 ? SkeletonModel::COCO : SkeletonModel::MPI;
  auto body_part = getPoseBodyPartMapping(pose_model);
  Skeletons skeletons;
  auto n_people = keypoints.getSize(0);
  auto n_parts = keypoints.getSize(1);
  for (auto n = 0; n < n_people; ++n) {
    auto sk = skeletons.add_skeletons();
    sk->set_model(model);
    for (auto p = 0; p < n_parts; ++p) {
      auto sk_point = get_skeleton_point(sk, body_part[p]);
      auto base_index = keypoints.getSize(2)*(n*n_parts + p);
      sk_point->set_x(keypoints[base_index]);
      sk_point->set_y(keypoints[base_index + 1]);
      sk_point->set_score(keypoints[base_index + 2]);
    }
  }
  return skeletons;
}

int main(int argc, char* argv[]) {
  // general options variable
  std::string uri, service, zipkin_host;
  int zipkin_port;
  // openpose options variables
  std::string image_path, model_pose, model_folder, net_resolution, output_resolution;
  int32_t num_gpu_start, scale_number;
  double scale_gap;

  is::po::options_description opts("Options");
  auto&& opt_add = opts.add_options();
  // general options
  opt_add("uri,u", is::po::value<std::string>(&uri)->required(), "broker uri");
  opt_add("name", is::po::value<std::string>(&service)->default_value("OpenPose.Skeletons"), "service name");
  opt_add("zipkin-host,z", is::po::value<std::string>(&zipkin_host)->default_value("zipkin.default"),
          "zipkin hostname");
  opt_add("zipkin-port,p", is::po::value<int>(&zipkin_port)->default_value(9411), "zipkin port");
  // openpose options
  // REMOVE THIS OPTION AFTER INITIAL TEST
  opt_add("image-path", is::po::value<std::string>(&image_path)->required(), "Process the desired image.");
  opt_add("model-pose", is::po::value<std::string>(&model_pose)->default_value("COCO"),
          "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, "
          "~10% faster), `MPI_4_layers` (15 keypoints, even faster but less "
          "accurate).");
  opt_add("model-folder", is::po::value<std::string>(&model_folder)->default_value("/models/"),
          "Folder path (absolute or relative) "
          "where the models (pose, face, ...) are "
          "located.");
  opt_add("net-resolution", is::po::value<std::string>(&net_resolution)->default_value("-1x368"),
          "Multiples of 16. If it is increased, the accuracy potentially "
          "increases. If it is"
          " decreased, the speed increases. For maximum speed-accuracy "
          "balance, it should keep the"
          " closest aspect ratio possible to the images or videos to be "
          "processed. Using `-1` in"
          " any of the dimensions, OP will choose the optimal aspect ratio "
          "depending on the user's"
          " input value. E.g. the default `-1x368` is equivalent to "
          "`656x368` in 16:9 resolutions,"
          " e.g. full HD (1980x1080) and HD (1280x720) resolutions.");
  opt_add("output-resolution", is::po::value<std::string>(&output_resolution)->default_value("-1x-1"),
          "The image resolution (display and "
          "output). Use \"-1x-1\" to force the "
          "program to use the"
          " input image resolution.");
  opt_add("num-gpu-start", is::po::value<int32_t>(&num_gpu_start)->default_value(0), "GPU device start number.");
  opt_add("scale_gap ", is::po::value<double>(&scale_gap)->default_value(0.3),
          "Scale gap between scales. No effect unless scale_number > 1. "
          "Initial scale is always 1."
          " If you want to change the initial scale, you actually want to "
          "multiply the"
          " `net_resolution` by your desired initial scale.");
  opt_add("scale-number", is::po::value<int32_t>(&scale_number)->default_value(1), "Number of scales to average.");

  auto vm = is::parse_program_options(argc, argv, opts);
  // read openpose options and check range when necessary
  const auto output_size = op::flagsToPoint(output_resolution, "-1x-1");
  const auto net_input_size = op::flagsToPoint(net_resolution, "-1x368");
  const auto pose_model = op::flagsToPoseModel(model_pose);

  // // Check no contradictory flags enabled
  // if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.)
  //   op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__,
  //             __FILE__);
  if (scale_gap <= 0. && scale_number > 1) is::critical("\'scale_gap\' must be greater than 0 or scale_number = 1.");

  // initialize openpose required classes

  op::ScaleAndSizeExtractor scaleAndSizeExtractor(net_input_size, output_size, scale_number, scale_gap);
  op::CvMatToOpInput cvmat_to_input;
  op::PoseExtractorCaffe pose_extractor{pose_model, model_folder, num_gpu_start};

  // initialize resources on desired thread (in this case single thread, i.e. we init resources
  // here)
  pose_extractor.initializationOnThread();

  // load image. REMOVE AFTER FIRST TEST
  cv::Mat input_image = op::loadImage(image_path, CV_LOAD_IMAGE_COLOR);
  if (input_image.empty()) is::critical("Could not open or find the image: {}", image_path);
  const op::Point<int> image_size{input_image.cols, input_image.rows};


  // get desired scale sizes
  std::vector<double> scale_input_to_netinputs;
  std::vector<op::Point<int>> net_inputs_size;
  double scale_input_to_output;
  op::Point<int> output_res;
  std::tie(scale_input_to_netinputs, net_inputs_size, scale_input_to_output, output_res) =
      scaleAndSizeExtractor.extract(image_size);

  // format input image to OpenPose input and output formats
  const auto net_input_array = cvmat_to_input.createArray(input_image, scale_input_to_netinputs, net_inputs_size);

  // estimate pose keypoints
  pose_extractor.forwardPass(net_input_array, image_size, scale_input_to_netinputs);
  const auto pose_keypoints = pose_extractor.getPoseKeypoints();

  is::info("People detected: {}", pose_keypoints.getSize(0));
  is::info("Number of body parts: {}", pose_keypoints.getSize(1));
  auto skeletons = make_skeletons(pose_keypoints, pose_model);
  skeletons.PrintDebugString();

  return 0;
}