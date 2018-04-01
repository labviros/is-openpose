#include <is/msgs/image.pb.h>
#include <is/is.hpp>
#include "skeleton.pb.h"

#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>

using namespace is::vision;

std::string parse_entity_id(std::string const& topic) {
  auto first_dot = topic.find_first_of('.');
  if (first_dot == std::string::npos) return "";
  auto second_dot = topic.find_first_of('.', first_dot + 1);
  if (second_dot == std::string::npos) return "";
  return topic.substr(first_dot + 1, second_dot - first_dot - 1);
}

SkeletonType get_skeleton_type(std::string const& name) {
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

void set_skeleton_links(Skeletons* sk, op::PoseModel model) {
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

Skeletons make_skeletons(op::Array<float> const& keypoints, op::PoseModel const pose_model) {
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

int main(int argc, char* argv[]) {
  // general options variable
  std::string uri, service, zipkin_host;
  int zipkin_port, prefetch;
  // openpose options variables
  std::string model_pose, model_folder, net_resolution, output_resolution;
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
  opt_add("prefetch", is::po::value<int>(&prefetch)->default_value(1),
          "number of messages buffered. Negative means without limit.");
  // openpose options
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
  if (scale_gap <= 0. && scale_number > 1) is::critical("\'scale_gap\' must be greater than 0 or scale_number = 1.");

  // initialize openpose required classes
  op::ScaleAndSizeExtractor scaleAndSizeExtractor(net_input_size, output_size, scale_number, scale_gap);
  op::CvMatToOpInput cvmat_to_input;
  op::PoseExtractorCaffe pose_extractor{pose_model, model_folder, num_gpu_start};
  pose_extractor.initializationOnThread();

  is::Tracer tracer(service, zipkin_host, zipkin_port);
  auto channel = is::rmq::Channel::CreateFromUri(uri);
  auto tag = is::consumer_id();
  is::declare_queue(channel, service, tag, /*exclusive*/ false, /*prefetch*/ prefetch);
  is::subscribe(channel, service, "CameraGateway.*.Frame");

  for (;;) {
    auto envelope = channel->BasicConsumeMessage();

    auto start_time = is::current_time();
    auto span = tracer.extract(envelope, tag);

    auto maybe_image = is::unpack<Image>(envelope);
    if (!maybe_image) { continue; }

    std::vector<char> coded(maybe_image->data().begin(), maybe_image->data().end());
    auto input_image = cv::imdecode(coded, CV_LOAD_IMAGE_COLOR);
    auto id = parse_entity_id(envelope->RoutingKey());
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

    auto skeletons = make_skeletons(pose_keypoints, pose_model);
    auto msg = is::pack_proto(skeletons);
    tracer.inject(msg, span->context());
    is::publish(channel, fmt::format("OpenPose.{}.Skeletons", id), msg);

    span->Finish();
    channel->BasicAck(envelope);
    is::info("Took: {}", is::current_time() - start_time);
  }

  return 0;
}