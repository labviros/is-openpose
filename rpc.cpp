#include <is/msgs/common.pb.h>
#include <is/msgs/image.pb.h>
#include "skeleton.pb.h"
#include <is/core.hpp>
#include <is/rpc.hpp>
#include <is/rpc/interceptors/log-interceptor.hpp>
#include "skeletons.hpp"

using namespace is::vision;
using namespace is::common;

int main(int argc, char* argv[]) {
  // general options variable
  std::string uri, service;
  int prefetch;
  OpenPoseOptions op_options;

  is::po::options_description opts("Options");
  auto&& opt_add = opts.add_options();
  // general options
  opt_add("uri,u", is::po::value<std::string>(&uri)->required(), "broker uri");
  opt_add("name", is::po::value<std::string>(&service)->default_value("OpenPose.Skeletons"), "service name");
  opt_add("prefetch", is::po::value<int>(&prefetch)->default_value(1),
          "number of messages buffered. Negative means without limit.");
  // openpose options
  opt_add("model-pose", is::po::value<std::string>(&op_options.model_pose)->default_value("MPI_4_layers"),
          "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, "
          "~10% faster), `MPI_4_layers` (15 keypoints, even faster but less "
          "accurate).");
  opt_add("model-folder", is::po::value<std::string>(&op_options.model_folder)->default_value("/models/"),
          "Folder path (absolute or relative) "
          "where the models (pose, face, ...) are "
          "located.");
  opt_add("net-resolution", is::po::value<std::string>(&op_options.net_resolution)->default_value("-1x368"),
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
  opt_add("output-resolution", is::po::value<std::string>(&op_options.output_resolution)->default_value("-1x-1"),
          "The image resolution (display and "
          "output). Use \"-1x-1\" to force the "
          "program to use the"
          " input image resolution.");
  opt_add("num-gpu-start", is::po::value<int32_t>(&op_options.num_gpu_start)->default_value(0),
          "GPU device start number.");
  opt_add("scale_gap ", is::po::value<double>(&op_options.scale_gap)->default_value(0.3),
          "Scale gap between scales. No effect unless scale_number > 1. "
          "Initial scale is always 1."
          " If you want to change the initial scale, you actually want to "
          "multiply the"
          " `net_resolution` by your desired initial scale.");
  opt_add("scale-number", is::po::value<int32_t>(&op_options.scale_number)->default_value(1),
          "Number of scales to average.");

  auto vm = is::parse_program_options(argc, argv, opts);

  is::info("Loading OpenPose Model {}", op_options.model_pose);
  OpenPoseExtractor op_extractor(op_options);
  is::info("Connecting to {}", uri);
  auto channel = is::rmq::Channel::CreateFromUri(uri);
  is::info("Connected");
  is::ServiceProvider provider;
  is::LogInterceptor log_interceptor;
  provider.add_interceptor(log_interceptor);
  provider.connect(channel);
  auto provider_tag = provider.declare_queue("OpenPose");

  provider.delegate<Image, Skeletons>(provider_tag, "GetSkeletons",
                                      [&](Image const& image, Skeletons* skeletons) -> Status {
                                        *skeletons = op_extractor.extract(image);
                                        return is::make_status(StatusCode::OK);
                                      });

  provider.run();
  return 0;
}