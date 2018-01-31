import cv2
import numpy as np
import matplotlib.cm as cm
import ispy
from ismsgs.common_pb2 import SamplingSettings
from ismsgs.camera_pb2 import CameraConfig
from ismsgs.image_pb2 import *
from ismsgs.skeleton_pb2 import *
from google.protobuf.empty_pb2 import Empty

c = ispy.Connection('192.168.1.100', 5672)

n_cam=4
color_space = ColorSpace(value=ColorSpaces.Value('RGB'))
sampling = SamplingSettings(frequency=5.0)
camera_config = CameraConfig(sampling=sampling, image=ImageSettings(color_space=color_space))

cmap = cm.get_cmap('gist_rainbow')
images = {}
sk_images = {}
all_skeletons = {}
for n in range(n_cam):
  images[n] = np.zeros((728, 1288, 3), np.uint8)
  sk_images[n] = np.zeros((728, 1288, 3), np.uint8)

def get_id(context):
  return int(context['routing_key'].split('.')[-2])

def on_image(c, context, msg):
  global images
  cam_id = get_id(context)
  buf = np.frombuffer(msg.data, dtype=np.uint8)
  images[cam_id] = cv2.imdecode(buf, flags=cv2.IMREAD_COLOR)
  
  text = "CameraGateway.{}".format(cam_id)
  fontFace = cv2.FONT_HERSHEY_SIMPLEX
  fontScale = 1.25
  thickness = 2
  text_size, baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)
  baseline += thickness
  org = (0, text_size[1])
  pt1, pt2 = (0,0), text_size
  cv2.rectangle(img=images[cam_id], pt1=pt1, pt2=pt2, color=(0,0,0), thickness=-1)
  pt1, pt2 = (0, text_size[1] + thickness), (text_size[0], text_size[1] + thickness)
  cv2.line(img=images[cam_id], pt1=pt1, pt2=pt2, color=(0,0,0), thickness=thickness)
  cv2.putText(img=images[cam_id], text=text, org=org, fontFace=fontFace, fontScale=fontScale, color=(255,255,255), thickness=thickness, bottomLeftOrigin=False)
  
def on_skeletons(c, context, msg):
  global images
  global sk_images
  cam_id = get_id(context) 
  links = msg.links
  sk_images[cam_id] = np.copy(images[cam_id])

  for skeleton in msg.skeletons:
    available_parts = {}
    for part in skeleton.parts:
      if (part.x + part.y + part.score) > 0.0:
        available_parts[part.type] = {'x':part.x, 'y':part.y}
    
    n_links = len(links)
    for link, n in zip(links, range(n_links)):
      if link.begin in available_parts.keys() and link.end in available_parts.keys():
        pt1 = (int(available_parts[link.begin]['x']), int(available_parts[link.begin]['y']))
        pt2 = (int(available_parts[link.end]['x']), int(available_parts[link.end]['y']))
        c = cmap(1.0* n / n_links)
        color = (int(255.0*c[0]), int(255.0*c[1]), int(255.0*c[2]))
        cv2.line(img=sk_images[cam_id], pt1=pt1, pt2=pt2, color=color, thickness=6)

  im_full = np.hstack((np.vstack((sk_images[0], sk_images[1])), np.vstack((sk_images[2], sk_images[3]))))
  im_full = cv2.resize(im_full, (0,0), fx=0.5, fy=0.5)
  cv2.imshow('is', im_full)
  cv2.waitKey(1)

def config_reply(c, context, msg):
    print('Camera config reply: {}'.format(context['headers']['rpc-status']))
  
set_config = c.rpc('CameraGateway.{}.SetConfig', Empty, config_reply)
for n in range(n_cam):
  set_config(camera_config, n)
  c.subscribe('CameraGateway.{}.Frame'.format(n), Image, on_image)
  c.subscribe('OpenPose.{}.Skeletons'.format(n), Skeletons, on_skeletons)

c.listen()