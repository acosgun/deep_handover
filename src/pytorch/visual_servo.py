#!/usr/bin/env python2
import rospy
import torch
import cv2
import numpy as np

import sensor_msgs.msg
import deep_handover.msg
import geometry_msgs.msg
import cv_bridge
bridge = cv_bridge.CvBridge()
from torchvision import transforms

class VisualServo():

    def __init__(self):
        rospy.init_node('visual_servo', anonymous=False)

        self.ctrl_pub = rospy.Publisher("visual_control",deep_handover.msg.Control,queue_size=10)
        rospy.Subscriber("/camera/color/image_raw", sensor_msgs.msg.Image, self.image_callback)
        rospy.Subscriber("/tcp_wrench",geometry_msgs.msg.Wrench,self.ft_callback)

        self.img_rgb = None
        self.ft_msg = None
        self.net = torch.load("models/handover_2.pt")

        # self.net = torch.load("face_track_nednet.pt")
        self.net.eval()
        self.net.cuda()

        cv2.namedWindow("Camera",0)
        self.loop()

    def loop(self):
        #create blank message
        msg = deep_handover.msg.Control()

        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

        out_smooth = np.zeros((1))
        smooth_a = 0.1

        #publish the latest control message at 30Hz
        r = rospy.Rate(30)

        while not rospy.is_shutdown():

            if not self.img_rgb is None and not self.ft_msg is None:

                #crop the center of the image
                h,w,c = self.img_rgb.shape
                s = min(h,w)
                # x1 = (w-s)/2
                # x2 = x1 + s
                # y1 = (h-s)/2
                # y2 = y1 + s
                #
                # img_crop= self.img_rgb[y1:y2,x1:x2,:]

                y1 = (h-s)/2
                y2 = y1 + s
                img_crop= self.img_rgb[y1:y2,640-s:640,:]

                img_resized = cv2.resize(img_crop,(224,224))
                cv2.imshow("Camera",img_resized[:,:,::-1])
                cv2.waitKey(1)
                # img_resized = img_resized[:,:,::-1].copy()
                img_tensor = transform(img_resized)
                # print(img_tensor)
                img_tensor.unsqueeze_(0)
                img_var = torch.autograd.Variable(img_tensor).cuda()
                force_tensor = torch.tensor([
                self.ft_msg.force.x,
                self.ft_msg.force.y,
                self.ft_msg.force.z,
                self.ft_msg.torque.x,
                self.ft_msg.torque.y,
                self.ft_msg.torque.z
                ]).unsqueeze_(0)

                force_var = torch.autograd.Variable(force_tensor).cuda()
                out = self.net(img_var,force_var).cpu().data[0].numpy() # * 1.8

                # low pass filter
                out_smooth += smooth_a * (out - out_smooth)


                msg = deep_handover.msg.Control()
                # msg.vx,msg.vy,msg.vz,msg.rx,msg.ry,rz = out_smooth

                msg.gripper_open = float(out_smooth) > .5
                # print(msg)


            self.ctrl_pub.publish(msg)
            r.sleep()





    def image_callback(self,msg):
        img_rgb = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.img_rgb = img_rgb

    def ft_callback(self,msg):
        self.ft_msg = msg
if __name__ == "__main__":

    VisualServo()
