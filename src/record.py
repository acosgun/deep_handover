#!/usr/bin/env python2
import rospy
import std_msgs.msg
import sensor_msgs.msg
import geometry_msgs.msg
import deep_handover.msg

import cv2
import cv_bridge
bridge = cv_bridge.CvBridge()

import sys
import os
import json

class Record():
    def __init__(self,record_dir):
        rospy.init_node('recorder', anonymous=False)
        rospy.Subscriber("/camera/color/image_raw", sensor_msgs.msg.Image, self.image_callback)
        rospy.Subscriber("visual_control",deep_handover.msg.Control,self.control_callback)
        rospy.Subscriber("record_enabled",std_msgs.msg.Bool,self.record_callback)
        rospy.Subscriber("/tcp_wrench",geometry_msgs.msg.Wrench,self.ft_callback)

        self.record_dir = record_dir
        self.img = None
        self.record_enabled = False
        self.annotation_dict = None
        self.save_interval = 5 #frames
        self.control_msg = None

        self.meta_path = os.path.join(record_dir,"meta.json")

        #try to open the meta data in this folder. Create it if it doesn't exist
        try:
            with open(self.meta_path,"rb") as f:
                self.meta_data = json.load(f)
            print("loaded json")
        except:
            self.meta_data = {}
            self.meta_data["image_count"] = 0
            self.save_meta()

        cv2.namedWindow("Recorder")

        self.loop()

    def save_meta(self):
         with open(self.meta_path,"wb") as f:
             json.dump(self.meta_data,f)

    def loop(self):
        r = rospy.Rate(30)
        frame_counter = 0
        while not rospy.is_shutdown():
            if not self.img is None:
                frame_counter += 1

                #crop a square image out the center of the rectangular one
                h,w,c = self.img.shape
                s = min(h,w)
                # x1 = (w-s)/2
                # x2 = x1 + s
                y1 = (h-s)/2
                y2 = y1 + s
                #img_crop= self.img[y1:y2,x1:x2,:]
                img_crop= self.img[y1:y2,640-s:640,:]

                #create a copy of the image to show the user with drawings
                img_show = img_crop.copy()
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_show,'Frame '+str(self.meta_data["image_count"]),(10,450), font, 1,(0,0,255),2,cv2.LINE_AA)
                #if recording is enabled
                if self.record_enabled and not self.control_msg is None:

                    self.annotation_dict = {
                    "vx": self.control_msg.vx,
                    "vy": self.control_msg.vy,
                    "vz": self.control_msg.vz,
                    "rx": self.control_msg.rx,
                    "ry": self.control_msg.ry,
                    "rz": self.control_msg.rz,
                    "gripper_open": self.control_msg.gripper_open,
                    "ft_x": self.ft_msg.force.x,
                    "ft_y": self.ft_msg.force.y,
                    "ft_z": self.ft_msg.force.z,
                    "ft_wx": self.ft_msg.torque.x,
                    "ft_wy": self.ft_msg.torque.y,
                    "ft_wz": self.ft_msg.torque.z
                    }

                    #draw recording text to the show image
                    cv2.putText(img_show,'Record Enabled...',(10,40), font, 1,(0,0,255),2,cv2.LINE_AA)


                    #if we are at a frame interval and the user is providing an input then make a recording
                    if frame_counter % self.save_interval == 0:

                        #create full file paths for frame and annotation
                        img_path = os.path.join(self.record_dir,"%05i_img.png" % self.meta_data["image_count"])
                        annotation_path = os.path.join(self.record_dir,"%05i_gt.txt" % self.meta_data["image_count"])

                        #write the image to file
                        cv2.imwrite(img_path,img_crop)

                        #write the annotation to file
                        with open(annotation_path,"wb") as f:
                            json.dump(self.annotation_dict,f)

                        #update the meta data with the new image count
                        self.meta_data["image_count"] += 1
                        self.save_meta()

                #show the image to the user
                cv2.imshow("Recorder",img_show)
                cv2.waitKey(1)

            r.sleep()



    def image_callback(self,msg):
        rgb_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        bgr_img = rgb_img[...,::-1]
        self.img = bgr_img

    def ft_callback(self,msg):
        self.ft_msg = msg

    def record_callback(self,msg):
        self.record_enabled = msg.data

    def control_callback(self,msg):
        self.control_msg = msg






if __name__ == "__main__":
    record_dir = sys.argv[1]
    Record(record_dir)
