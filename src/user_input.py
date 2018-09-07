#!/usr/bin/env python2
import rospy
import std_msgs.msg
import sensor_msgs.msg
import geometry_msgs.msg
import deep_handover.msg

class UserInput:

    def __init__(self):
        rospy.init_node('user_input', anonymous=False)
        rospy.Subscriber("joy", sensor_msgs.msg.Joy, self.joystick_callback)
        self.ctrl_pub = rospy.Publisher("visual_control",deep_handover.msg.Control,queue_size=10)
        self.record_pub = rospy.Publisher("record_enabled",std_msgs.msg.Bool,queue_size=10)
        
        self.gripper_open = True
        print("User Input Spinning")
        self.spin()

    def spin(self):
        #create blank message
        self.ctrl_msg = deep_handover.msg.Control()
        self.record_msg = std_msgs.msg.Bool()

        #publish the latest control message at 30Hz
        r = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.ctrl_pub.publish(self.ctrl_msg)
            self.record_pub.publish(self.record_msg)
            r.sleep()

    def joystick_callback(self,data):
        #Get all the axes and button values
        left_x, left_y, trig_l, right_x, right_y, trig_r, dpad_x, dpad_y = data.axes
        btn_a, btn_b, btn_x, btn_y, bump_l, bump_r, back, menu, _, stick_l, stick_r, _, _ = data.buttons

        #create a new robot control message
        msg = deep_handover.msg.Control()
        #translation
        msg.vx = - deadband(left_x)
        msg.vy = - deadband(left_y)
        msg.vz = deadband(trig_l/2.0 - trig_r/2.0)
        #rotation
        msg.rx =  deadband(right_y)
        msg.ry = - deadband(right_x)
        msg.rz = bump_r - bump_l
        
        #flags

        if btn_a:
            self.gripper_open = True
        if btn_b:
            self.gripper_open = False

        msg.gripper_open = self.gripper_open

        self.ctrl_msg = msg

        if btn_y:
            self.record_msg = std_msgs.msg.Bool(True)
        if btn_x:
            self.record_msg = std_msgs.msg.Bool(False)




def deadband(var,band=0.2):
    var = max(-1.0,min(var,1.0))

    if var > band:
        return (var-band) / (1.0-band)

    if var < -band:
        return (var+band) / (1.0-band)
    return 0.0

if __name__ == "__main__":

    UserInput()
