import pyrealsense2 as rs
# pyrealsense sdk 2.0ver import
import math

import argparse
import time

import cv2
import numpy as np

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh



fps_time = 0

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# pipeline is that import realsense 3 modules.
# config streaming values set your camera configure.
# enable_stream(camera_module, resolution_w,resolution_h,pixel_format,frame)
# frame was fixed values.  

joint_dist = [[0] * 15 for j in range(12)] # distance array values setup
joint_pointX = [[1] * 15 for j in range(12)] # for return value of axisX
joint_pointY = [[1] * 15 for j in range(12)] # for return value of axisY
joint_depth_array_temp = [[0] * 15 for j in range(12)]
frame_add = 0
angle_filter = [[[0]*7]*14]*12
angle_filter_count=0


# get list Median value
def getMedian(array): 
    a_len = len(array)               
    if (a_len == 0):
        return None 
    a_center = int(a_len / 2)     
    if (a_len % 2 == 1):   
        return array[a_center]   
    else:
        return (array[a_center - 1] + array[a_center]) / 2.0 


# depth data calling
def multiple_depth_data(tempX,tempY,i,humans,depth_data_array,image_rgb):
    for j in range(0,14):
        if humans[i] != None:   # first person

            if tempX[i][j] != 1 and tempY[i][j] != 1: # not int(1), axis X, Y
                temp = [0]
                minX,minY,maxX,maxY = tempX[i][j]-2,tempY[i][j]-2,tempX[i][j]+2,tempY[i][j]+2
                if minX < 0:
                    minX = 0
                elif minY < 0:
                    minY = 0
                elif maxX > 640:
                    maxX = 640
                elif maxY > 480:
                    maxY = 480
                for avgX in range(minX,maxX):
                    for avgY in range(minY,maxY):
                        if int(depth_data_array[avgY][avgX]/80) == 0: # array size = 480*640
                            continue
                        # debug print(int((depth_data_array[avgY][avgX]/80)*37.8)) # 1cm = 37.8 pixel
                        temp.append(int(depth_data_array[avgY][avgX]/80)*37.8)
                temp.sort()
                joint_depth_array_temp[i][j] = int(getMedian(temp))  #depth filtering
            # debuging
            # if int(joint_depth_array_temp[i][j]) != 0:
                # cv2.putText(image_rgb,str(int(joint_depth_array_temp[i][j])),(tempX[i][j],tempY[i][j]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
         
    return joint_depth_array_temp,image_rgb


 # Not Yet. many peoples, pair of arm angle. 
def Angle_calculator(joint_pointX,joint_pointY,joint_dist, angle_filter,angle_filter_count,humans,i):
    if humans[i] != None:
        if joint_pointX[i][1] != 1 and joint_pointX[i][2] != 1 and joint_pointX[i][3] != 1 and joint_dist[i][1] != 0 and joint_dist[i][2] != 0 and joint_dist[i][3] != 0  : # Rshoulder
            x1=joint_pointX[i][1] - joint_pointX[i][2]
            y1=joint_pointY[i][1] - joint_pointY[i][2]
            x2=joint_pointX[i][3] - joint_pointX[i][2]
            y2=joint_pointY[i][3] - joint_pointY[i][2]
            x3=joint_pointX[i][3]
            y3=joint_pointY[i][3]
            z1=joint_dist[i][1] - joint_dist[i][2]
            z2=joint_dist[i][3] - joint_dist[i][2]
            z3=joint_dist[i][3]
            angle_filter[i][2][angle_filter_count] = int((math.acos(((x1*x2) + (y1*y2) + (z1*z2)) / (math.sqrt(math.pow(x1, 2) + math.pow(y1, 2) + math.pow(z1, 2)) * math.sqrt(pow(x2, 2) + math.pow(y2, 2) + math.pow(z2, 2))))) * 180 / math.pi)
        else:
            angle_filter[i][2][angle_filter_count] = 0

        if joint_pointX[i][1] != 1 and joint_pointX[i][5] != 1 and joint_pointX[i][6] != 1 and joint_dist[i][1] != 0 and joint_dist[i][5] != 0 and joint_dist[i][6] != 0  : # Lshoulder 
            x1=joint_pointX[i][1] - joint_pointX[i][5]
            y1=joint_pointY[i][1] - joint_pointY[i][5]
            x2=joint_pointX[i][6] - joint_pointX[i][5]
            y2=joint_pointY[i][6] - joint_pointY[i][5]
            x3=joint_pointX[i][6]
            y3=joint_pointY[i][6]
            z1=joint_dist[i][1] - joint_dist[i][5]
            z2=joint_dist[i][6] - joint_dist[i][5]
            z3=joint_dist[i][6]
            angle_filter[i][5][angle_filter_count] = int((math.acos(((x1*x2) + (y1*y2) + (z1*z2)) / (math.sqrt(math.pow(x1, 2) + math.pow(y1, 2) + math.pow(z1, 2)) * math.sqrt(pow(x2, 2) + math.pow(y2, 2) + math.pow(z2, 2))))) * 180 / math.pi)
        else:
            angle_filter[i][5][angle_filter_count] = 0

        if joint_pointX[i][0] != 1 and joint_pointX[i][1] != 1 and joint_pointX[i][2] != 1 and joint_dist[i][0] != 0 and joint_dist[i][1] != 0 and joint_dist[i][2] != 0  : # Neck 
            x1=joint_pointX[i][0] - joint_pointX[i][1]
            y1=joint_pointY[i][0] - joint_pointY[i][1]
            x2=joint_pointX[i][2] - joint_pointX[i][1]
            y2=joint_pointY[i][2] - joint_pointY[i][1]
            x3=joint_pointX[i][2]
            y3=joint_pointY[i][2]
            z1=joint_dist[i][0] - joint_dist[i][1]
            z2=joint_dist[i][2] - joint_dist[i][1]
            z3=joint_dist[i][2]
            angle_filter[i][1][angle_filter_count] = int((math.acos(((x1*x2) + (y1*y2) + (z1*z2)) / (math.sqrt(math.pow(x1, 2) + math.pow(y1, 2) + math.pow(z1, 2)) * math.sqrt(pow(x2, 2) + math.pow(y2, 2) + math.pow(z2, 2))))) * 180 / math.pi)
            if angle_filter[i][1][angle_filter_count] > 90 :
               angle_filter[i][1][angle_filter_count] = 180 - angle_filter[i][1][angle_filter_count]
        else:
            angle_filter[i][1][angle_filter_count] = 0

        angle_filter_count = angle_filter_count + 1

        if angle_filter_count == 7:
            angle_filter_count = 0
        
    return angle_filter,angle_filter_count


if __name__ == '__main__':
    # Configure set camera data
    parser = argparse.ArgumentParser(description='SSJointTracker')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default='true',
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    # important calling  (tf-pose-estimation call)

    profile = pipeline.start(config)
    # start camera modules

    depth_sensor = profile.get_device().first_depth_sensor()
    # get depth sensor device info
    
    if depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.laser_power, float(10))
        depth_sensor.set_option(rs.option.visual_preset, float(1))
        depth_sensor.set_option(rs.option.motion_range, float(65))
        depth_sensor.set_option(rs.option.confidence_threshold, float(1))
    
    while True:

        frames=pipeline.wait_for_frames()
        # one frame reading

        image_data_rgb = frames.get_color_frame()
        image_data_depth = frames.get_depth_frame() # hide image
        # get image module data

        if not image_data_rgb or not image_data_depth:
            continue
        # empty data filtering and flow process
       
        image_rgb = np.asanyarray(image_data_rgb.get_data())
        depth_data_array = np.asanyarray(image_data_depth.get_data())
        
        # array format is numpy.
        # save your image data. in numpyarray.
        
      
        if args.zoom < 1.0:
            canvas = np.zeros_like(image_rgb)
            img_scaled = cv2.resize(image_rgb, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
            dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
            canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
            image_rgb = canvas

        elif args.zoom > 1.0:
            img_scaled = cv2.resize(image_rgb, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (img_scaled.shape[1] - image_rgb.shape[1]) // 2
            dy = (img_scaled.shape[0] - image_rgb.shape[0]) // 2
            image1 = img_scaled[dy:image1.shape[0], dx:image_rgb.shape[1]]
        humans = e.inference(image_rgb)
        image_rgb = TfPoseEstimator.draw_humans(image_rgb,humans,imgcopy=False)

        if humans:
        # distance values call (humans list length = people).
        # Nose = 0
        # Neck = 1
        # RShoulder = 2
        # RElbow = 3
        # RWrist = 4
        # LShoulder = 5
        # LElbow = 6
        # LWrist = 7
        # RHip = 8
        # RKnee = 9
        # RAnkle = 10
        # LHip = 11
        # LKnee = 12
        # LAnkle = 13
        # Background = 18

            joint_pointX, joint_pointY = TfPoseEstimator.joint_pointer(image_rgb,humans,imgcopy=False)
            print(joint_pointX)
            if len(humans) > 0 :
                for i in range(0,len(humans)):
                    joint_dist,image_rgb = multiple_depth_data(joint_pointX,joint_pointY,i,humans,depth_data_array,image_rgb)
                    angle_filter,angle_filter_count = Angle_calculator(joint_pointX,joint_pointY,joint_dist,angle_filter,angle_filter_count,humans,i)
           
            cv2.putText(image_rgb,str(getMedian(angle_filter[i][2])),(joint_pointX[0][2],joint_pointY[0][2]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
            cv2.putText(image_rgb,str(getMedian(angle_filter[i][5])),(joint_pointX[0][5],joint_pointY[0][5]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
            cv2.putText(image_rgb,str(getMedian(angle_filter[i][1])),(joint_pointX[0][1],joint_pointY[0][1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
  
        cv2.putText(image_rgb,
            "FPS: %f" % (1.0 / (time.time() - fps_time)),
            (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 2)

        res=(1280,960)
        image_rgb = cv2.resize(image_rgb,res,interpolation=cv2.INTER_AREA)
        cv2.imshow('SSJointTracker', image_rgb)
        fps_time = time.time()        

        if cv2.waitKey(1) == 27:
            pipeline.stop()
            break

        frame_add += 1
        print("frame count",frame_add," : ")
        

        print("-------")
    cv2.destroyAllWindows()
