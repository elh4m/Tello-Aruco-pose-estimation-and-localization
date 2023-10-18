import cv2 
from cv2 import aruco
import numpy as np
import time
from djitellopy import Tello
from scipy.spatial.transform import Rotation 
import math
import random


#radian*(180/np.pi)

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    # assert(isRotationMatrix(R))
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

# def calculate_T_aruco_origin(marker_positions, marker_IDs):

#     R1 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
#     R2 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
#     R3 = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
#     R4 = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
   
#     marker_id_scalar = marker_IDs.item()  # Convert ndarray to scalar integer

#     if marker_id_scalar == 0:
#         rotation_matrix = R1
#     elif marker_id_scalar in [1, 2, 3]:
#         rotation_matrix = R2
#     elif marker_id_scalar == 4:
#         rotation_matrix = R3
#     elif marker_id_scalar in [5, 6, 7]:
#         rotation_matrix = R4
#     else:
#         raise ValueError(f"No rotation matrix defined for marker ID {marker_id_scalar}")

#     marker_position = marker_positions[marker_id_scalar]

#     # Calculate R_aruco_origin
#     R_aruco_origin = np.c_[rotation_matrix, marker_position]

#     # Calculate T_aruco_origin
#     T_aruco_origin = np.r_[R_aruco_origin, [[0, 0, 0, 1]]]

#     return T_aruco_origin


calib_data_path = "/home/elham/thesis/drone_project_thesis/camera_calibration_packages/camera_calibration/tello_drone_cameracalib_checkerboard/calib_data_droneB/MultiMatrix.npz"

calib_data = np.load(calib_data_path)

print(calib_data.files)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]

MARKER_SIZE = 7.5 # centimeters

marker_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)

param_markers = aruco.DetectorParameters_create()

tello = Tello()
tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()

while True:
    frame = frame_read.frame
    
                  
    R1=np.array([[ 0 , 0 , 1],
                 [ 1 , 0 ,0],
                 [0 , 1, 0]])
    R2 = np.array([[1, 0, 0],
                  [0, 0, -1], 
                  [0, 1, 0]])
    R3 = np.array([[0, 0, -1],
                    [-1, 0, 0], 
                    [0, 1, 0]])
    R4 = np.array([[-1, 0, 0],
                    [0, 0, 1], 
                    [0, 1, 0]])

    R_manual_aruco_cam=np.array([[1, 0, 0],
                    [0, -1, 0], 
                    [0, 0, -1]])
    #data for home experiment
#     marker_dictionary = {
#     0: {
#         "position": [-15, 30, 0],
#         "rotation_matrix": R1
#     },
#     1: {
#         "position": [0, 75, 0],
#         "rotation_matrix": R2
#     },
#     2: {
#         "position": [1.0, 2.0, 3.0],
#         "rotation_matrix": R2
#     },
#     3: {
#         "position": [1.0, 2.0, 3.0],
#         "rotation_matrix": R2
#     },
#     4: {
#         "position": [105, 30, 0],
#         "rotation_matrix": R3
#     },
#     5: {
#         "position": [4.0, 5.0, 6.0],
#         "rotation_matrix": R4
#     },
#     6: {
#         "position": [7.0, 8.0, 9.0],
#         "rotation_matrix": R4
#     },
#     7: {
#         "position": [60, -15, 0],
#         "rotation_matrix": R4
#     }

# }
    # R1=cv2.transpose(R1)
    #data for lab experiment
    marker_dictionary = {
        0: {
            "position": [-15,60, 0],
            "rotation_matrix": R1
        
        },
        8: {
            "position": [-30,0, 0],
            "rotation_matrix": R1
            },
        1: {
            "position": [0, 120, 0],
            "rotation_matrix": R2
        },
        2: {
            "position": [60, 120, 0],
            "rotation_matrix": R2
        },
        3: {
            "position": [120, 120, 0],
            "rotation_matrix": R2
        },
        4: {
            "position": [180, 120, 0],
            "rotation_matrix": R2
        },
        5: {
            "position": [210, 60, 6.0],
            "rotation_matrix": R3
        },
        6: {
            "position": [120, -30, 0],
            "rotation_matrix": R4
        },
        7: {
            "position": [60, -30, 0],
            "rotation_matrix": R4
        }

    }

    while True:
        frame = frame_read.frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_IDs, reject = aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    
    )
    # Check if marker_IDs is not None
        if marker_IDs is not None:
            for marker_ID in marker_IDs:
                if all(1<= marker_ID <= 4 or 6<= marker_ID <=9):
                    first_id = min(marker_IDs)
                    # first_id=int(first_id)
                    first_id_index = marker_IDs.tolist().index(first_id)
                    
                    # Get the corresponding corners and continue processing
                    first_id_corners = marker_corners[first_id_index]
                    first_id=int(first_id)
            # if marker_IDs is not None:
            #     for marker_ID in marker_IDs:
            #         # Convert the marker ID to an appropriate key type
            #         marker_ID = int(marker_IDs[0])
                elif marker_ID==0 or marker_ID==5:
                    first_id=marker_ID
                    first_id_index = marker_IDs.tolist().index(first_id)
                    
                    # Get the corresponding corners and continue processing
                    first_id_corners = marker_corners[first_id_index]
                    # Access the dictionary using the converted marker ID
                    first_id=int(first_id)
                if first_id in marker_dictionary:
                    marker_info = marker_dictionary[first_id]
                    rotation_matrix = marker_info["rotation_matrix"]
                    position = marker_info["position"]
                # Calculate R_aruco_origin
                    R_aruco_origin = np.c_[rotation_matrix,position]

                    # Calculate T_aruco_origin
                    T_aruco_origin = np.r_[R_aruco_origin, [[0, 0, 0, 1]]]
                    print(f" aruco origin {T_aruco_origin}")
                    #T_aruco_origin=calculate_T_aruco_origin(marker_positions,marker_IDs)
                                
                else:
                    print(f"Marker {first_id} not found in the dictionary.")

                    break
        else:
            print("No markers detected. Waiting for another marker...")
            time.sleep(2)
            break 
        

        if marker_corners:
            rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
               first_id_corners, MARKER_SIZE, cam_mat, dist_coef
            )
            
            print(f"the rotation vector values are {rVec} /n") 
            # print(f"the translation vector values are {tVec} /n") 
            total_markers = range(0, marker_IDs.size)
            print(f"marker ids list:{marker_IDs}")
            print(f"marker corners list:{marker_corners}")
            print(f"size rvec:{rVec.size}")

            print(f"total markers:{total_markers}")
            for ids, corners, i in zip(marker_IDs, marker_corners, range(len(marker_IDs))):
                cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA)
            
                corners = corners.reshape(4, 2)
                corners = corners.astype(int)
                top_right = corners[0].ravel()
                top_left = corners[1].ravel()
                bottom_right = corners[2].ravel()
                bottom_left = corners[3].ravel()
            
                ######################## calculate the transformation matrix T_aruco_cam  ################
                
                R_aruco_cam_original, _ = cv2.Rodrigues(rVec)
                # print(f"R_aruco_cam {R_aruco_cam_original}")

                tVec_reshaped = tVec[0].reshape(3, 1)
                R_aruco_cam_new1 = np.hstack((R_aruco_cam_original, tVec_reshaped))
                T_aruco_cam_original = np.vstack((R_aruco_cam_new1, [[0, 0, 0, 1]]))

                T_cam_aruco = np.linalg.inv(T_aruco_cam_original)
                np.set_printoptions(suppress=True, precision=6)
                print(f"T_cam_aruco:{T_cam_aruco}/n")
                # np.set_printoptions(suppress=True, precision=6)
                # print(f"T_aruco_cam_original2:{T_aruco_cam_original}/n")
            

                #########calculating the transformation between image frame and robot frame
                r_image_robot=np.array([[0, -1, 0],
                        [0, 0, -1], 
                        [1, 0, 0]])
                t_image_robot=np.array([[0], [0], [0]])
                r_image_robot = np.hstack((r_image_robot, t_image_robot))
                T_image_robot = np.vstack((r_image_robot, [[0, 0, 0, 1]]))
                ##########calculate T_camera_origin########
                # T_cam_origin_original=np.matmul(T_aruco_origin,T_aruco_cam_original)
                # T_cam_origin=np.matmul(T_aruco_origin,T_cam_aruco) #first correct calculation
                T_cam_robot=np.matmul(T_cam_aruco,T_image_robot)
                T_cam_origin=np.matmul(T_aruco_origin,T_cam_robot)
                np.set_printoptions(suppress=True, precision=6)
                print(f"T_cam_origin : {T_cam_origin}/n")
                t_vector_new= T_cam_origin[:3, 3].reshape(3, 1)#stands for inverse
                # print(f"t_vector_cam_origin: {t_vector_new}/n")
                # rotation_matrix_o = T_cam_origin_original[:3, :3]
                rotation_matrix= T_cam_origin[:3, :3]
                rotation_matrix2= T_cam_aruco[:3, :3]
                rotation_matrix3= T_aruco_origin[:3, :3]
                rotation_matrix4= T_aruco_cam_original[:3, :3]

                
        
                #extracting the angles between two elements from rotation matrix
                
            
                # r =  Rotation.from_matrix(R1)
                # angles = r.as_euler("zyx",degrees=True)
                # print(f"agles for aruco_origin:{angles}")

                # r =  Rotation.from_matrix(R_manual_aruco_cam)
                # angles = r.as_euler("zyx",degrees=True)
                # print(f"R_manual_aruco_cam:{angles}")

                # r =  Rotation.from_matrix(rotation_matrix)
                # angles = r.as_euler("zyx",degrees=True)
                # print(f"agles for cam_origin:{angles}")

                # r =  Rotation.from_matrix(rotation_matrix2)
                # angles = r.as_euler("zyx",degrees=True)
                # print(f"agles for cam_aruco:{angles}")

                # r =  Rotation.from_matrix(rotation_matrix4)
                # angles = r.as_euler("zyx",degrees=True)
                # print(f"agles for aruco_cam:{angles}")

                # r =  Rotation.from_matrix(rotation_matrix3)
                # angles = r.as_euler("zyx",degrees=True)
                # print(f"agles for aruco_origin:{angles}")
                ####################################                                                                                                                                                                                                                                              

                # distance = np.sqrt(tVec[0][2] ** 2 + tVec[0][0] ** 2 + tVec[0][1] ** 2)
                
                # Draw the pose of the marker
               
                # point = cv2.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i][0][1], tVec[i][0][1], 4, 4)
                point = cv2.drawFrameAxes(frame, cam_mat, dist_coef, rVec[0], tVec[0], 4, 4)
                
                # cv2.putText(
                #     frame,
                #     f"id: {ids[0]} Dist: {round(distance, 2)}",#distance between cam and aruco
                #     top_right,
                #     cv2.FONT_HERSHEY_PLAIN,
                #     1.3,
                #     (0, 0, 255),
                #     2,
                #     cv2.LINE_AA,
                # )
                cv2.putText(
                    frame,
                    #f"x:{round(tVec[i][0][0],1)} y: {round(tVec[i][0][1],1)} ",
                    # f"x_cam:{round(t_vector[0][0],1)} y_cam: {round(t_vector[1][0],1)}",
                    f"x_cam:{round(t_vector_new[0][0],1)} y_cam: {round(t_vector_new[1][0],1)}",
                    
                    bottom_right,
                    cv2.FONT_HERSHEY_PLAIN,
                    1.0,#The font scale (size) of the textq
                    (0, 0, 255),# The color of the text (in BGR format) 
                    2,#The thickness of the text
                    cv2.LINE_AA,)
                    
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()



