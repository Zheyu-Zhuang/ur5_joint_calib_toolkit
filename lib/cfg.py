import argparse
import cv2.aruco as aruco


parser = argparse.ArgumentParser()
#
parser.add_argument('--dataset', type=str, help='directory for saving the data')
#
parser.add_argument('--aruco_bit', type=int, default=4,
                    help='format of aruco dictionary')
parser.add_argument('--board_dim', type=int, nargs="+", default=[4, 6],
                    help='width, height of checkerboard (unit: squares)')
parser.add_argument('--square_len', type=float, default=0.029,
                    help='measured in metre')
parser.add_argument('--marker_len', type=float, default=0.022,
                    help='measured in metre')
parser.add_argument('--camera_topic', type=str, default='/camera/color/image_raw')
args = parser.parse_args()

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)  # 4X4 = 4x4 bit markers
charuco_board = aruco.CharucoBoard_create(args.board_dim[0], args.board_dim[1],
                                          args.square_len, args.marker_len,
                                          aruco_dict)

