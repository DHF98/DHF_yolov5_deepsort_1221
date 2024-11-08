import argparse
import os
import numpy as np
import cv2
import torch
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.augmentations import letterbox
from utils.torch_utils import select_device
import warnings, math
from deepsort_tracker.deepsort import DeepSort
from collections import deque

warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
# 检测参数
parser.add_argument('--weights', default=r"best4.0.pt", type=str, help='weights path')
parser.add_argument('--source', default=r"C:\Users\18155\Videos\英迪格.mp4", type=str, help='img or video(.mp4)path')
parser.add_argument('--save', default=r"./save1", type=str, help='save img or video path')
parser.add_argument('--vis', default=True, action='store_true', help='visualize image')
parser.add_argument('--device', type=str, default="0", help='use gpu or cpu')
parser.add_argument('--imgsz', type=tuple, default=(640, 640), help='image size')
parser.add_argument('--merge_nms', default=False, action='store_true', help='merge class')
parser.add_argument('--conf_thre', type=float, default=0.5, help='conf_thre')
parser.add_argument('--iou_thre', type=float, default=0.5, help='iou_thre')
parser.add_argument('--save_txt', default=r"save_txt", type=str, help='save_txt')

# 跟踪参数
parser.add_argument('--track_model', default=r"./track_models/ckpt.t7", type=str, help='track model')
parser.add_argument('--max_dist', type=float, default=0.1, help='max dist')
parser.add_argument('--min_confidence', type=float, default=0.3, help='min confidence')
parser.add_argument('--nms_max_overlap', type=float, default=1, help='nms_max_overlap')
parser.add_argument('--max_iou_distance', type=float, default=0.7, help='max_iou_distance')
parser.add_argument('--max_age', type=int, default=30, help='max_age')
parser.add_argument('--n_init', type=int, default=3, help='n_init')
parser.add_argument('--nn_budget', type=int, default=100, help='nn_budget')

opt = parser.parse_args()

# 高度
H = 6.5
alpha = 8  # 角度a
# 相机内参
calib = np.array([[2900, 0.0, 640],
                  [0.0, 2900, 360],
                  [0.0, 0.0, 1.0]])


def draw_measure_line(H, calib, u, v, alpha):
    alpha = alpha  # 角度a

    # 相机焦距
    fy = calib[1][1]
    # 相机光心
    u0 = calib[0][2]
    v0 = calib[1][2]

    pi = math.pi

    Q_pie = [u - u0, v - v0]
    gamma_pie = math.atan(Q_pie[1] / fy) * 180 / np.pi

    beta_pie = alpha + gamma_pie

    if beta_pie == 0:
        beta_pie = 1e-2

    z_in_cam = (H / math.sin(beta_pie / 180 * pi)) * math.cos(gamma_pie * pi / 180)

    return z_in_cam


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


class Detector:
    def __init__(self, device, model_path=r'./best_dist_model.pt', imgsz=(1080, 1080), conf=0.5, iou=0.0625,
                 merge_nms=False):


        self.device = device
        self.model = DetectMultiBackend(model_path, device=self.device, dnn=False)
        self.names = self.model.names

        self.stride = self.model.stride

        self.imgsz = check_img_size(imgsz, s=self.stride)

        self.conf = conf

        self.iou = iou

        self.merge_nms = merge_nms

        use_cuda = False if opt.device == "cpu" else True

        self.tracker = DeepSort(opt.track_model, max_dist=opt.max_dist, min_confidence=opt.min_confidence,
                                nms_max_overlap=opt.nms_max_overlap,
                                max_iou_distance=opt.max_iou_distance, max_age=opt.max_age, n_init=opt.n_init,
                                nn_budget=opt.nn_budget,
                                use_cuda=use_cuda)

        self.trajectories = {}

        self.max_trajectory_length = 5

        self.id_in = 0
        self.id_in_list = []

        self.id_out = 0
        self.id_out_list = []

    @torch.no_grad()
    def __call__(self, image: np.ndarray):
        img_vis = image.copy()
        img = letterbox(image, self.imgsz, stride=self.stride)[0]
        # print(img.shape)
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        im = img.float()  # uint8 to fp16/32
        im /= 255.0
        im = im[None]
        # inference
        pred = self.model(im, augment=False, visualize=False)

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=self.conf, iou_thres=self.iou, classes=None,
                                   agnostic=self.merge_nms, max_det=1000)

        point_list = []
        for i, det in enumerate(pred):  # detections per image
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], image.shape).round()
            online_targets = self.tracker.update(det[:, :6].cpu(), image)
            obj_list = []
            for t in online_targets:
                speed_km_per_h = 0
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                cls = t[5]
                obj_list.append(cls)
                color = get_color(int(cls) + 2)
                point_list.append([self.names[int(cls)], int(tlwh[0] + tlwh[2] / 2), int(tlwh[1] + tlwh[3] / 2),int(tlwh[2]),int(tlwh[3]), int(tid)])
                # center = (int(tlwh[0] + tlwh[2] / 2), int(tlwh[1] + tlwh[3] / 2))
                bottom_center = (int(tlwh[0] + tlwh[2] / 2), int(tlwh[1] + tlwh[3]))
                cv2.rectangle(img_vis, (int(tlwh[0]), int(tlwh[1])),
                              (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])), color, 2)

                zc_cam_other = draw_measure_line(H, calib, int(tlwh[0] + tlwh[2] / 2), int(tlwh[1] + tlwh[3]), alpha)

                # if self.names[int(cls)] == "person":
                if tid not in self.trajectories:
                    self.trajectories[tid] = deque(maxlen=self.max_trajectory_length)

                trajectory_point = {
                    'bottom_center': bottom_center,
                    'zc_cam_other': zc_cam_other
                }
                self.trajectories[tid].appendleft(trajectory_point)

                # 截断轨迹长度
                if len(self.trajectories[tid]) > self.max_trajectory_length:
                    self.trajectories[tid] = self.trajectories[tid][:self.max_trajectory_length]

                for i in range(1, len(self.trajectories[tid])):
                    time_interval = 1 / fps
                    speed_m_per_s = abs(self.trajectories[tid][i]['zc_cam_other'] - self.trajectories[tid][i - 1][
                        'zc_cam_other']) / time_interval
                    speed_km_per_h = speed_m_per_s * 3.6  # 转换为公里/小时

                cv2.putText(img_vis, f"{self.names[int(cls)]} {int(tid)}",
                            (int(tlwh[0]), int(tlwh[1])), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

                # 显示速度，纵坐标比类名和跟踪ID的位置稍低一些
                # cv2.putText(img_vis, f"{speed_km_per_h:.1f} km/h",
                #             (int(tlwh[0]), int(tlwh[1]) + 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)

        return img_vis, point_list


if __name__ == '__main__':
    print("开始生成数据")
    # 创建文件夹的逻辑
    existing_folders = [d for d in os.listdir(opt.save) if os.path.isdir(os.path.join(opt.save, d))]
    next_folder_number = len(existing_folders) + 1
    save_folder = os.path.join(opt.save, str(next_folder_number))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 创建文本保存的文件夹
    save_txt_folder = os.path.join(save_folder, opt.save_txt)
    if not os.path.exists(save_txt_folder):
        os.makedirs(save_txt_folder)

    device = select_device(opt.device)
    print(device)
    model = Detector(device=device, model_path=opt.weights, imgsz=opt.imgsz, conf=opt.conf_thre,
                     iou=opt.iou_thre,
                     merge_nms=opt.merge_nms)

    capture = cv2.VideoCapture(opt.source)
    frame_id = 0
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    outVideo = cv2.VideoWriter(os.path.join(opt.save, os.path.basename(opt.source).split('.')[-2] + "_out.mp4"), fourcc,
                               fps, size)

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        img_vis, point_list = model(frame)
        frame_id += 1
        with open(os.path.join(save_txt_folder, str(frame_id) + '.txt'), 'w') as outtxt:  # 使用新的保存路径
            for point in point_list:
                label, centerx, centery, width,height, tid = point[0], point[1], point[2], point[3],point[4],point[5]
                outtxt.write(
                    str(label) + ' ' + str(centerx) + ' ' + str(centery) + ' '+ str(width) + ' '+ str(height) + ' ' + str(tid) + ' ' + '\n')

        outVideo.write(img_vis)

        img_vis = cv2.resize(img_vis, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('track', img_vis)
        cv2.waitKey(30)

    capture.release()
    outVideo.release()
