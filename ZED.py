import math
import numpy as np
import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
from threading import Lock, Thread
from time import sleep

# 全局变量
lock = Lock()
run_signal = False
exit_signal = False
image_net = None
detections = None
keypoints_batch = None

# COCO格式关键点连接关系（17个关键点）
skeleton_human = [
    [0, 1], [0, 2], [1, 3], [2, 4],  # 鼻子到眼睛到耳朵
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # 肩膀到手腕
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],  # 臀部到脚踝
    [5, 11], [6, 12],  # 躯干连接
    [3, 5], [4, 6]  # 新增耳朵到肩膀连线
]


def torch_thread(weights, img_size, conf_thres=0.5, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections, keypoints_batch
    try:
        model = YOLO(weights)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    while not exit_signal:
        if run_signal and image_net is not None:
            try:
                with lock:
                    img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2RGB)
                    results = model.predict(img, imgsz=img_size, conf=conf_thres, iou=iou_thres, verbose=False)[0]

                    # 提取检测结果和关键点
                    detections = results.boxes.data.cpu().numpy()
                    keypoints_batch = results.keypoints.data.cpu().numpy() if results.keypoints else None
            except Exception as e:
                print(f"检测过程中出现错误: {e}")
            finally:
                run_signal = False
        sleep(0.01)


def draw_skeleton(frame, keypoints, x1, y1, x2, y2, conf_threshold=0.2):  # 降低置信度阈值
    if keypoints is None or len(keypoints.shape) != 2:
        return

    # 转换归一化坐标到像素坐标
    h, w = frame.shape[:2]
    pixel_points = keypoints.copy()
    if np.max(pixel_points[:, :2]) <= 1.0:  # 检查是否归一化
        pixel_points[:, 0] *= w
        pixel_points[:, 1] *= h

    # 稍微扩大检测框范围
    expand_ratio = 0.1
    x1 = max(0, x1 - expand_ratio * (x2 - x1))
    x2 = min(w, x2 + expand_ratio * (x2 - x1))
    y1 = max(0, y1 - expand_ratio * (y2 - y1))
    y2 = min(h, y2 + expand_ratio * (y2 - y1))

    # 过滤低置信度的关键点以及不在检测框内的关键点
    valid_points = []
    for i, point in enumerate(pixel_points):
        x, y, conf = point
        if conf > conf_threshold and x1 <= x <= x2 and y1 <= y <= y2:
            valid_points.append(point)
        else:
            valid_points.append(np.array([np.nan, np.nan, 0]))
    valid_points = np.array(valid_points)

    # 绘制连接线
    for connection in skeleton_human:
        start_idx, end_idx = connection
        if start_idx >= len(valid_points) or end_idx >= len(valid_points):
            continue

        start_x, start_y, start_conf = valid_points[start_idx]
        end_x, end_y, end_conf = valid_points[end_idx]

        if not np.isnan(start_x) and not np.isnan(start_y) and not np.isnan(end_x) and not np.isnan(end_y):
            start_pt = (int(start_x), int(start_y))
            end_pt = (int(end_x), int(end_y))
            conf = min(start_conf, end_conf)

            if conf > conf_threshold:
                cv2.line(frame, start_pt, end_pt, (0, 200, 0), 2, lineType=cv2.LINE_AA)

    # 绘制两眼之间连线
    left_eye_idx = 1
    right_eye_idx = 2
    left_eye = valid_points[left_eye_idx]
    right_eye = valid_points[right_eye_idx]
    if not np.isnan(left_eye[0]) and not np.isnan(left_eye[1]) and not np.isnan(right_eye[0]) and not np.isnan(right_eye[1]):
        left_eye_pt = (int(left_eye[0]), int(left_eye[1]))
        right_eye_pt = (int(right_eye[0]), int(right_eye[1]))
        conf = min(left_eye[2], right_eye[2])
        if conf > conf_threshold:
            cv2.line(frame, left_eye_pt, right_eye_pt, (0, 200, 0), 2, lineType=cv2.LINE_AA)

    # 绘制关键点
    for i, (x, y, conf) in enumerate(valid_points):
        if not np.isnan(x) and not np.isnan(y) and conf > conf_threshold:
            color = (0, 0, 255) if i in [5, 6, 11, 12] else (255, 0, 0)  # 重要关节特殊颜色
            cv2.circle(frame, (int(x), int(y)), 5, color, -1, lineType=cv2.LINE_AA)


def main():
    global image_net, exit_signal, run_signal, detections, keypoints_batch

    # 初始化ZED相机
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_maximum_distance = 20

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"ZED相机初始化失败: {err}")
        return

    # 启动检测线程
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8m-pose.pt')
    parser.add_argument('--img-size', type=int, default=640)
    opt = parser.parse_args()

    capture_thread = Thread(target=torch_thread, args=(opt.weights, opt.img_size))
    capture_thread.start()

    # 初始化图像容器
    image_left = sl.Mat()
    point_cloud = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    try:
        while not exit_signal:
            # 获取新帧
            err = zed.grab(runtime_params)
            if err == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image_left, sl.VIEW.LEFT)
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

                # 准备检测数据
                with lock:
                    frame = image_left.get_data()
                    image_net = frame.copy()
                    run_signal = True

                # 等待检测完成
                while run_signal and not exit_signal:
                    sleep(0.001)

                # 绘制结果
                with lock:
                    if detections is not None and keypoints_batch is not None:
                        # 确保检测框和关键点数量一致
                        num_detections = len(detections)
                        num_keypoints = len(keypoints_batch)
                        num_valid = min(num_detections, num_keypoints)

                        for i in range(num_valid):
                            det = detections[i]
                            keypoints = keypoints_batch[i]

                            # 解析检测框
                            x1, y1, x2, y2, conf, cls_id = map(float, det[:6])
                            if int(cls_id) != 0:  # 只处理人类别
                                continue

                            # 计算中心点
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)

                            # 获取三维坐标
                            err, point = point_cloud.get_value(cx, cy)
                            if np.isfinite(point[2]):
                                distance = math.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2)
                                label = f"{distance:.2f}m"

                                # 绘制检测框
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                              (0, 255, 0), 2)
                                cv2.putText(frame, label, (int(x1) + 5, int(y1) + 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            # 绘制关键点
                            draw_skeleton(frame, keypoints, x1, y1, x2, y2)

                # 显示结果
                cv2.imshow("ZED Pose Estimation", frame)

                # 退出检测
                if cv2.waitKey(10) == 27:
                    exit_signal = True
            elif err != sl.ERROR_CODE.ERROR_CODE_TRY_AGAIN:
                print(f"获取帧时出现错误: {err}")
                break

    except Exception as e:
        print(f"主循环中出现错误: {e}")
    finally:
        # 清理资源
        zed.close()
        cv2.destroyAllWindows()
        capture_thread.join()


if __name__ == "__main__":
    with torch.no_grad():
        main()