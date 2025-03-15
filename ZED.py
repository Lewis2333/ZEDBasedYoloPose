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
lock = Lock()  # 线程管理 防止出现多目标检测框的情况
run_signal = False  # 判断是否进行检测
exit_signal = False  # 判断线程是否进行
image_net = None  # 记录图片数据
detections = None  # 记录检测数据
keypoints_batch = None  # 记录关键点数据

# COCO格式关键点连接关系（17个关键点）

skeleton_human = [
    [0, 1], [0, 2], [1, 3], [2, 4],  # 鼻子到眼睛到耳朵
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # 肩膀到手腕
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],  # 臀部到脚踝
    [5, 11], [6, 12],  # 躯干连接
    [3, 5], [4, 6], # 两只眼睛到肩膀
    [1, 2] # 两只眼连接
]
'''
附：coco人体所有关键点
0	nose	鼻子
1	left_eye	左眼
2	right_eye	右眼
3	left_ear	左耳
4	right_ear	右耳
5	left_shoulder	左肩
6	right_shoulder	右肩
7	left_elbow	左手肘
8	right_elbow	右手肘
9	left_wrist	左手腕
10	right_wrist	右手腕
11	left_hip	左臀部
12	right_hip	右臀部
13	left_knee	左膝盖
14	right_knee	右膝盖
15	left_ankle	左脚踝
16	right_ankle	右脚
'''

# conf_thres 置信度
# iou_thres 交并比阈值
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
                    # 将图片BGRA格式转换为符合yolo的RGB格式
                    img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2RGB)
                    results = model.predict(img, imgsz=img_size, conf=conf_thres, iou=iou_thres, verbose=False)[0]

                    # 提取检测结果和关键点
                    detections = results.boxes.data.cpu().numpy()
                    # 三元判断 点是否为空
                    keypoints_batch = results.keypoints.data.cpu().numpy() if results.keypoints else None
            except Exception as e:
                print(f"检测过程中出现错误: {e}")
            finally:
                run_signal = False
        sleep(0.01)


# conf_threshold 代表置信度
def draw_skeleton(frame, keypoints, x1, y1, x2, y2, conf_threshold=0.2):
    # 判断是否为二维数组 若不是则说明数据有误 返回行列(17,3) 数
    if keypoints is None or len(keypoints.shape) != 2:
        return

    # 转换归一化坐标到像素坐标
    # 获取前两个元素高度 宽度 第三个元素通道数不获取
    h, w = frame.shape[:2]
    pixel_points = keypoints.copy()
    # 获取二维数组前两列 即x y 并获取最大值
    if np.max(pixel_points[:, :2]) <= 1.0:
        # 检查最大值是否归一化 若最大值都归一化则将x乘以宽度 转化为像素x坐标,同理y转化
        pixel_points[:, 0] *= w
        pixel_points[:, 1] *= h

    # 稍微扩大检测框范围 防止出现检测点飘出框的问题
    # 并防止检测框超出屏幕范围 即0和w/h 点
    expand_ratio = 0.1
    x1 = max(0, x1 - expand_ratio * (x2 - x1))
    x2 = min(w, x2 + expand_ratio * (x2 - x1))
    y1 = max(0, y1 - expand_ratio * (y2 - y1))
    y2 = min(h, y2 + expand_ratio * (y2 - y1))

    # 过滤低置信度的关键点以及不在检测框内的关键点
    valid_points = []
    for i, point in enumerate(pixel_points):
        # x y 置信度信息赋值
        x, y, conf = point
        # 判断置信度是否超过了阈值 并判断关键点是否在框内
        if conf > conf_threshold and x1 <= x <= x2 and y1 <= y <= y2:
            valid_points.append(point)
        else:
            # 添加无效点数组 x y 无效 置信度0
            valid_points.append(np.array([np.nan, np.nan, 0]))
    # 将列表转化为np数组 便于后续处理
    valid_points = np.array(valid_points)

    # 绘制连接线
    for connection in skeleton_human:
        start_idx, end_idx = connection
        # 防止下标越界 防止没检测到的点执行下面代码
        if start_idx >= len(valid_points) or end_idx >= len(valid_points):
            continue

        start_x, start_y, start_conf = valid_points[start_idx]
        end_x, end_y, end_conf = valid_points[end_idx]
        # 判断是否为合法的点
        if not np.isnan(start_x) and not np.isnan(start_y) and not np.isnan(end_x) and not np.isnan(end_y):
            # cv2.line 需要整数 所以强制转换一下
            start_pt = (int(start_x), int(start_y))
            end_pt = (int(end_x), int(end_y))
            # 置信度取最小的
            conf = min(start_conf, end_conf)
            # 置信度达到阈值即画线
            if conf > conf_threshold:
                cv2.line(frame, start_pt, end_pt, (0, 200, 0), 2, lineType=cv2.LINE_AA)

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
    # 用了8m模型比较精确 但是图像帧数比较低，如果想流畅的话就用8n轻量化模型
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

                # 保证线程安全情况下 绘制结果
                with lock:
                    if detections is not None and keypoints_batch is not None:
                        # 确保检测框和关键点组数一致 防止后续for循环数组越界
                        num_detections = len(detections)
                        num_keypoints = len(keypoints_batch)
                        num_valid = min(num_detections, num_keypoints)

                        for i in range(num_valid):
                            det = detections[i]
                            keypoints = keypoints_batch[i]
                            # 解析检测框
                            x1, y1, x2, y2, conf, cls_id = map(float, det[:6])
                            # 只处理人类别 否则跳出循环
                            if int(cls_id) != 0:
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
                                #绘制距离
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
