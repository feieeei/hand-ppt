import cv2
import mediapipe as mp
import pyautogui
import time
# 配置基础选项
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 载入下载的模型文件
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)

# 手写骨架连接线 
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 大拇指
    (0, 5), (5, 6), (6, 7), (7, 8),        # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),   # 中指
    (9, 13), (13, 14), (14, 15), (15, 16), # 无名指
    (13, 17), (17, 18), (18, 19), (19, 20),# 小指
    (0, 17)                                # 手掌根部连接
]

# ================= 2. 防误触状态变量 =================
COOLDOWN_TIME = 1.5      
SWIPE_THRESHOLD = 50     
last_trigger_time = 0    
start_x = None           

# ================= 3. 主程序循环 =================
cap = cv2.VideoCapture(0)
print("请在画面上半部分，张开手掌左右滑动进行翻页。按 'q' 退出。")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1) 
    h, w, c = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 将图像转换为新版 API 需要的格式
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    # 核心：使用新版 detect 方法提取骨骼数据
    detection_result = landmarker.detect(mp_image)

    # 绘制触发区
    active_zone_y = int(h * 0.6)
    cv2.line(img, (0, active_zone_y), (w, active_zone_y), (255, 0, 0), 2)
    cv2.putText(img, "Recognition line", (10, active_zone_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 如果检测到了手
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            
            # 提取21个关键点的像素坐标
            lm_list = []
            for id, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                # 画关节圆点
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
            
            # 画骨架连线
            for connection in HAND_CONNECTIONS:
                pt1 = (lm_list[connection[0]][1], lm_list[connection[0]][2])
                pt2 = (lm_list[connection[1]][1], lm_list[connection[1]][2])
                cv2.line(img, pt1, pt2, (0, 255, 0), 2)

            # --- 下面是原汁原味的翻页逻辑 ---
            center_x, center_y = lm_list[9][1], lm_list[9][2]

            if center_y < active_zone_y:
                fingers_up = 0
                finger_tips = [8, 12, 16, 20] 
                for tip in finger_tips:
                    if lm_list[tip][2] < lm_list[tip - 2][2]:
                        fingers_up += 1

                if fingers_up >= 3: # 手掌张开
                    if start_x is None:
                        start_x = center_x 
                    else:
                        diff_x = center_x - start_x
                        current_time = time.time()

                        if current_time - last_trigger_time > COOLDOWN_TIME:
                            if diff_x > SWIPE_THRESHOLD: 
                                print("--> 下一页 (Next)")
                                pyautogui.press('right') 
                                last_trigger_time = current_time 
                                start_x = None           
                                cv2.putText(img, "NEXT ->", (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                            
                            elif diff_x < -SWIPE_THRESHOLD:
                                print("<-- 上一页 (Prev)")
                                pyautogui.press('left')  
                                last_trigger_time = current_time 
                                start_x = None           
                                cv2.putText(img, "<- PREV", (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                else:
                    start_x = None
            else:
                start_x = None

    cv2.imshow("PPT Controller - New API", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()