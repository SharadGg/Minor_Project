# # """
# # webcam_action_detector.py

# # Requirements:
# #     pip install opencv-python mediapipe

# # What it does:
# #     - Opens webcam
# #     - Uses MediaPipe Holistic (pose + hands)
# #     - Draws a bounding box around the person (computed from pose landmarks)
# #     - Displays FPS in top-left
# #     - Displays a simple action label in top-right: "Punch", "Point", "Hands Up", or "Unknown"
# #     - Draws landmarks for debugging/visual feedback

# # Notes:
# #     - This is rule-based (heuristics) for demonstration. For production-grade
# #       action recognition use a trained classifier over temporal features.
# # """

# # import time
# # import math
# # import cv2
# # import mediapipe as mp

# # # ---------- Config / thresholds (tweak if needed) ----------
# # FPS_SMOOTHING = 0.9
# # PUNCH_Z_DIFF = 0.12        # how much wrist z must be closer to camera than shoulder z
# # POINT_INDEX_EXTENDED = 0.06 # normalized distance index_tip - index_pip relative to image diagonal
# # HANDS_UP_Y_DIFF = 0.10     # wrist y must be significantly above shoulder y (normalized)
# # # -----------------------------------------------------------

# # mp_holistic = mp.solutions.holistic
# # mp_drawing = mp.solutions.drawing_utils
# # mp_pose = mp.solutions.pose

# # def norm_dist(a, b):
# #     """Euclidean distance between two normalized (x,y) tuples."""
# #     return math.hypot(a[0] - b[0], a[1] - b[1])

# # def detect_action(landmarks, handedness, image_shape):
# #     """
# #     landmarks: holistic landmarks object (contains pose_landmarks and hand_landmarks)
# #     handedness: tuple with left/right info (not always needed)
# #     image_shape: (h, w)
# #     returns: string label
# #     """

# #     h, w = image_shape
# #     pose = landmarks.pose_landmarks
# #     left_hand = landmarks.left_hand_landmarks
# #     right_hand = landmarks.right_hand_landmarks

# #     # If no pose, cannot decide
# #     if not pose:
# #         return "No person"

# #     # convenience helper to fetch pose landmark coordinates as normalized tuples
# #     def lm_pose(i):
# #         lm = pose.landmark[i]
# #         return (lm.x, lm.y, lm.z, lm.visibility)

# #     # indices from MediaPipe Pose
# #     LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
# #     RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
# #     LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value
# #     RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST.value
# #     NOSE = mp_pose.PoseLandmark.NOSE.value

# #     # get shoulders and wrists
# #     l_sh = lm_pose(LEFT_SHOULDER)
# #     r_sh = lm_pose(RIGHT_SHOULDER)
# #     l_wr = lm_pose(LEFT_WRIST)
# #     r_wr = lm_pose(RIGHT_WRIST)

# #     # Basic "hands up" detection: if either wrist y is above (less than) shoulder y by threshold
# #     # Note: y is normalized 0..1 top..bottom
# #     hands_up = False
# #     if l_wr[1] < l_sh[1] - HANDS_UP_Y_DIFF or r_wr[1] < r_sh[1] - HANDS_UP_Y_DIFF:
# #         hands_up = True

# #     # Punch detection using z (towards camera -> smaller z values in MediaPipe)
# #     # If wrist z is significantly less (more forward) than same-side shoulder z => punching forward
# #     punch = False
# #     # check left punch
# #     if l_wr[2] < l_sh[2] - PUNCH_Z_DIFF:
# #         punch = True
# #     # check right punch
# #     if r_wr[2] < r_sh[2] - PUNCH_Z_DIFF:
# #         punch = True

# #     # Point detection uses hand landmarks: index finger extended while other fingers less extended.
# #     # We'll check for either hand.
# #     point = False
# #     diag = math.hypot(h, w)
# #     # helper for hand landmarks (MediaPipe hands: 0=WRIST, 5=index_mcp, 6=index_pip, 8=index_tip, 12=middle_tip etc.)
# #     def hand_is_pointing(hand_landmarks):
# #         # returns True if index finger noticeably extended compared to others
# #         try:
# #             # normalized coords (x,y)
# #             n = lambda i: (hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y)
# #             wrist = n(0)
# #             idx_tip = n(8)
# #             idx_pip = n(6)
# #             mid_tip = n(12)
# #             ring_tip = n(16)
# #             pinky_tip = n(20)

# #             # distance wrist->index_tip normalized by image diag
# #             wrist_idx_dist = math.hypot((idx_tip[0]-wrist[0])*w, (idx_tip[1]-wrist[1])*h) / diag
# #             index_extended = (math.hypot((idx_tip[0]-idx_pip[0])*w, (idx_tip[1]-idx_pip[1])*h) / diag) > POINT_INDEX_EXTENDED
# #             # check other fingers are not extended as much
# #             mid_ext = (math.hypot((mid_tip[0]-wrist[0])*w, (mid_tip[1]-wrist[1])*h) / diag)
# #             ring_ext = (math.hypot((ring_tip[0]-wrist[0])*w, (ring_tip[1]-wrist[1])*h) / diag)
# #             pinky_ext = (math.hypot((pinky_tip[0]-wrist[0])*w, (pinky_tip[1]-wrist[1])*h) / diag)

# #             # index noticeably more extended than middle+ring+pinky (heuristic)
# #             others_avg = (mid_ext + ring_ext + pinky_ext) / 3.0
# #             if index_extended and wrist_idx_dist > others_avg + 0.02:
# #                 return True
# #             return False
# #         except Exception:
# #             return False

# #     if left_hand:
# #         if hand_is_pointing(left_hand):
# #             point = True
# #     if right_hand:
# #         if hand_is_pointing(right_hand):
# #             point = True

# #     # Priority: Punch > Point > Hands Up > Unknown
# #     if punch:
# #         return "Punch"
# #     if point:
# #         return "Point"
# #     if hands_up:
# #         return "Hands Up"
# #     return "Unknown"


# # def main():
# #     cap = cv2.VideoCapture(0)
# #     if not cap.isOpened():
# #         print("Could not open webcam.")
# #         return

# #     # To compute FPS
# #     prev_time = time.time()
# #     fps = 0.0

# #     # Initialize MediaPipe Holistic
# #     with mp_holistic.Holistic(
# #         static_image_mode=False,
# #         model_complexity=1,
# #         enable_segmentation=False,
# #         refine_face_landmarks=False,
# #         min_detection_confidence=0.5,
# #         min_tracking_confidence=0.5,
# #     ) as holistic:

# #         while True:
# #             ret, frame = cap.read()
# #             if not ret:
# #                 break

# #             # Flip for natural (mirror) view:
# #             frame = cv2.flip(frame, 1)
# #             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #             h, w, _ = frame.shape

# #             results = holistic.process(frame_rgb)

# #             # Draw pose and hands landmarks (optional)
# #             if results.pose_landmarks:
# #                 mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
# #                                           mp_drawing.DrawingSpec(color=(80,110,10), thickness=2, circle_radius=2),
# #                                           mp_drawing.DrawingSpec(color=(80,256,121), thickness=2, circle_radius=2))
# #             if results.left_hand_landmarks:
# #                 mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
# #             if results.right_hand_landmarks:
# #                 mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# #             # Compute bounding box from pose landmarks if available
# #             bbox = None
# #             if results.pose_landmarks:
# #                 xs = [lm.x for lm in results.pose_landmarks.landmark if (lm.visibility if hasattr(lm,'visibility') else 1.0) > 0.1]
# #                 ys = [lm.y for lm in results.pose_landmarks.landmark if (lm.visibility if hasattr(lm,'visibility') else 1.0) > 0.1]
# #                 if xs and ys:
# #                     x_min = int(max(0, min(xs) * w) - 10)
# #                     x_max = int(min(w - 1, max(xs) * w) + 10)
# #                     y_min = int(max(0, min(ys) * h) - 10)
# #                     y_max = int(min(h - 1, max(ys) * h) + 10)
# #                     bbox = (x_min, y_min, x_max, y_max)
# #                     # draw box
# #                     cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# #             # Detect action with heuristics
# #             label = detect_action(results, None, (h, w))

# #             # Compute FPS (smoothed)
# #             now = time.time()
# #             instantaneous_fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
# #             prev_time = now
# #             fps = FPS_SMOOTHING * fps + (1 - FPS_SMOOTHING) * instantaneous_fps

# #             # Put FPS in top-left
# #             fps_text = f"FPS: {int(fps)}"
# #             cv2.rectangle(frame, (5, 5), (150, 35), (0,0,0), -1)  # background box for readability
# #             cv2.putText(frame, fps_text, (10, 28),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

# #             # Put action label in top-right
# #             label_text = f"Action: {label}"
# #             # compute size to draw rectangle
# #             (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
# #             top_right_x = w - text_w - 20
# #             cv2.rectangle(frame, (top_right_x - 10, 5), (w - 5, 35), (0,0,0), -1)
# #             cv2.putText(frame, label_text, (top_right_x, 28),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

# #             # Show bounding-box label if present
# #             if bbox:
# #                 x_min, y_min, x_max, y_max = bbox
# #                 # optionally show confidence or simple 'Person' label
# #                 cv2.putText(frame, "Person", (x_min, y_min - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

# #             cv2.imshow('Webcam Action & FPS', frame)
# #             key = cv2.waitKey(1) & 0xFF
# #             if key == 27 or key == ord('q'):  # ESC or q to quit
# #                 break

# #     cap.release()
# #     cv2.destroyAllWindows()

# # if __name__ == "__main__":
# #     main()


# """
# webcam_action_detector_extended.py

# Requirements:
#     pip install opencv-python mediapipe

# Features:
#     - Webcam input
#     - MediaPipe Holistic for pose + hands
#     - Draw bounding box around person
#     - Show FPS (top-left)
#     - Show detected action (top-right). Heuristic action labels:
#         "Punch", "Point", "Hands Up", "Both Hands Up", "Wave", "Clap", "Fist", "Unknown", "No person"
#     - Uses a short history buffer to detect temporal actions like Wave and Clap.
# Notes:
#     - Heuristics work best under good lighting and frontal camera view.
#     - Tweak thresholds in CONFIG as needed for your camera/subject.
# """

# import time
# import math
# from collections import deque
# import cv2
# import mediapipe as mp

# # -------- CONFIG / thresholds (tweak to suit your environment) ----------
# HISTORY_LEN = 12                # number of frames to keep for temporal heuristics
# WAVE_MIN_PEAKS = 2              # require at least this many direction changes to call a wave
# WAVE_MIN_AMPLITUDE = 0.04       # normalized x-amplitude to count as wave
# CLAP_DIST_THRESHOLD = 0.12      # normalized (0..1) distance between hand centers for clap
# FIST_FINGER_DISTANCE = 0.07     # normalized average fingertip->wrist distance for closed fist
# PUNCH_Z_DIFF = 0.12             # z difference (wrist forward relative to shoulder) to call punch
# HANDS_UP_Y_DIFF = 0.08          # y difference (wrist higher than shoulder) to call hands-up
# POINT_INDEX_EXTENDED = 0.06     # index finger extension threshold
# VISIBILITY_MIN = 0.3            # minimum visibility for pose landmarks to be considered
# FPS_SMOOTHING = 0.9
# # ------------------------------------------------------------------------

# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# # temporal buffers for left/right wrist x positions and inter-hand distances
# class TemporalBuffers:
#     def __init__(self, maxlen=HISTORY_LEN):
#         self.left_x = deque(maxlen=maxlen)
#         self.right_x = deque(maxlen=maxlen)
#         self.left_xy = deque(maxlen=maxlen)   # (x,y)
#         self.right_xy = deque(maxlen=maxlen)
#         self.hands_dist = deque(maxlen=maxlen) # euclidean normalized dist between hands
#         self.times = deque(maxlen=maxlen)

#     def push(self, now, left_xy, right_xy):
#         # left_xy/right_xy are either (x,y) normalized or None
#         self.times.append(now)
#         if left_xy is not None:
#             self.left_x.append(left_xy[0])
#             self.left_xy.append(left_xy)
#         else:
#             self.left_x.append(None)
#             self.left_xy.append(None)
#         if right_xy is not None:
#             self.right_x.append(right_xy[0])
#             self.right_xy.append(right_xy)
#         else:
#             self.right_x.append(None)
#             self.right_xy.append(None)

#         # compute hands distance if both present
#         if left_xy is not None and right_xy is not None:
#             dx = (left_xy[0] - right_xy[0])
#             dy = (left_xy[1] - right_xy[1])
#             dist = math.hypot(dx, dy)  # normalized (since landmarks are 0..1)
#             self.hands_dist.append(dist)
#         else:
#             self.hands_dist.append(None)

# # helper: Euclidean distance between two normalized 2D points
# def norm_dist2(a, b):
#     return math.hypot(a[0]-b[0], a[1]-b[1])

# # helper: get pose landmark safely
# def get_pose_lm(landmarks, idx):
#     try:
#         lm = landmarks.landmark[idx]
#         return (lm.x, lm.y, lm.z, lm.visibility)
#     except Exception:
#         return None

# # detect if a hand's index finger is extended more than a threshold
# def hand_is_pointing(hand_landmarks, image_diag):
#     try:
#         # MediaPipe HAND landmarks indices: 0=wrist, 5=index_mcp, 6=index_pip, 8=index_tip, 12=middle_tip...
#         n = lambda i: (hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y)
#         wrist = n(0)
#         idx_tip = n(8)
#         idx_pip = n(6)
#         # normalized pixel distances (use diag to scale)
#         idx_seg = math.hypot((idx_tip[0]-idx_pip[0]), (idx_tip[1]-idx_pip[1]))
#         # simply compare index extension to threshold
#         return idx_seg > POINT_INDEX_EXTENDED
#     except Exception:
#         return False

# # detect closed fist by average fingertip->wrist small
# def hand_is_fist(hand_landmarks):
#     try:
#         n = lambda i: (hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y)
#         wrist = n(0)
#         tips = [n(i) for i in (8, 12, 16, 20)]  # index, middle, ring, pinky tips
#         dists = [math.hypot(t[0]-wrist[0], t[1]-wrist[1]) for t in tips]
#         avg = sum(dists)/len(dists)
#         return avg < FIST_FINGER_DISTANCE
#     except Exception:
#         return False

# # detect clap: hands come close together (uses recent history)
# def detect_clap(buff: TemporalBuffers):
#     # if any recent hands_dist below threshold -> clap
#     for d in reversed(buff.hands_dist):
#         if d is None:
#             continue
#         if d < CLAP_DIST_THRESHOLD:
#             return True
#     return False

# # detect wave: check for alternating direction changes in wrist x over history with amplitude
# def detect_wave_from_series(x_series):
#     # x_series: deque of normalized x positions (or None). We need at least 4 valid points
#     vals = [x for x in x_series if x is not None]
#     if len(vals) < 5:
#         return False
#     # compute diffs and signs
#     diffs = [vals[i+1]-vals[i] for i in range(len(vals)-1)]
#     signs = [1 if d>0 else (-1 if d<0 else 0) for d in diffs]
#     # count sign changes ignoring zeros
#     prev = None
#     changes = 0
#     for s in signs:
#         if s == 0:
#             continue
#         if prev is None:
#             prev = s
#         else:
#             if s != prev:
#                 changes += 1
#                 prev = s
#     # amplitude: max-min in vals
#     amp = max(vals)-min(vals)
#     return (changes >= WAVE_MIN_PEAKS) and (amp >= WAVE_MIN_AMPLITUDE)

# # combined action detector using heuristics and the temporal buffers
# def detect_action(results, buff: TemporalBuffers, image_shape):
#     h, w = image_shape
#     pose = results.pose_landmarks
#     left_hand = results.left_hand_landmarks
#     right_hand = results.right_hand_landmarks

#     if not pose:
#         return "No person"

#     # convenience to get pose coords
#     def p(i):
#         lm = pose.landmark[i]
#         return (lm.x, lm.y, lm.z, lm.visibility)

#     # key indices
#     L_SH, R_SH, L_WR, R_WR = (mp_pose.PoseLandmark.LEFT_SHOULDER.value,
#                                mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
#                                mp_pose.PoseLandmark.LEFT_WRIST.value,
#                                mp_pose.PoseLandmark.RIGHT_WRIST.value)

#     left_sh = p(L_SH)
#     right_sh = p(R_SH)
#     left_wr = p(L_WR)
#     right_wr = p(R_WR)

#     # hands up checks
#     left_hand_up = (left_wr[1] < left_sh[1] - HANDS_UP_Y_DIFF) and (left_wr[3] > VISIBILITY_MIN)
#     right_hand_up = (right_wr[1] < right_sh[1] - HANDS_UP_Y_DIFF) and (right_wr[3] > VISIBILITY_MIN)

#     both_hands_up = left_hand_up and right_hand_up
#     any_hand_up = left_hand_up or right_hand_up

#     # punch detection by z: wrist forward (smaller z) relative to same-side shoulder
#     punch = False
#     if left_wr[2] < left_sh[2] - PUNCH_Z_DIFF and left_wr[3] > VISIBILITY_MIN:
#         punch = True
#     if right_wr[2] < right_sh[2] - PUNCH_Z_DIFF and right_wr[3] > VISIBILITY_MIN:
#         punch = True

#     # point detection via either hand
#     point = False
#     if left_hand and hand_is_pointing(left_hand, math.hypot(h,w)):
#         point = True
#     if right_hand and hand_is_pointing(right_hand, math.hypot(h,w)):
#         point = True

#     # fist detection
#     fist = False
#     if left_hand and hand_is_fist(left_hand):
#         fist = True
#     if right_hand and hand_is_fist(right_hand):
#         fist = True

#     # wave detection from temporal buffers
#     left_wave = detect_wave_from_series(buff.left_x)
#     right_wave = detect_wave_from_series(buff.right_x)
#     wave = left_wave or right_wave

#     # clap detection via hands distance history
#     clap = detect_clap(buff)

#     # Priority order: Clap > Punch > Point > Wave > Both Hands Up > Hands Up > Fist > Unknown
#     if clap:
#         return "Clap"
#     if punch:
#         return "Punch"
#     if point:
#         return "Point"
#     if wave:
#         return "Wave"
#     if both_hands_up:
#         return "Both Hands Up"
#     if any_hand_up:
#         return "Hands Up"
#     if fist:
#         return "Fist"
#     return "Unknown"

# def main():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Could not open webcam.")
#         return

#     prev_time = time.time()
#     fps = 0.0
#     buffers = TemporalBuffers(maxlen=HISTORY_LEN)

#     with mp_holistic.Holistic(
#         static_image_mode=False,
#         model_complexity=1,
#         enable_segmentation=False,
#         refine_face_landmarks=False,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5,
#     ) as holistic:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame = cv2.flip(frame, 1)  # mirror
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             h, w, _ = frame.shape

#             results = holistic.process(frame_rgb)

#             # draw landmarks
#             if results.pose_landmarks:
#                 mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
#             if results.left_hand_landmarks:
#                 mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#             if results.right_hand_landmarks:
#                 mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

#             # compute bounding box from pose landmarks
#             bbox = None
#             if results.pose_landmarks:
#                 xs = [lm.x for lm in results.pose_landmarks.landmark if (lm.visibility if hasattr(lm,'visibility') else 1.0) > 0.1]
#                 ys = [lm.y for lm in results.pose_landmarks.landmark if (lm.visibility if hasattr(lm,'visibility') else 1.0) > 0.1]
#                 if xs and ys:
#                     x_min = int(max(0, min(xs) * w) - 10)
#                     x_max = int(min(w - 1, max(xs) * w) + 10)
#                     y_min = int(max(0, min(ys) * h) - 10)
#                     y_max = int(min(h - 1, max(ys) * h) + 10)
#                     bbox = (x_min, y_min, x_max, y_max)
#                     cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#             # update temporal buffers with current wrist positions if available
#             now = time.time()
#             left_xy = None
#             right_xy = None
#             try:
#                 if results.pose_landmarks:
#                     lw = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
#                     rw = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
#                     left_xy = (lw.x, lw.y) if lw.visibility > 0.0 else None
#                     right_xy = (rw.x, rw.y) if rw.visibility > 0.0 else None
#             except Exception:
#                 pass
#             buffers.push(now, left_xy, right_xy)

#             label = detect_action(results, buffers, (h, w))

#             # compute FPS (smoothed)
#             now2 = time.time()
#             instantaneous_fps = 1.0 / (now2 - prev_time) if now2 != prev_time else 0.0
#             prev_time = now2
#             fps = FPS_SMOOTHING * fps + (1 - FPS_SMOOTHING) * instantaneous_fps

#             # Draw FPS top-left
#             fps_text = f"FPS: {int(fps)}"
#             cv2.rectangle(frame, (5,5), (150,35), (0,0,0), -1)
#             cv2.putText(frame, fps_text, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

#             # Draw Action label top-right
#             label_text = f"Action: {label}"
#             (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
#             top_right_x = w - text_w - 20
#             cv2.rectangle(frame, (top_right_x-10,5), (w-5,35), (0,0,0), -1)
#             cv2.putText(frame, label_text, (top_right_x,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

#             # optional small indicator of left/right hand distances (helpful debug)
#             # show last hands distance if available
#             last_hand_dist = next((d for d in reversed(buffers.hands_dist) if d is not None), None)
#             if last_hand_dist is not None:
#                 dist_text = f"HandsDist: {last_hand_dist:.2f}"
#                 cv2.putText(frame, dist_text, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

#             if bbox:
#                 x_min, y_min, x_max, y_max = bbox
#                 cv2.putText(frame, "Person", (x_min, y_min-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

#             cv2.imshow('Webcam Action Detector (extended)', frame)
#             key = cv2.waitKey(1) & 0xFF
#             if key == 27 or key == ord('q'):
#                 break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


"""
code_fixed_improved.py

Webcam action + 33-point pose skeleton display (MediaPipe Holistic + OpenCV)
Includes heuristic action detection (Punch, Point, Wave, Clap, Hands Up, Both Hands Up, Fist).
Compatibility improvements and robustness fixes (visibility checks, constants, FPS guard).

Requires:
    pip install opencv-python mediapipe

Run:
    python code_fixed_improved.py
Press:
    q or ESC -> quit
    p -> pause/resume
    l -> toggle landmark labels (indices/names)
"""

# import time
# import math
# from collections import deque
# import cv2
# import mediapipe as mp

# # ---------------- CONFIG ----------------
# HISTORY_LEN = 12
# WAVE_MIN_PEAKS = 2
# WAVE_MIN_AMPLITUDE = 0.04
# CLAP_DIST_THRESHOLD = 0.12
# FIST_FINGER_DISTANCE = 0.07
# PUNCH_Z_DIFF = 0.12
# HANDS_UP_Y_DIFF = 0.08
# POINT_INDEX_EXTENDED = 0.06
# VISIBILITY_MIN = 0.3
# FPS_SMOOTHING = 0.9
# # SHOW landmark indices & names (set False to reduce clutter)
# SHOW_LANDMARK_INDICES = True
# SHOW_LANDMARK_NAMES = True
# LANDMARK_FONT_SCALE = 0.45
# LANDMARK_FONT_THICKNESS = 1
# # ----------------------------------------

# mp_holistic = mp.solutions.holistic
# mp_pose = mp.solutions.pose
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# class TemporalBuffers:
#     """Temporal storage of recent wrist x positions and inter-hand distances"""
#     def __init__(self, maxlen=HISTORY_LEN):
#         self.left_x = deque(maxlen=maxlen)
#         self.right_x = deque(maxlen=maxlen)
#         self.left_xy = deque(maxlen=maxlen)
#         self.right_xy = deque(maxlen=maxlen)
#         self.hands_dist = deque(maxlen=maxlen)
#         self.times = deque(maxlen=maxlen)

#     def push(self, now, left_xy, right_xy):
#         """Push current time and (normalized) wrist coordinates or None"""
#         self.times.append(now)
#         # store x only if we have full xy available
#         self.left_x.append(left_xy[0] if left_xy is not None else None)
#         self.right_x.append(right_xy[0] if right_xy is not None else None)
#         self.left_xy.append(left_xy)
#         self.right_xy.append(right_xy)
#         if left_xy is not None and right_xy is not None:
#             dx = left_xy[0] - right_xy[0]
#             dy = left_xy[1] - right_xy[1]
#             self.hands_dist.append(math.hypot(dx, dy))
#         else:
#             self.hands_dist.append(None)

# def hand_is_pointing(hand_landmarks):
#     """Heuristic: index finger tip is noticeably away from its PIP joint"""
#     try:
#         tip = hand_landmarks.landmark[8]
#         pip = hand_landmarks.landmark[6]
#         seg = math.hypot(tip.x - pip.x, tip.y - pip.y)
#         return seg > POINT_INDEX_EXTENDED
#     except Exception:
#         return False

# def hand_is_fist(hand_landmarks):
#     """Heuristic: average distance from four finger tips to wrist is small"""
#     try:
#         wrist = hand_landmarks.landmark[0]
#         tips = [hand_landmarks.landmark[i] for i in (8, 12, 16, 20)]
#         dists = [math.hypot(tip.x - wrist.x, tip.y - wrist.y) for tip in tips]
#         avg = sum(dists)/len(dists)
#         return avg < FIST_FINGER_DISTANCE
#     except Exception:
#         return False

# def detect_clap(buff: TemporalBuffers):
#     """If any recent hand distance is below threshold -> clap"""
#     for d in reversed(buff.hands_dist):
#         if d is None:
#             continue
#         if d < CLAP_DIST_THRESHOLD:
#             return True
#     return False

# def detect_wave_from_series(x_series):
#     """Detect lateral oscillation in a series of x positions"""
#     vals = [x for x in x_series if x is not None]
#     if len(vals) < 5:
#         return False
#     # remove consecutive duplicates (flat sections)
#     cleaned = [vals[0]]
#     for v in vals[1:]:
#         if abs(v - cleaned[-1]) > 1e-6:
#             cleaned.append(v)
#     if len(cleaned) < 5:
#         return False
#     diffs = [cleaned[i+1] - cleaned[i] for i in range(len(cleaned)-1)]
#     signs = [1 if d>0 else (-1 if d<0 else 0) for d in diffs]
#     prev = None
#     changes = 0
#     for s in signs:
#         if s == 0:
#             continue
#         if prev is None:
#             prev = s
#         else:
#             if s != prev:
#                 changes += 1
#                 prev = s
#     amp = max(cleaned) - min(cleaned)
#     return (changes >= WAVE_MIN_PEAKS) and (amp >= WAVE_MIN_AMPLITUDE)

# def detect_action(results, buff: TemporalBuffers):
#     """Return string label of detected action based on pose + hand heuristics"""
#     hpose = results.pose_landmarks
#     lhand = results.left_hand_landmarks
#     rhand = results.right_hand_landmarks

#     if not hpose:
#         return "No person"

#     def p(i):
#         lm = hpose.landmark[i]
#         return (lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0))

#     L_SH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
#     R_SH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
#     L_WR = mp_pose.PoseLandmark.LEFT_WRIST.value
#     R_WR = mp_pose.PoseLandmark.RIGHT_WRIST.value

#     left_sh = p(L_SH)
#     right_sh = p(R_SH)
#     left_wr = p(L_WR)
#     right_wr = p(R_WR)

#     left_hand_up = (left_wr[1] < left_sh[1] - HANDS_UP_Y_DIFF) and (left_wr[3] >= VISIBILITY_MIN)
#     right_hand_up = (right_wr[1] < right_sh[1] - HANDS_UP_Y_DIFF) and (right_wr[3] >= VISIBILITY_MIN)

#     both_hands_up = left_hand_up and right_hand_up
#     any_hand_up = left_hand_up or right_hand_up

#     punch = False
#     if left_wr[2] < left_sh[2] - PUNCH_Z_DIFF and left_wr[3] >= VISIBILITY_MIN:
#         punch = True
#     if right_wr[2] < right_sh[2] - PUNCH_Z_DIFF and right_wr[3] >= VISIBILITY_MIN:
#         punch = True

#     point = False
#     if lhand and hand_is_pointing(lhand):
#         point = True
#     if rhand and hand_is_pointing(rhand):
#         point = True

#     fist = False
#     if lhand and hand_is_fist(lhand):
#         fist = True
#     if rhand and hand_is_fist(rhand):
#         fist = True

#     left_wave = detect_wave_from_series(buff.left_x)
#     right_wave = detect_wave_from_series(buff.right_x)
#     wave = left_wave or right_wave

#     clap = detect_clap(buff)

#     # Priority order (adjust if you prefer different priorities)
#     if clap:
#         return "Clap"
#     if punch:
#         return "Punch"
#     if point:
#         return "Point"
#     if wave:
#         return "Wave"
#     if both_hands_up:
#         return "Both Hands Up"
#     if any_hand_up:
#         return "Hands Up"
#     if fist:
#         return "Fist"
#     return "Unknown"

# def draw_pose_skeleton_with_labels(image, pose_landmarks):
#     """Draw 33 pose landmarks + optional index/name labels"""
#     h, w, _ = image.shape

#     # DrawingSpec using circle_radius for compatibility across versions
#     landmark_spec = mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=3)
#     connection_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)

#     mp_drawing.draw_landmarks(
#         image,
#         pose_landmarks,
#         mp_holistic.POSE_CONNECTIONS,
#         landmark_drawing_spec=landmark_spec,
#         connection_drawing_spec=connection_spec
#     )

#     if not (SHOW_LANDMARK_INDICES or SHOW_LANDMARK_NAMES):
#         return

#     for idx, lm in enumerate(pose_landmarks.landmark):
#         # convert normalized coords to pixel coords
#         px = int(lm.x * w)
#         py = int(lm.y * h)
#         # skip if landmark is out of frame or not visible
#         if px < 0 or px >= w or py < 0 or py >= h:
#             continue
#         if getattr(lm, "visibility", 1.0) < VISIBILITY_MIN:
#             continue

#         # small filled circle for better visibility
#         cv2.circle(image, (px, py), 4, (0, 200, 255), -1)

#         # prepare label
#         label = ""
#         if SHOW_LANDMARK_INDICES:
#             label = f"{idx}"
#         if SHOW_LANDMARK_NAMES:
#             try:
#                 lmname = mp_pose.PoseLandmark(idx).name
#             except Exception:
#                 lmname = ""
#             if label:
#                 label = f"{label}:{lmname}"
#             else:
#                 label = lmname

#         # draw shadowed text for readability
#         text_pos = (px + 6, py - 6)
#         cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
#                     LANDMARK_FONT_SCALE, (0,0,0), LANDMARK_FONT_THICKNESS+2, cv2.LINE_AA)
#         cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
#                     LANDMARK_FONT_SCALE, (220,220,220), LANDMARK_FONT_THICKNESS, cv2.LINE_AA)

# def main():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Cannot open webcam")
#         return

#     prev_time = time.time()
#     fps = 0.0
#     buffers = TemporalBuffers(maxlen=HISTORY_LEN)

#     paused = False
#     global SHOW_LANDMARK_INDICES, SHOW_LANDMARK_NAMES

#     with mp_holistic.Holistic(
#         static_image_mode=False,
#         model_complexity=1,
#         enable_segmentation=False,
#         refine_face_landmarks=False,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     ) as holistic:
#         while True:
#             if not paused:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 frame = cv2.flip(frame, 1)
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 h, w, _ = frame.shape

#                 # Process
#                 results = holistic.process(frame_rgb)

#                 # Draw pose + labels
#                 if results.pose_landmarks:
#                     draw_pose_skeleton_with_labels(frame, results.pose_landmarks)

#                 # Draw hands (use mp.solutions.hands connections constant)
#                 if results.left_hand_landmarks:
#                     mp_drawing.draw_landmarks(
#                         frame,
#                         results.left_hand_landmarks,
#                         mp_hands.HAND_CONNECTIONS,
#                         landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=2, circle_radius=3),
#                         connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,256,121), thickness=2)
#                     )
#                 if results.right_hand_landmarks:
#                     mp_drawing.draw_landmarks(
#                         frame,
#                         results.right_hand_landmarks,
#                         mp_hands.HAND_CONNECTIONS,
#                         landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=2, circle_radius=3),
#                         connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,256,121), thickness=2)
#                     )

#                 # bounding box from visible pose landmarks
#                 bbox = None
#                 if results.pose_landmarks:
#                     xs = [lm.x for lm in results.pose_landmarks.landmark if (getattr(lm, 'visibility', 1.0)) > VISIBILITY_MIN]
#                     ys = [lm.y for lm in results.pose_landmarks.landmark if (getattr(lm, 'visibility', 1.0)) > VISIBILITY_MIN]
#                     if xs and ys:
#                         x_min = int(max(0, min(xs) * w) - 10)
#                         x_max = int(min(w - 1, max(xs) * w) + 10)
#                         y_min = int(max(0, min(ys) * h) - 10)
#                         y_max = int(min(h - 1, max(ys) * h) + 10)
#                         bbox = (x_min, y_min, x_max, y_max)
#                         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

#                 # update temporal buffers with wrists (only if visible enough)
#                 now = time.time()
#                 left_xy = None
#                 right_xy = None
#                 try:
#                     if results.pose_landmarks:
#                         lw = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
#                         rw = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
#                         if getattr(lw, 'visibility', 0.0) >= VISIBILITY_MIN:
#                             left_xy = (lw.x, lw.y)
#                         if getattr(rw, 'visibility', 0.0) >= VISIBILITY_MIN:
#                             right_xy = (rw.x, rw.y)
#                 except Exception:
#                     pass
#                 buffers.push(now, left_xy, right_xy)

#                 # detect action
#                 label = detect_action(results, buffers)

#                 # compute fps (smoothed) with guard
#                 now2 = time.time()
#                 dt = now2 - prev_time if now2 - prev_time > 1e-6 else 1e-6
#                 instantaneous_fps = 1.0 / dt
#                 prev_time = now2
#                 fps = FPS_SMOOTHING * fps + (1 - FPS_SMOOTHING) * instantaneous_fps

#                 # Draw FPS top-left
#                 fps_text = f"FPS: {int(fps)}"
#                 cv2.rectangle(frame, (5,5), (150,35), (0,0,0), -1)
#                 cv2.putText(frame, fps_text, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

#                 # Draw Action label top-right
#                 label_text = f"Action: {label}"
#                 (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
#                 top_right_x = w - text_w - 20
#                 cv2.rectangle(frame, (top_right_x-10,5), (w-5,35), (0,0,0), -1)
#                 cv2.putText(frame, label_text, (top_right_x,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

#                 # show last hands distance for debug
#                 last_hand_dist = next((d for d in reversed(buffers.hands_dist) if d is not None), None)
#                 if last_hand_dist is not None:
#                     dist_text = f"HandsDist: {last_hand_dist:.2f}"
#                     cv2.putText(frame, dist_text, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

#                 if bbox:
#                     x_min, y_min, x_max, y_max = bbox
#                     cv2.putText(frame, "Person", (x_min, y_min-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

#                 cv2.imshow('Webcam Action + 33-point Skeleton', frame)

#             # key handling (works while paused as well)
#             key = cv2.waitKey(1) & 0xFF
#             if key == 27 or key == ord('q'):
#                 break
#             if key == ord('p'):
#                 paused = not paused
#             if key == ord('l'):
#                 # toggle labels
#                 SHOW_LANDMARK_INDICES = not SHOW_LANDMARK_INDICES
#                 SHOW_LANDMARK_NAMES = not SHOW_LANDMARK_NAMES

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


# """
# code_multi_person.py

# Multi-person webcam action + 33-point pose skeleton display (MediaPipe Holistic + OpenCV)
# - Person detection via MobileNet-SSD (OpenCV dnn).
# - Each detected person is cropped and processed by MediaPipe Holistic.
# - Landmarks are transformed back to full-frame coordinates and drawn.
# - Simple centroid tracker assigns persistent IDs and maintains per-person temporal buffers
#   so action heuristics (Wave, Clap, Punch, Point, Fist, Hands Up) run per person.

# Requires:
#     pip install opencv-python mediapipe

# On first run the script will attempt to download:
#     - MobileNetSSD_deploy.prototxt
#     - MobileNetSSD_deploy.caffemodel

# Run:
#     python code_multi_person.py

# Keys:
#     q / ESC -> quit
#     p       -> pause/resume
#     l       -> toggle landmark labels (indices/names)
# """
# import os
# import math
# import time
# import urllib.request
# from collections import deque
# import cv2
# import numpy as np
# import mediapipe as mp
# from mediapipe.framework.formats import landmark_pb2

# # ---------------- CONFIG ----------------
# HISTORY_LEN = 12
# WAVE_MIN_PEAKS = 2
# WAVE_MIN_AMPLITUDE = 0.04
# CLAP_DIST_THRESHOLD = 0.12
# FIST_FINGER_DISTANCE = 0.07
# PUNCH_Z_DIFF = 0.12
# HANDS_UP_Y_DIFF = 0.08
# POINT_INDEX_EXTENDED = 0.06
# VISIBILITY_MIN = 0.3
# FPS_SMOOTHING = 0.9

# PERSON_CONF_THRESHOLD = 0.5  # detector confidence threshold
# TRACK_MAX_DISTANCE = 120     # max pixel distance to match existing track
# DETECT_EVERY_N_FRAMES = 1    # set >1 to reduce detector frequency (speed optimization)

# # SHOW landmark indices & names (set False to reduce clutter)
# SHOW_LANDMARK_INDICES = True
# SHOW_LANDMARK_NAMES = True
# LANDMARK_FONT_SCALE = 0.45
# LANDMARK_FONT_THICKNESS = 1
# # ----------------------------------------

# # MobileNetSSD model files (will download if missing)
# PROTOTXT_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt"
# CAFFEMODEL_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel"
# PROTOTXT_FN = "MobileNetSSD_deploy.prototxt"
# CAFFEMODEL_FN = "MobileNetSSD_deploy.caffemodel"

# # class id for 'person' in MobileNetSSD
# MOBILENET_PERSON_CLASS_ID = 15

# mp_holistic = mp.solutions.holistic
# mp_pose = mp.solutions.pose
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# # -------------------------------------------------------
# # Utility: download model files if not present
# # -------------------------------------------------------
# def download_if_missing(fname, url):
#     if os.path.exists(fname):
#         return
#     print(f"[INFO] Downloading {fname} ... (this may take a while)")
#     urllib.request.urlretrieve(url, fname)
#     print(f"[INFO] Saved {fname}")

# # -------------------------------------------------------
# # Temporal buffers per person (for wave/clap detection)
# # -------------------------------------------------------
# class TemporalBuffers:
#     def __init__(self, maxlen=HISTORY_LEN):
#         self.left_x = deque(maxlen=maxlen)
#         self.right_x = deque(maxlen=maxlen)
#         self.left_xy = deque(maxlen=maxlen)
#         self.right_xy = deque(maxlen=maxlen)
#         self.hands_dist = deque(maxlen=maxlen)
#         self.times = deque(maxlen=maxlen)
#     def push(self, now, left_xy, right_xy):
#         self.times.append(now)
#         self.left_x.append(left_xy[0] if left_xy is not None else None)
#         self.right_x.append(right_xy[0] if right_xy is not None else None)
#         self.left_xy.append(left_xy)
#         self.right_xy.append(right_xy)
#         if left_xy is not None and right_xy is not None:
#             dx = left_xy[0] - right_xy[0]
#             dy = left_xy[1] - right_xy[1]
#             self.hands_dist.append(math.hypot(dx, dy))
#         else:
#             self.hands_dist.append(None)

# # -------------------------------------------------------
# # Heuristic gesture checks (same as original)
# # -------------------------------------------------------
# def hand_is_pointing(hand_landmarks):
#     try:
#         tip = hand_landmarks.landmark[8]
#         pip = hand_landmarks.landmark[6]
#         seg = math.hypot(tip.x - pip.x, tip.y - pip.y)
#         return seg > POINT_INDEX_EXTENDED
#     except Exception:
#         return False

# def hand_is_fist(hand_landmarks):
#     try:
#         wrist = hand_landmarks.landmark[0]
#         tips = [hand_landmarks.landmark[i] for i in (8, 12, 16, 20)]
#         dists = [math.hypot(tip.x - wrist.x, tip.y - wrist.y) for tip in tips]
#         avg = sum(dists)/len(dists)
#         return avg < FIST_FINGER_DISTANCE
#     except Exception:
#         return False

# def detect_clap(buff: TemporalBuffers):
#     for d in reversed(buff.hands_dist):
#         if d is None:
#             continue
#         if d < CLAP_DIST_THRESHOLD:
#             return True
#     return False

# def detect_wave_from_series(x_series):
#     vals = [x for x in x_series if x is not None]
#     if len(vals) < 5:
#         return False
#     cleaned = [vals[0]]
#     for v in vals[1:]:
#         if abs(v - cleaned[-1]) > 1e-6:
#             cleaned.append(v)
#     if len(cleaned) < 5:
#         return False
#     diffs = [cleaned[i+1] - cleaned[i] for i in range(len(cleaned)-1)]
#     signs = [1 if d>0 else (-1 if d<0 else 0) for d in diffs]
#     prev = None
#     changes = 0
#     for s in signs:
#         if s == 0:
#             continue
#         if prev is None:
#             prev = s
#         else:
#             if s != prev:
#                 changes += 1
#                 prev = s
#     amp = max(cleaned) - min(cleaned)
#     return (changes >= WAVE_MIN_PEAKS) and (amp >= WAVE_MIN_AMPLITUDE)

# def detect_action_for_person(results_for_person, buff: TemporalBuffers):
#     # results_for_person: object with pose_landmarks, left_hand_landmarks, right_hand_landmarks
#     hpose = results_for_person.pose_landmarks
#     lhand = results_for_person.left_hand_landmarks
#     rhand = results_for_person.right_hand_landmarks

#     if not hpose:
#         return "No person"

#     def p(i):
#         lm = hpose.landmark[i]
#         return (lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0))

#     L_SH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
#     R_SH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
#     L_WR = mp_pose.PoseLandmark.LEFT_WRIST.value
#     R_WR = mp_pose.PoseLandmark.RIGHT_WRIST.value

#     left_sh = p(L_SH)
#     right_sh = p(R_SH)
#     left_wr = p(L_WR)
#     right_wr = p(R_WR)

#     left_hand_up = (left_wr[1] < left_sh[1] - HANDS_UP_Y_DIFF) and (left_wr[3] >= VISIBILITY_MIN)
#     right_hand_up = (right_wr[1] < right_sh[1] - HANDS_UP_Y_DIFF) and (right_wr[3] >= VISIBILITY_MIN)

#     both_hands_up = left_hand_up and right_hand_up
#     any_hand_up = left_hand_up or right_hand_up

#     punch = False
#     if left_wr[2] < left_sh[2] - PUNCH_Z_DIFF and left_wr[3] >= VISIBILITY_MIN:
#         punch = True
#     if right_wr[2] < right_sh[2] - PUNCH_Z_DIFF and right_wr[3] >= VISIBILITY_MIN:
#         punch = True

#     point = False
#     if lhand and hand_is_pointing(lhand):
#         point = True
#     if rhand and hand_is_pointing(rhand):
#         point = True

#     fist = False
#     if lhand and hand_is_fist(lhand):
#         fist = True
#     if rhand and hand_is_fist(rhand):
#         fist = True

#     left_wave = detect_wave_from_series(buff.left_x)
#     right_wave = detect_wave_from_series(buff.right_x)
#     wave = left_wave or right_wave

#     clap = detect_clap(buff)

#     if clap:
#         return "Clap"
#     if punch:
#         return "Punch"
#     if point:
#         return "Point"
#     if wave:
#         return "Wave"
#     if both_hands_up:
#         return "Both Hands Up"
#     if any_hand_up:
#         return "Hands Up"
#     if fist:
#         return "Fist"
#     return "Unknown"

# # -------------------------------------------------------
# # Helpers to translate landmarks from crop-space -> full-frame normalized coords
# # We'll create new NormalizedLandmarkList objects with adjusted coords
# # -------------------------------------------------------
# def transform_landmarks_from_roi(landmark_list, roi_x, roi_y, roi_w, roi_h, full_w, full_h):
#     """Given a mediapipe NormalizedLandmarkList (landmark_list) where coords are normalized
#        w.r.t the ROI (0..1), shift & scale them into full image normalized coords and
#        return a new NormalizedLandmarkList."""
#     if landmark_list is None:
#         return None
#     out = landmark_pb2.NormalizedLandmarkList()
#     for lm in landmark_list.landmark:
#         # lm.x/lm.y are normalized wrt ROI; convert to pixels then to full normalized coords
#         abs_x = roi_x + lm.x * roi_w
#         abs_y = roi_y + lm.y * roi_h
#         new = out.landmark.add()
#         new.x = abs_x / float(full_w)
#         new.y = abs_y / float(full_h)
#         # z is relative to ROI — approximate by scaling with roi size ratio
#         # maintain the sign and magnitude but scale by roi diagonal ratio -> approximate
#         roi_diag = math.hypot(roi_w, roi_h)
#         full_diag = math.hypot(full_w, full_h)
#         if roi_diag > 0:
#             new.z = (lm.z * (roi_diag / full_diag))
#         else:
#             new.z = lm.z
#         # transfer visibility if present
#         if hasattr(lm, 'visibility'):
#             new.visibility = lm.visibility
#     return out

# # -------------------------------------------------------
# # Simple centroid-based tracker
# # Keeps a dictionary track_id -> center (x,y), last_seen_time
# # -------------------------------------------------------
# class SimpleTracker:
#     def __init__(self):
#         self.next_id = 1
#         self.tracks = {}  # id -> dict(center=(x,y), last_seen=ts)
#     def update(self, detections_centers):
#         """
#         detections_centers: list of (cx, cy)
#         returns list of track_ids aligned with detections_centers
#         """
#         assigned_ids = []
#         used_prev = set()
#         # naive greedy nearest neighbor match
#         for (cx, cy) in detections_centers:
#             best_id = None
#             best_dist = None
#             for tid, info in self.tracks.items():
#                 if tid in used_prev:
#                     continue
#                 px, py = info['center']
#                 d = math.hypot(px - cx, py - cy)
#                 if best_dist is None or d < best_dist:
#                     best_dist = d
#                     best_id = tid
#             if best_id is not None and best_dist is not None and best_dist <= TRACK_MAX_DISTANCE:
#                 # assign
#                 assigned_ids.append(best_id)
#                 used_prev.add(best_id)
#                 self.tracks[best_id]['center'] = (cx, cy)
#                 self.tracks[best_id]['last_seen'] = time.time()
#             else:
#                 # new track
#                 tid = self.next_id
#                 self.next_id += 1
#                 self.tracks[tid] = {'center': (cx, cy), 'last_seen': time.time()}
#                 assigned_ids.append(tid)
#         # cleanup stale tracks (not seen for a while)
#         now = time.time()
#         stale_ids = [tid for tid,info in self.tracks.items() if now - info['last_seen'] > 2.0]
#         for tid in stale_ids:
#             del self.tracks[tid]
#         return assigned_ids

# # -------------------------------------------------------
# # Draw pose skeleton + labels (works with NormalizedLandmarkList of full-image normalized coords)
# # -------------------------------------------------------
# def draw_pose_skeleton_with_labels(image, pose_landmarks):
#     h, w, _ = image.shape
#     landmark_spec = mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=3)
#     connection_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)

#     # mp_drawing.draw_landmarks expects a LandmarkList object with normalized coords
#     try:
#         mp_drawing.draw_landmarks(
#             image,
#             pose_landmarks,
#             mp_holistic.POSE_CONNECTIONS,
#             landmark_drawing_spec=landmark_spec,
#             connection_drawing_spec=connection_spec
#         )
#     except Exception:
#         # defensive: if drawing fails, just skip
#         return

#     if not (SHOW_LANDMARK_INDICES or SHOW_LANDMARK_NAMES):
#         return

#     for idx, lm in enumerate(pose_landmarks.landmark):
#         px = int(lm.x * w)
#         py = int(lm.y * h)
#         if px < 0 or px >= w or py < 0 or py >= h:
#             continue
#         if getattr(lm, 'visibility', 1.0) < VISIBILITY_MIN:
#             continue
#         cv2.circle(image, (px, py), 4, (0, 200, 255), -1)
#         label = ""
#         if SHOW_LANDMARK_INDICES:
#             label = f"{idx}"
#         if SHOW_LANDMARK_NAMES:
#             try:
#                 lmname = mp_pose.PoseLandmark(idx).name
#             except Exception:
#                 lmname = ""
#             if label:
#                 label = f"{label}:{lmname}"
#             else:
#                 label = lmname
#         text_pos = (px + 6, py - 6)
#         cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
#                     LANDMARK_FONT_SCALE, (0,0,0), LANDMARK_FONT_THICKNESS+2, cv2.LINE_AA)
#         cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
#                     LANDMARK_FONT_SCALE, (220,220,220), LANDMARK_FONT_THICKNESS, cv2.LINE_AA)

# # -------------------------------------------------------
# # Main
# # -------------------------------------------------------
# def main():
#     # ensure detector model files exist (download if missing)
#     try:
#         download_if_missing(PROTOTXT_FN, PROTOTXT_URL)
#         download_if_missing(CAFFEMODEL_FN, CAFFEMODEL_URL)
#     except Exception as e:
#         print("[WARN] could not auto-download detector files:", e)
#         print("If you already have MobileNetSSD_deploy.prototxt and MobileNetSSD_deploy.caffemodel in this folder, ok.")
#         print("Otherwise place them in this script's directory or ensure internet connectivity.")

#     # load person detector (MobileNet-SSD)
#     detector_net = None
#     if os.path.exists(PROTOTXT_FN) and os.path.exists(CAFFEMODEL_FN):
#         detector_net = cv2.dnn.readNetFromCaffe(PROTOTXT_FN, CAFFEMODEL_FN)
#         # try to use OpenCV's CUDA backend if available? (optional)
#         # detector_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
#     else:
#         print("[ERROR] Detector model files missing. Multi-person detection won't work.")
#         print("You can still run single-person Holistic by removing the detector code.")
#         return

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Cannot open webcam")
#         return

#     prev_time = time.time()
#     fps = 0.0

#     buffers_by_id = {}  # track_id -> TemporalBuffers
#     tracker = SimpleTracker()
#     paused = False
#     global SHOW_LANDMARK_INDICES, SHOW_LANDMARK_NAMES

#     frame_idx = 0

#     # We'll use a single Holistic instance (re-used) for speed.
#     with mp_holistic.Holistic(
#         static_image_mode=False,
#         model_complexity=1,
#         enable_segmentation=False,
#         refine_face_landmarks=False,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     ) as holistic:

#         while True:
#             if not paused:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 frame = cv2.flip(frame, 1)
#                 full_h, full_w, _ = frame.shape
#                 frame_disp = frame.copy()

#                 detections = []
#                 # optionally run detector every N frames
#                 if (frame_idx % DETECT_EVERY_N_FRAMES) == 0:
#                     blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
#                     detector_net.setInput(blob)
#                     dets = detector_net.forward()
#                     h = full_h; w = full_w
#                     for i in range(dets.shape[2]):
#                         conf = float(dets[0, 0, i, 2])
#                         cls = int(dets[0, 0, i, 1])
#                         if conf > PERSON_CONF_THRESHOLD and cls == MOBILENET_PERSON_CLASS_ID:
#                             x1 = int(dets[0, 0, i, 3] * w)
#                             y1 = int(dets[0, 0, i, 4] * h)
#                             x2 = int(dets[0, 0, i, 5] * w)
#                             y2 = int(dets[0, 0, i, 6] * h)
#                             # clip
#                             x1 = max(0, min(w-1, x1))
#                             x2 = max(0, min(w-1, x2))
#                             y1 = max(0, min(h-1, y1))
#                             y2 = max(0, min(h-1, y2))
#                             if x2 - x1 < 20 or y2 - y1 < 20:
#                                 continue
#                             detections.append((x1, y1, x2, y2, conf))
#                 else:
#                     # reuse previous detections from tracker keys (naive fallback)
#                     detections = []

#                 # Compute detection centers and update tracker
#                 centers = []
#                 padded_rois = []
#                 for (x1,y1,x2,y2,conf) in detections:
#                     cx = int((x1+x2)/2)
#                     cy = int((y1+y2)/2)
#                     centers.append((cx, cy))
#                     # pad bounding box a little for context
#                     pad_x = int(0.05 * (x2-x1))
#                     pad_y = int(0.10 * (y2-y1))
#                     rx1 = max(0, x1 - pad_x)
#                     ry1 = max(0, y1 - pad_y)
#                     rx2 = min(full_w-1, x2 + pad_x)
#                     ry2 = min(full_h-1, y2 + pad_y)
#                     padded_rois.append((rx1, ry1, rx2, ry2))

#                 track_ids = tracker.update(centers) if centers else []

#                 # For each detection/track, run Holistic on the crop and map back
#                 per_person_results = {}  # tid -> results object (with pose, left_hand, right_hand landmarks)
#                 for idx, tid in enumerate(track_ids):
#                     rx1, ry1, rx2, ry2 = padded_rois[idx]
#                     roi = frame[ry1:ry2, rx1:rx2]
#                     if roi.size == 0:
#                         continue
#                     # MediaPipe expects RGB
#                     roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
#                     # process ROI
#                     res = holistic.process(roi_rgb)
#                     # transform pose + hands to full-frame normalized coords
#                     roi_w = (rx2 - rx1)
#                     roi_h = (ry2 - ry1)
#                     transformed_pose = transform_landmarks_from_roi(res.pose_landmarks, rx1, ry1, roi_w, roi_h, full_w, full_h)
#                     transformed_left = transform_landmarks_from_roi(res.left_hand_landmarks, rx1, ry1, roi_w, roi_h, full_w, full_h)
#                     transformed_right = transform_landmarks_from_roi(res.right_hand_landmarks, rx1, ry1, roi_w, roi_h, full_w, full_h)
#                     # we will store a simple object with attributes used later
#                     person_obj = type("PersonRes", (), {})()
#                     person_obj.pose_landmarks = transformed_pose
#                     person_obj.left_hand_landmarks = transformed_left
#                     person_obj.right_hand_landmarks = transformed_right
#                     per_person_results[tid] = (person_obj, (rx1, ry1, rx2, ry2))
#                     # draw bbox and person's temporary id
#                     cv2.rectangle(frame_disp, (rx1, ry1), (rx2, ry2), (255, 160, 0), 2)
#                     cv2.putText(frame_disp, f"ID:{tid}", (rx1+4, ry1+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,160,0), 2)

#                     # ensure TemporalBuffers exists for this track
#                     if tid not in buffers_by_id:
#                         buffers_by_id[tid] = TemporalBuffers(maxlen=HISTORY_LEN)

#                     # extract wrist positions from transformed_pose (these are normalized full-frame)
#                     left_xy = None
#                     right_xy = None
#                     try:
#                         pl = transformed_pose
#                         if pl:
#                             lw = pl.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
#                             rw = pl.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
#                             if getattr(lw, 'visibility', 1.0) >= VISIBILITY_MIN:
#                                 left_xy = (lw.x, lw.y)
#                             if getattr(rw, 'visibility', 1.0) >= VISIBILITY_MIN:
#                                 right_xy = (rw.x, rw.y)
#                     except Exception:
#                         pass
#                     buffers_by_id[tid].push(time.time(), left_xy, right_xy)

#                 # For each tracked person, compute action and draw skeleton/labels
#                 for tid, (person_obj, bbox) in per_person_results.items():
#                     # detect action using per-person results and buffers
#                     action_label = detect_action_for_person(person_obj, buffers_by_id.get(tid, TemporalBuffers()))
#                     # draw pose skeleton (transformed_pose) onto frame_disp
#                     if person_obj.pose_landmarks:
#                         draw_pose_skeleton_with_labels(frame_disp, person_obj.pose_landmarks)
#                     # draw hands if present
#                     # drawing of hands expects NormalizedLandmarkList in full-frame coords
#                     if person_obj.left_hand_landmarks:
#                         try:
#                             mp_drawing.draw_landmarks(
#                                 frame_disp,
#                                 person_obj.left_hand_landmarks,
#                                 mp_hands.HAND_CONNECTIONS,
#                                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=2, circle_radius=3),
#                                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,256,121), thickness=2)
#                             )
#                         except Exception:
#                             pass
#                     if person_obj.right_hand_landmarks:
#                         try:
#                             mp_drawing.draw_landmarks(
#                                 frame_disp,
#                                 person_obj.right_hand_landmarks,
#                                 mp_hands.HAND_CONNECTIONS,
#                                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=2, circle_radius=3),
#                                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,256,121), thickness=2)
#                             )
#                         except Exception:
#                             pass

#                     # annotate action label near bbox
#                     rx1, ry1, rx2, ry2 = bbox
#                     label_text = f"ID:{tid} {action_label}"
#                     cv2.rectangle(frame_disp, (rx1, ry2+2), (rx1 + 180, ry2+26), (0,0,0), -1)
#                     cv2.putText(frame_disp, label_text, (rx1+6, ry2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 2)

#                 # FPS
#                 now2 = time.time()
#                 dt = now2 - prev_time if now2 - prev_time > 1e-6 else 1e-6
#                 prev_time = now2
#                 fps = FPS_SMOOTHING * fps + (1 - FPS_SMOOTHING) * (1.0/dt)
#                 cv2.putText(frame_disp, f"FPS: {int(fps)}", (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

#                 cv2.imshow('Multi-person Action + 33-point Skeleton', frame_disp)
#                 frame_idx += 1

#             # key handling
#             key = cv2.waitKey(1) & 0xFF
#             if key == 27 or key == ord('q'):
#                 break
#             if key == ord('p'):
#                 paused = not paused
#             if key == ord('l'):
#                 SHOW_LANDMARK_INDICES = not SHOW_LANDMARK_INDICES
#                 SHOW_LANDMARK_NAMES = not SHOW_LANDMARK_NAMES

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()



"""
code_multi_person.py

Multi-person webcam action + 33-point pose skeleton display (MediaPipe Holistic + OpenCV)
- Person detection via MobileNet-SSD (OpenCV dnn).
- Each detected person is cropped and processed by MediaPipe Holistic.
- Landmarks are transformed back to full-frame coordinates and drawn.
- Simple centroid tracker assigns persistent IDs and maintains per-person temporal buffers
  so action heuristics (Wave, Clap, Punch, Point, Fist, Hands Up) run per person.

Requires:
    pip install opencv-python mediapipe

On first run the script will attempt to download:
    - MobileNetSSD_deploy.prototxt
    - MobileNetSSD_deploy.caffemodel

Run:
    python code_multi_person.py

Keys:
    q / ESC -> quit
    p       -> pause/resume
    l       -> toggle landmark labels (indices/names)
"""
import os
import math
import time
import urllib.request
from collections import deque, defaultdict
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# ---------------- CONFIG ----------------
HISTORY_LEN = 12
WAVE_MIN_PEAKS = 2
WAVE_MIN_AMPLITUDE = 0.04
CLAP_DIST_THRESHOLD = 0.12
FIST_FINGER_DISTANCE = 0.07
PUNCH_Z_DIFF = 0.12
HANDS_UP_Y_DIFF = 0.08
POINT_INDEX_EXTENDED = 0.06
VISIBILITY_MIN = 0.3
FPS_SMOOTHING = 0.9

PERSON_CONF_THRESHOLD = 0.5  # detector confidence threshold
TRACK_MAX_DISTANCE = 120     # max pixel distance to match existing track
DETECT_EVERY_N_FRAMES = 1    # set >1 to reduce detector frequency (speed optimization)

# SHOW landmark indices & names (set False to reduce clutter)
SHOW_LANDMARK_INDICES = True
SHOW_LANDMARK_NAMES = True
LANDMARK_FONT_SCALE = 0.45
LANDMARK_FONT_THICKNESS = 1
# ----------------------------------------

# MobileNetSSD model files (will download if missing)
PROTOTXT_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt"
CAFFEMODEL_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel"
PROTOTXT_FN = "MobileNetSSD_deploy.prototxt"
CAFFEMODEL_FN = "MobileNetSSD_deploy.caffemodel"

# class id for 'person' in MobileNetSSD
MOBILENET_PERSON_CLASS_ID = 15

mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# -------------------------------------------------------
# Utility: download model files if not present
# -------------------------------------------------------
def download_if_missing(fname, url):
    if os.path.exists(fname):
        return
    print(f"[INFO] Downloading {fname} ... (this may take a while)")
    urllib.request.urlretrieve(url, fname)
    print(f"[INFO] Saved {fname}")

# -------------------------------------------------------
# Temporal buffers per person (for wave/clap detection)
# -------------------------------------------------------
class TemporalBuffers:
    def __init__(self, maxlen=HISTORY_LEN):
        self.left_x = deque(maxlen=maxlen)
        self.right_x = deque(maxlen=maxlen)
        self.left_xy = deque(maxlen=maxlen)
        self.right_xy = deque(maxlen=maxlen)
        self.hands_dist = deque(maxlen=maxlen)
        self.times = deque(maxlen=maxlen)
    def push(self, now, left_xy, right_xy):
        self.times.append(now)
        self.left_x.append(left_xy[0] if left_xy is not None else None)
        self.right_x.append(right_xy[0] if right_xy is not None else None)
        self.left_xy.append(left_xy)
        self.right_xy.append(right_xy)
        if left_xy is not None and right_xy is not None:
            dx = left_xy[0] - right_xy[0]
            dy = left_xy[1] - right_xy[1]
            self.hands_dist.append(math.hypot(dx, dy))
        else:
            self.hands_dist.append(None)

# -------------------------------------------------------
# Heuristic gesture checks (same as original)
# -------------------------------------------------------
def hand_is_pointing(hand_landmarks):
    try:
        tip = hand_landmarks.landmark[8]
        pip = hand_landmarks.landmark[6]
        seg = math.hypot(tip.x - pip.x, tip.y - pip.y)
        return seg > POINT_INDEX_EXTENDED
    except Exception:
        return False

def hand_is_fist(hand_landmarks):
    try:
        wrist = hand_landmarks.landmark[0]
        tips = [hand_landmarks.landmark[i] for i in (8, 12, 16, 20)]
        dists = [math.hypot(tip.x - wrist.x, tip.y - wrist.y) for tip in tips]
        avg = sum(dists)/len(dists)
        return avg < FIST_FINGER_DISTANCE
    except Exception:
        return False

def detect_clap(buff: TemporalBuffers):
    for d in reversed(buff.hands_dist):
        if d is None:
            continue
        if d < CLAP_DIST_THRESHOLD:
            return True
    return False

def detect_wave_from_series(x_series):
    vals = [x for x in x_series if x is not None]
    if len(vals) < 5:
        return False
    cleaned = [vals[0]]
    for v in vals[1:]:
        if abs(v - cleaned[-1]) > 1e-6:
            cleaned.append(v)
    if len(cleaned) < 5:
        return False
    diffs = [cleaned[i+1] - cleaned[i] for i in range(len(cleaned)-1)]
    signs = [1 if d>0 else (-1 if d<0 else 0) for d in diffs]
    prev = None
    changes = 0
    for s in signs:
        if s == 0:
            continue
        if prev is None:
            prev = s
        else:
            if s != prev:
                changes += 1
                prev = s
    amp = max(cleaned) - min(cleaned)
    return (changes >= WAVE_MIN_PEAKS) and (amp >= WAVE_MIN_AMPLITUDE)

def detect_action_for_person(results_for_person, buff: TemporalBuffers):
    # results_for_person: object with pose_landmarks, left_hand_landmarks, right_hand_landmarks
    hpose = results_for_person.pose_landmarks
    lhand = results_for_person.left_hand_landmarks
    rhand = results_for_person.right_hand_landmarks

    if not hpose:
        return "No person"

    def p(i):
        lm = hpose.landmark[i]
        return (lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0))

    L_SH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    R_SH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    L_WR = mp_pose.PoseLandmark.LEFT_WRIST.value
    R_WR = mp_pose.PoseLandmark.RIGHT_WRIST.value

    left_sh = p(L_SH)
    right_sh = p(R_SH)
    left_wr = p(L_WR)
    right_wr = p(R_WR)

    left_hand_up = (left_wr[1] < left_sh[1] - HANDS_UP_Y_DIFF) and (left_wr[3] >= VISIBILITY_MIN)
    right_hand_up = (right_wr[1] < right_sh[1] - HANDS_UP_Y_DIFF) and (right_wr[3] >= VISIBILITY_MIN)

    both_hands_up = left_hand_up and right_hand_up
    any_hand_up = left_hand_up or right_hand_up

    punch = False
    if left_wr[2] < left_sh[2] - PUNCH_Z_DIFF and left_wr[3] >= VISIBILITY_MIN:
        punch = True
    if right_wr[2] < right_sh[2] - PUNCH_Z_DIFF and right_wr[3] >= VISIBILITY_MIN:
        punch = True

    point = False
    if lhand and hand_is_pointing(lhand):
        point = True
    if rhand and hand_is_pointing(rhand):
        point = True

    fist = False
    if lhand and hand_is_fist(lhand):
        fist = True
    if rhand and hand_is_fist(rhand):
        fist = True

    left_wave = detect_wave_from_series(buff.left_x)
    right_wave = detect_wave_from_series(buff.right_x)
    wave = left_wave or right_wave

    clap = detect_clap(buff)

    if clap:
        return "Clap"
    if punch:
        return "Punch"
    if point:
        return "Point"
    if wave:
        return "Wave"
    if both_hands_up:
        return "Both Hands Up"
    if any_hand_up:
        return "Hands Up"
    if fist:
        return "Fist"
    return "Unknown"

# -------------------------------------------------------
# Helpers to translate landmarks from crop-space -> full-frame normalized coords
# We'll create new NormalizedLandmarkList objects with adjusted coords
# -------------------------------------------------------
def transform_landmarks_from_roi(landmark_list, roi_x, roi_y, roi_w, roi_h, full_w, full_h):
    """Given a mediapipe NormalizedLandmarkList (landmark_list) where coords are normalized
       w.r.t the ROI (0..1), shift & scale them into full image normalized coords and
       return a new NormalizedLandmarkList."""
    if landmark_list is None:
        return None
    out = landmark_pb2.NormalizedLandmarkList()
    for lm in landmark_list.landmark:
        # lm.x/lm.y are normalized wrt ROI; convert to pixels then to full normalized coords
        abs_x = roi_x + lm.x * roi_w
        abs_y = roi_y + lm.y * roi_h
        new = out.landmark.add()
        new.x = abs_x / float(full_w)
        new.y = abs_y / float(full_h)
        # z is relative to ROI — approximate by scaling with roi size ratio
        # maintain the sign and magnitude but scale by roi diagonal ratio -> approximate
        roi_diag = math.hypot(roi_w, roi_h)
        full_diag = math.hypot(full_w, full_h)
        if roi_diag > 0:
            new.z = (lm.z * (roi_diag / full_diag))
        else:
            new.z = lm.z
        # transfer visibility if present
        if hasattr(lm, 'visibility'):
            new.visibility = lm.visibility
    return out

# -------------------------------------------------------
# Simple centroid-based tracker
# Keeps a dictionary track_id -> center (x,y), last_seen_time
# -------------------------------------------------------
class SimpleTracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = {}  # id -> dict(center=(x,y), last_seen=ts)
    def update(self, detections_centers):
        """
        detections_centers: list of (cx, cy)
        returns list of track_ids aligned with detections_centers
        """
        assigned_ids = []
        used_prev = set()
        # naive greedy nearest neighbor match
        for (cx, cy) in detections_centers:
            best_id = None
            best_dist = None
            for tid, info in self.tracks.items():
                if tid in used_prev:
                    continue
                px, py = info['center']
                d = math.hypot(px - cx, py - cy)
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_id = tid
            if best_id is not None and best_dist is not None and best_dist <= TRACK_MAX_DISTANCE:
                # assign
                assigned_ids.append(best_id)
                used_prev.add(best_id)
                self.tracks[best_id]['center'] = (cx, cy)
                self.tracks[best_id]['last_seen'] = time.time()
            else:
                # new track
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {'center': (cx, cy), 'last_seen': time.time()}
                assigned_ids.append(tid)
        # cleanup stale tracks (not seen for a while)
        now = time.time()
        stale_ids = [tid for tid,info in self.tracks.items() if now - info['last_seen'] > 2.0]
        for tid in stale_ids:
            del self.tracks[tid]
        return assigned_ids

# -------------------------------------------------------
# Draw pose skeleton + labels (works with NormalizedLandmarkList of full-image normalized coords)
# -------------------------------------------------------
def draw_pose_skeleton_with_labels(image, pose_landmarks):
    h, w, _ = image.shape
    landmark_spec = mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=3)
    connection_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)

    # mp_drawing.draw_landmarks expects a LandmarkList object with normalized coords
    try:
        mp_drawing.draw_landmarks(
            image,
            pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_spec,
            connection_drawing_spec=connection_spec
        )
    except Exception:
        # defensive: if drawing fails, just skip
        return

    if not (SHOW_LANDMARK_INDICES or SHOW_LANDMARK_NAMES):
        return

    for idx, lm in enumerate(pose_landmarks.landmark):
        px = int(lm.x * w)
        py = int(lm.y * h)
        if px < 0 or px >= w or py < 0 or py >= h:
            continue
        if getattr(lm, 'visibility', 1.0) < VISIBILITY_MIN:
            continue
        cv2.circle(image, (px, py), 4, (0, 200, 255), -1)
        label = ""
        if SHOW_LANDMARK_INDICES:
            label = f"{idx}"
        if SHOW_LANDMARK_NAMES:
            try:
                lmname = mp_pose.PoseLandmark(idx).name
            except Exception:
                lmname = ""
            if label:
                label = f"{label}:{lmname}"
            else:
                label = lmname
        text_pos = (px + 6, py - 6)
        cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    LANDMARK_FONT_SCALE, (0,0,0), LANDMARK_FONT_THICKNESS+2, cv2.LINE_AA)
        cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    LANDMARK_FONT_SCALE, (220,220,220), LANDMARK_FONT_THICKNESS, cv2.LINE_AA)

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    # ensure detector model files exist (download if missing)
    try:
        download_if_missing(PROTOTXT_FN, PROTOTXT_URL)
        download_if_missing(CAFFEMODEL_FN, CAFFEMODEL_URL)
    except Exception as e:
        print("[WARN] could not auto-download detector files:", e)
        print("If you already have MobileNetSSD_deploy.prototxt and MobileNetSSD_deploy.caffemodel in this folder, ok.")
        print("Otherwise place them in this script's directory or ensure internet connectivity.")

    # load person detector (MobileNet-SSD)
    detector_net = None
    if os.path.exists(PROTOTXT_FN) and os.path.exists(CAFFEMODEL_FN):
        detector_net = cv2.dnn.readNetFromCaffe(PROTOTXT_FN, CAFFEMODEL_FN)
        # try to use OpenCV's CUDA backend if available? (optional)
        # detector_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    else:
        print("[ERROR] Detector model files missing. Multi-person detection won't work.")
        print("You can still run single-person Holistic by removing the detector code.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    prev_time = time.time()
    fps = 0.0

    # <-- Important: initialize buffers_by_id here (per-track temporal buffers)
    buffers_by_id = {}  # track_id -> TemporalBuffers

    tracker = SimpleTracker()
    paused = False
    global SHOW_LANDMARK_INDICES, SHOW_LANDMARK_NAMES

    frame_idx = 0

    # We'll use a single Holistic instance (re-used) for speed.
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                full_h, full_w, _ = frame.shape
                frame_disp = frame.copy()

                detections = []
                # optionally run detector every N frames
                if (frame_idx % DETECT_EVERY_N_FRAMES) == 0:
                    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
                    detector_net.setInput(blob)
                    dets = detector_net.forward()
                    h = full_h; w = full_w
                    for i in range(dets.shape[2]):
                        conf = float(dets[0, 0, i, 2])
                        cls = int(dets[0, 0, i, 1])
                        if conf > PERSON_CONF_THRESHOLD and cls == MOBILENET_PERSON_CLASS_ID:
                            x1 = int(dets[0, 0, i, 3] * w)
                            y1 = int(dets[0, 0, i, 4] * h)
                            x2 = int(dets[0, 0, i, 5] * w)
                            y2 = int(dets[0, 0, i, 6] * h)
                            # clip
                            x1 = max(0, min(w-1, x1))
                            x2 = max(0, min(w-1, x2))
                            y1 = max(0, min(h-1, y1))
                            y2 = max(0, min(h-1, y2))
                            if x2 - x1 < 20 or y2 - y1 < 20:
                                continue
                            detections.append((x1, y1, x2, y2, conf))
                else:
                    # reuse previous detections from tracker keys (naive fallback)
                    detections = []

                # Compute detection centers and update tracker
                centers = []
                padded_rois = []
                for (x1,y1,x2,y2,conf) in detections:
                    cx = int((x1+x2)/2)
                    cy = int((y1+y2)/2)
                    centers.append((cx, cy))
                    # pad bounding box a little for context
                    pad_x = int(0.05 * (x2-x1))
                    pad_y = int(0.10 * (y2-y1))
                    rx1 = max(0, x1 - pad_x)
                    ry1 = max(0, y1 - pad_y)
                    rx2 = min(full_w-1, x2 + pad_x)
                    ry2 = min(full_h-1, y2 + pad_y)
                    padded_rois.append((rx1, ry1, rx2, ry2))

                track_ids = tracker.update(centers) if centers else []

                # For each detection/track, run Holistic on the crop and map back
                per_person_results = {}  # tid -> results object (with pose, left_hand, right_hand landmarks)
                for idx, tid in enumerate(track_ids):
                    rx1, ry1, rx2, ry2 = padded_rois[idx]
                    roi = frame[ry1:ry2, rx1:rx2]
                    if roi.size == 0:
                        continue
                    # MediaPipe expects RGB
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    # process ROI
                    res = holistic.process(roi_rgb)
                    # transform pose + hands to full-frame normalized coords
                    roi_w = (rx2 - rx1)
                    roi_h = (ry2 - ry1)
                    transformed_pose = transform_landmarks_from_roi(res.pose_landmarks, rx1, ry1, roi_w, roi_h, full_w, full_h)
                    transformed_left = transform_landmarks_from_roi(res.left_hand_landmarks, rx1, ry1, roi_w, roi_h, full_w, full_h)
                    transformed_right = transform_landmarks_from_roi(res.right_hand_landmarks, rx1, ry1, roi_w, roi_h, full_w, full_h)
                    # we will store a simple object with attributes used later
                    person_obj = type("PersonRes", (), {})()
                    person_obj.pose_landmarks = transformed_pose
                    person_obj.left_hand_landmarks = transformed_left
                    person_obj.right_hand_landmarks = transformed_right
                    per_person_results[tid] = (person_obj, (rx1, ry1, rx2, ry2))
                    # draw bbox and person's temporary id
                    cv2.rectangle(frame_disp, (rx1, ry1), (rx2, ry2), (255, 160, 0), 2)
                    cv2.putText(frame_disp, f"ID:{tid}", (rx1+4, ry1+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,160,0), 2)

                    # ensure TemporalBuffers exists for this track
                    if tid not in buffers_by_id:
                        buffers_by_id[tid] = TemporalBuffers(maxlen=HISTORY_LEN)

                    # extract wrist positions from transformed_pose (these are normalized full-frame)
                    left_xy = None
                    right_xy = None
                    try:
                        pl = transformed_pose
                        if pl:
                            lw = pl.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
                            rw = pl.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                            if getattr(lw, 'visibility', 1.0) >= VISIBILITY_MIN:
                                left_xy = (lw.x, lw.y)
                            if getattr(rw, 'visibility', 1.0) >= VISIBILITY_MIN:
                                right_xy = (rw.x, rw.y)
                    except Exception:
                        pass
                    buffers_by_id[tid].push(time.time(), left_xy, right_xy)

                # For each tracked person, compute action and draw skeleton/labels
                # We'll also collect action tallies to show in top-right
                action_tally = defaultdict(int)
                for tid, (person_obj, bbox) in per_person_results.items():
                    # detect action using per-person results and buffers
                    action_label = detect_action_for_person(person_obj, buffers_by_id.get(tid, TemporalBuffers()))
                    action_tally[action_label] += 1

                    # draw pose skeleton (transformed_pose) onto frame_disp
                    if person_obj.pose_landmarks:
                        draw_pose_skeleton_with_labels(frame_disp, person_obj.pose_landmarks)

                    # draw hands landmarks
                    if person_obj.left_hand_landmarks:
                        try:
                            mp_drawing.draw_landmarks(
                                frame_disp,
                                person_obj.left_hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=2, circle_radius=3),
                                connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,256,121), thickness=2)
                            )
                        except Exception:
                            pass
                    if person_obj.right_hand_landmarks:
                        try:
                            mp_drawing.draw_landmarks(
                                frame_disp,
                                person_obj.right_hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=2, circle_radius=3),
                                connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,256,121), thickness=2)
                            )
                        except Exception:
                            pass

                    # annotate detected action near bbox bottom
                    rx1, ry1, rx2, ry2 = bbox
                    cv2.rectangle(frame_disp, (rx1, ry2+2), (rx1 + 180, ry2+26), (0,0,0), -1)
                    cv2.putText(frame_disp, f"{action_label}", (rx1+6, ry2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 2)

                # Prepare top-right summary text from action_tally.
                # We'll prioritize showing Punch, Point, Wave, Clap, Both Hands Up, Hands Up, Fist, Unknown
                order = ["Punch", "Point", "Wave", "Clap", "Both Hands Up", "Hands Up", "Fist", "Unknown", "No person"]
                summary_items = []
                for key in order:
                    cnt = action_tally.get(key, 0)
                    if cnt > 0:
                        summary_items.append(f"{key}:{cnt}")
                if not summary_items:
                    summary_items = ["No actions"]

                summary_text = "  ".join(summary_items)
                # draw background rectangle top-right
                (text_w, text_h), _ = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                pad = 10
                box_x1 = max(0, full_w - text_w - 2*pad - 10)
                box_y1 = 6
                box_x2 = full_w - 6
                box_y2 = box_y1 + text_h + 2*pad//3
                cv2.rectangle(frame_disp, (box_x1, box_y1), (box_x2, box_y2), (0,0,0), -1)
                cv2.putText(frame_disp, summary_text, (box_x1 + pad//2, box_y1 + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

                # compute fps (smoothed)
                now2 = time.time()
                instantaneous_fps = 1.0 / (now2 - prev_time) if now2 != prev_time else 0.0
                prev_time = now2
                fps = FPS_SMOOTHING * fps + (1 - FPS_SMOOTHING) * instantaneous_fps

                # Draw FPS top-left
                fps_text = f"FPS: {int(fps)}"
                cv2.rectangle(frame_disp, (5,5), (150,35), (0,0,0), -1)
                cv2.putText(frame_disp, fps_text, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                cv2.imshow('Multi-person Action + 33-point Skeleton', frame_disp)
                frame_idx += 1

            # key handling
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            if key == ord('p'):
                paused = not paused
            if key == ord('l'):
                SHOW_LANDMARK_INDICES = not SHOW_LANDMARK_INDICES
                SHOW_LANDMARK_NAMES = not SHOW_LANDMARK_NAMES

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
