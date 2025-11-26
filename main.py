import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import random

# 1. 관상 결과 리스트
fortune_results = [
    "대길! 오늘 하루 운이 엄청납니다.",
    "새로운 귀인을 만날 관상입니다.",
    "금전운이 따르는 복코시네요!",
    "예상치 못한 행운이 찾아옵니다.",
    "건강만 챙기면 만사형통입니다.",
    "오늘은 입조심을 해야 할 날입니다.",
    "주변 사람들에게 베풀면 배로 돌아옵니다.",
    "도화살이 있네요! 인기가 많겠습니다.",
    "리더십이 뛰어난 장군감입니다."
]

# 2. MediaPipe 설정
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 3. 한글 출력 함수
def putText_korean(img, text, position, font_size=30, color=(0, 255, 0)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("malgun.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# 4. 얼굴 특징 추출 함수 (세로 비율 기준)
def get_face_signature(landmarks):
    # 이마(10), 턱(152), 코끝(1), 왼쪽눈(33), 오른쪽눈(263), 왼쪽입꼬리(61), 오른쪽입꼬리(291)
    top = np.array([landmarks[10].x, landmarks[10].y])
    chin = np.array([landmarks[152].x, landmarks[152].y])
    nose = np.array([landmarks[1].x, landmarks[1].y])
    l_eye = np.array([landmarks[33].x, landmarks[33].y])
    r_eye = np.array([landmarks[263].x, landmarks[263].y])
    mouth_center = np.array([landmarks[13].x, landmarks[13].y]) 

    # 얼굴 세로 길이 (기준점)
    face_height = np.linalg.norm(top - chin)
    
    if face_height < 0.01:
        return None

    signature = []
    
    eye_center = (l_eye + r_eye) / 2
    # 1. 이마 ~ 눈 비율
    signature.append(np.linalg.norm(top - eye_center) / face_height)
    # 2. 눈 ~ 코 비율
    signature.append(np.linalg.norm(eye_center - nose) / face_height)
    # 3. 코 ~ 입 비율
    signature.append(np.linalg.norm(nose - mouth_center) / face_height)
    # 4. 입 ~ 턱 비율
    signature.append(np.linalg.norm(mouth_center - chin) / face_height)

    return np.array(signature)

# 5. 사용자 데이터베이스
known_users = []

# --- [수정됨] 유사도 임계값 ---
# 0.15로 설정 (깐깐하게 구분함)
SIMILARITY_THRESHOLD = 0.15

cap = cv2.VideoCapture(0)

print(f"프로그램 시작! (임계값: {SIMILARITY_THRESHOLD})")
print("'R' 리셋 / 'Q' 종료")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    current_fortune = None
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            current_signature = get_face_signature(face_landmarks.landmark)
            
            if current_signature is not None:
                # DB에서 가장 닮은 사람 찾기
                best_match_user = None
                min_dist = float('inf')

                for user in known_users:
                    dist = np.linalg.norm(user['signature'] - current_signature)
                    if dist < min_dist:
                        min_dist = dist
                        best_match_user = user
                
                # 기준치(0.15) 통과 여부 확인
                if min_dist < SIMILARITY_THRESHOLD and best_match_user is not None:
                    current_fortune = best_match_user['fortune']
                    found_match = True
                else:
                    # 새로운 사람 등록
                    new_fortune = random.choice(fortune_results)
                    known_users.append({
                        'signature': current_signature,
                        'fortune': new_fortune
                    })
                    current_fortune = new_fortune
                    found_match = False
                    print(f"신규 등록! (오차: {min_dist:.4f})")
                
                # 결과 출력
                h, w, c = image.shape
                cx, cy = int(face_landmarks.landmark[10].x * w), int(face_landmarks.landmark[10].y * h)
                
                if current_fortune:
                    color = (0, 255, 255) if found_match else (50, 255, 50) 
                    status = "[본인]" if found_match else "[신규]"
                    image = putText_korean(image, f"{status} {current_fortune}", (cx - 150, cy - 80), 25, color)

    cv2.putText(image, f"Users: {len(known_users)} | TH: {SIMILARITY_THRESHOLD}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow('AI Face Memory Fortune', image)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        known_users = []
        print("기억 초기화 완료")

cap.release()
cv2.destroyAllWindows()