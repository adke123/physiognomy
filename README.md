# physiognomy
# 🔮 AI Face Fortune Teller (AI 관상가 양반)

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=Python&logoColor=white"/>
  <img src="https://img.shields.io/badge/MediaPipe-FaceMesh-00A6D6?style=for-the-badge&logo=google&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-Calculation-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
</div>

<br/>

**실시간 웹캠을 통해 사용자의 얼굴을 인식하고 관상(운세)을 분석해주는 AI 프로그램**입니다.
단순히 랜덤한 텍스트를 띄우는 것이 아니라, **얼굴의 고유 비율(Face Signature)**을 계산하여 사용자를 기억합니다. 한 번 관상을 본 사람은 다시 카메라에 비춰도 **동일한 운세**가 유지됩니다.

## 📸 Demo Screenshot
![App Screenshot](./screenshot.png)

## ✨ Key Features

* **👥 사용자 식별 (Face Memory)**:
    * 얼굴의 468개 랜드마크 중 주요 부위(눈, 코, 입, 턱)의 **세로 비율**을 계산하여 고유 벡터를 생성합니다.
    * 이전에 방문한 사람인지 DB에서 검색하여, 동일인에게는 **일관된 운세**를 제공합니다.
* **🔄 강건한 인식 (Robust Tracking)**:
    * 눈 사이 거리가 아닌 **얼굴 세로 길이(이마-턱)**를 기준으로 삼아, 사용자가 고개를 좌우로 돌려도 안정적으로 인식합니다.
* **🇰🇷 한글 지원**:
    * OpenCV의 한글 깨짐 문제를 해결하기 위해 `Pillow(PIL)`를 사용하여 예쁜 한글 폰트를 렌더링합니다.
* **⚙️ 실시간 제어**:
    * `R` 키를 눌러 저장된 사용자 기억을 초기화할 수 있습니다.

## 🛠️ Tech Stack & Algorithm

### 1. Face Signature (얼굴 특징 추출)
단순한 이미지 매칭이 아닌, **기하학적 비율(Geometric Ratios)**을 사용합니다.
* MediaPipe FaceMesh를 통해 얼굴 랜드마크(x, y) 추출
* `얼굴 전체 길이(이마~턱)`를 기준으로 `눈`, `코`, `입`의 상대적 위치 비율 계산
* 이 방식은 카메라와의 거리(줌인/줌아웃)가 달라져도 비율이 유지되므로 인식률이 높습니다.

### 2. Similarity Measurement (유사도 측정)
* **유클리드 거리(Euclidean Distance)**를 사용하여 저장된 얼굴 벡터와 현재 얼굴 벡터를 비교합니다.
* **Threshold: 0.15** (이 값보다 오차가 작으면 동일인으로 판단)

