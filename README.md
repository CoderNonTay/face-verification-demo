# Face Recognition Demo – From Basics to Near-Production

Repo này trình bày **toàn bộ lộ trình học và triển khai Face Recognition** của tác giả, được chia thành **4 mini project tương ứng với 4 giai đoạn nâng cấp**: từ hiểu bản chất embedding cho tới mô phỏng **hệ thống Face Verification gần với sản phẩm thực tế trong ngân hàng / KYC**.

README này **kết hợp**:
- Phần mô tả hệ thống, kiến trúc, security, privacy (từ README ban đầu)
- Phần phân chia **stage rõ ràng theo từng file** (README multi-stage)

Mục tiêu không phải khoe code, mà là **thể hiện tư duy hệ thống AI đúng chuẩn doanh nghiệp**.

---

## 🧠 Tổng quan bài toán Face Recognition

Face Recognition hiện đại **không phải là bài toán phân loại (classification)**, mà là bài toán **metric learning**:

- Model học cách ánh xạ khuôn mặt → **embedding vector (512 chiều)**
- Hai khuôn mặt được so sánh bằng **cosine similarity**
- Không cần train lại model khi thêm người mới

Các khái niệm cốt lõi:
- **Embedding**: vector đặc trưng đại diện cho khuôn mặt
- **Cosine similarity**: độ giống nhau giữa 2 embedding
- **Verify (1:1)**: xác thực danh tính
- **Identify (1:N)**: nhận diện trong tập người đã biết

---

## 🧩 Tổng quan cấu trúc repo

```
Demo/
├── test_face.py                     # Stage 1
├── verify_identify_demo.py          # Stage 2
├── verify_identify_demo_advance.py  # Stage 3
├── webcam_recognition.py            # Stage 4
├── requirements.txt
└── db/                              # Lưu embedding (.npy)
```

---

# 🔹 Stage 1 – Face Embedding Fundamentals
### 📄 File: `test_face.py`

### 🎯 Mục tiêu
Xây dựng **nền tảng tư duy đúng** về Face Recognition:
- Model không "nhận diện ID"
- Model chỉ sinh ra embedding

### ✨ Chức năng
- Load pre-trained model InsightFace (ArcFace)
- Detect khuôn mặt trong ảnh
- Trích xuất embedding 512 chiều
- Tính cosine similarity giữa 2 khuôn mặt

### 📚 Kiến thức đạt được
- Embedding là gì và vì sao cần normalize
- Cosine similarity **không phải %**
- Vì sao cùng 1 người nhưng similarity không cố định

👉 Đây là **bước bắt buộc** trước khi làm bất kỳ hệ thống Face Recognition nào.

---

# 🔹 Stage 2 – Identify (1:N) vs Verify (1:1)
### 📄 File: `verify_identify_demo.py`

### 🎯 Mục tiêu
Phân biệt **2 bài toán hoàn toàn khác nhau trong thực tế**:

| Bài toán | Câu hỏi |
|-------|-------|
| Identify (1:N) | "Người này là ai trong DB?" |
| Verify (1:1) | "Người này có phải X không?" |

### ✨ Chức năng
- Lưu embedding vào DB (.npy)
- Identify (1:N):
  - So sánh embedding với toàn bộ DB
  - Trả về ID giống nhất nếu vượt threshold
- Verify (1:1):
  - So sánh embedding với **1 ID được chỉ định**

### 📚 Kiến thức đạt được
- Vì sao **banking/KYC không dùng Identify**
- Verify (1:1) là chuẩn xác thực danh tính
- Threshold phụ thuộc bài toán

---

# 🔹 Stage 3 – System Thinking & Secure Design
### 📄 File: `verify_identify_demo_advance.py`

### 🎯 Mục tiêu
Chuyển từ **demo ML** sang **mini system**:
- Rõ state
- Rõ luồng nghiệp vụ
- Có kiểm soát rủi ro

### ✨ Chức năng
- Tách rõ các pha:
  - Enroll
  - Verify
  - Identify
- Chuẩn hoá embedding (L2 normalization)
- Kiểm soát threshold theo mode

### 📚 Kiến thức đạt được
- Vì sao phải normalize embedding
- Vì sao không brute-force DB lớn
- Tư duy **security-first trong AI system**

---

# 🔹 Stage 4 – Near-Production Face Verification System
### 📄 File: `webcam_recognition.py`

### 🎯 Mục tiêu
Mô phỏng **hệ thống xác thực khuôn mặt gần với sản phẩm thật**:
- Real-time webcam
- Có UI
- Có state machine
- Có audit logic

### ✨ Chức năng
- Webcam face recognition real-time
- 3 chế độ hoạt động:
  - **Enroll**: đăng ký người mới
  - **Verify (1:1)**: xác thực danh tính (chuẩn banking)
  - **Identify (1:N)**: demo
- Popup UI nhập ID
- Kiểm soát:
  - ID trùng
  - ID không tồn tại
- Lưu embedding vào DB

### 📚 Kiến thức đạt được
- Luồng verify chuẩn:
  ```
  User nhập ID
  → Load embedding
  → Camera capture
  → Compare
  → PASS / FAIL
  ```
- Vì sao face chỉ là **1 yếu tố xác thực**
- Privacy-aware design

---

## 📊 Cosine Similarity & Threshold

- Cosine similarity ∈ [-1, 1]
- Không phải phần trăm

| Giá trị | Ý nghĩa |
|------|-------|
| > 0.8 | Rất giống |
| 0.7–0.8 | Chấp nhận |
| < 0.6 | Khác người |

Ngưỡng tham khảo:
- Verify: ~0.75
- Identify: ~0.7

---

## 🔐 Bảo mật & Quyền riêng tư (Privacy)

- Không lưu ảnh khuôn mặt
- Không log embedding vector
- DB chỉ chứa embedding đã chuẩn hoá
- Có thể mở rộng:
  - Rate limit
  - Account lock
  - Anti-spoofing

---

## 📝 Audit & Security Logging

Hệ thống có thể log các sự kiện:
- Verify PASS / FAIL
- Nhập ID không tồn tại

Ví dụ:
```
2025-01-15 19:22:10 | VERIFY_ID_NOT_FOUND | input_id=admin
```

👉 Phục vụ audit & phát hiện hành vi bất thường.

---

## 🛠 Công nghệ sử dụng

- Python 3.10
- InsightFace (ArcFace, SCRFD)
- OpenCV
- NumPy

---

## 🚀 Cách chạy

```bash
pip install -r requirements.txt
python webcam_recognition.py
```

---

## 📈 Hướng mở rộng

- Anti-spoofing (ảnh / video)
- Face quality gate
- Vector DB (FAISS / Milvus)
- REST API backend
- Multi-factor authentication

---

## 🎯 Tổng kết lộ trình học

| Stage | Trọng tâm |
|----|----|
| 1 | Hiểu embedding |
| 2 | Verify vs Identify |
| 3 | System & security |
| 4 | Near-production demo |

---

## 👤 Tác giả

**Đào Danh Đăng Phụng**  
Computer Science Graduate

> Repo phục vụ học tập, demo kỹ thuật và định hướng xây dựng hệ thống AI trong môi trường doanh nghiệp.

