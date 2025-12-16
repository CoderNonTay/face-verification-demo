import cv2
import numpy as np
from insightface.app import FaceAnalysis

# 1. Khởi tạo model
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

# 2. Hàm lấy embedding từ ảnh
def get_embedding(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)

    if len(faces) == 0:
        print(f"Không phát hiện mặt trong ảnh {image_path}")
        return None

    return faces[0].embedding

# 3. Lấy embedding 2 ảnh
emb1 = get_embedding("images/face1.jpg")
emb2 = get_embedding("images/face3.jpg")

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

# 4. So sánh cosine similarity
if emb1 is not None and emb2 is not None:
    similarity = cosine_similarity(emb1, emb2)
    print(f"🔍 Cosine similarity: {similarity:.4f}")

    if similarity > 0.7:
        print("Cùng người (khả năng cao)")
    else:
        print("Khác người")