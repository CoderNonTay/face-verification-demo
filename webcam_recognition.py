import cv2
import numpy as np
from insightface.app import FaceAnalysis

# -----------------------------
# Utils
# -----------------------------
def cosine_sim(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

# -----------------------------
# Init model
# -----------------------------
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

# -----------------------------
# Config
# -----------------------------
THRESHOLD = 0.7   # chuẩn ngân hàng
stored_embedding = None

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(1)

print("📷 Hướng dẫn:")
print("  S : Lưu khuôn mặt (Enroll)")
print("  R : Nhận diện")
print("  Q : Thoát")

mode = "IDLE"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        emb = face.embedding

        # Vẽ khung
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if stored_embedding is not None:
            sim = cosine_sim(emb, stored_embedding)

            if sim >= THRESHOLD:
                text = f"MATCH ({sim:.2f})"
                color = (0, 255, 0)
            else:
                text = f"NOT MATCH ({sim:.2f})"
                color = (0, 0, 255)

            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
        else:
            cv2.putText(
                frame,
                "No face enrolled",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

    cv2.imshow("Face Recognition - Webcam", frame)

    key = cv2.waitKey(1) & 0xFF

    # Enroll
    if key == ord("s") and faces:
        stored_embedding = faces[0].embedding
        print("✅ Face enrolled")

    # Exit
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
