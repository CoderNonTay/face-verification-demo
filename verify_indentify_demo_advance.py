import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis

# ================== CONFIG ==================
DB_PATH = "db"
ENROLL_FRAMES = 20
IDENTIFY_THRESHOLD = 0.7
VERIFY_THRESHOLD = 0.75

# ================== INIT ==================
os.makedirs(DB_PATH, exist_ok=True)

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture(1)

# ================== UTILS ==================
def normalize(v):
    return v / np.linalg.norm(v)

def cosine_sim(a, b):
    return np.dot(a, b)

def load_db():
    db = {}
    for f in os.listdir(DB_PATH):
        if f.endswith(".npy"):
            uid = f.replace(".npy", "")
            db[uid] = np.load(os.path.join(DB_PATH, f))
    return db

def input_id_popup(existing_ids, mode="enroll"):
    text = ""
    error_message = ""

    while True:
        img = np.zeros((220, 440, 3), dtype=np.uint8)

        title = "Enroll - Enter NEW ID" if mode == "enroll" else "Verify - Enter EXISTING ID"
        cv2.putText(img, title, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(img, text, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if error_message:
            cv2.putText(img, error_message, (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(img, "ENTER=OK | ESC=Cancel", (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Input ID", img)
        key = cv2.waitKey(30)

        if key == 27:  # ESC
            cv2.destroyWindow("Input ID")
            return None

        elif key == 13:  # ENTER
            if text == "":
                error_message = "ID cannot be empty"
                continue

            if mode == "enroll":
                if text in existing_ids:
                    error_message = "ID already exists"
                    text = ""
                    continue

            if mode == "verify":
                if text not in existing_ids:
                    error_message = "ID not found, please enter again"
                    text = ""
                    continue

            cv2.destroyWindow("Input ID")
            return text

        elif key == 8:  # BACKSPACE
            text = text[:-1]

        elif 32 <= key <= 126:
            text += chr(key)



def identify(emb, db):
    best_id = "Unknown"
    best_score = IDENTIFY_THRESHOLD
    for uid, ref in db.items():
        score = cosine_sim(emb, ref)
        if score > best_score:
            best_score = score
            best_id = uid
    return best_id, best_score

def verify(emb, db, uid):
    if uid not in db:
        return False, 0.0
    score = cosine_sim(emb, db[uid])
    return score > VERIFY_THRESHOLD, score

# ================== STATE ==================
mode = "IDENTIFY"   # IDENTIFY | ENROLL | VERIFY
db = load_db()

enroll_embeddings = []
current_user_id = None
verify_user_id = None

print("""
📷 Controls:
 S : Enroll
 I : Identify
 V : Verify
 Q : Quit
""")

# ================== MAIN LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    label = ""
    color = (0, 255, 0)

    if len(faces) > 0:
        face = faces[0]
        emb = normalize(face.embedding)

        x1, y1, x2, y2 = map(int, face.bbox)

        if mode == "ENROLL":
            enroll_embeddings.append(emb)
            label = f"Enrolling {current_user_id}: {len(enroll_embeddings)}/{ENROLL_FRAMES}"

            if len(enroll_embeddings) >= ENROLL_FRAMES:
                mean_emb = normalize(np.mean(enroll_embeddings, axis=0))
                np.save(f"{DB_PATH}/{current_user_id}.npy", mean_emb)
                db[current_user_id] = mean_emb
                print(f"✅ Saved {current_user_id}")
                mode = "IDENTIFY"

        elif mode == "IDENTIFY":
            uid, score = identify(emb, db)
            label = f"{uid} ({score:.2f})"

        elif mode == "VERIFY":
            ok, score = verify(emb, db, verify_user_id)
            label = f"{verify_user_id}: {'PASS' if ok else 'FAIL'} ({score:.2f})"
            color = (0, 255, 0) if ok else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, f"MODE: {mode}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Face System", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('s') and mode != "ENROLL":
        uid = input_id_popup(db.keys(), mode="enroll")
        if uid:
            enroll_embeddings = []
            current_user_id = uid
            mode = "ENROLL"

    elif key == ord('v'):
        uid = input_id_popup(db.keys(), mode="verify")
        if uid is None:
            mode = "IDENTIFY"   # hủy hoặc nhập sai
        else:
            verify_user_id = uid
            mode = "VERIFY"

 
# ================== CLEANUP ==================
cap.release()
cv2.destroyAllWindows()
