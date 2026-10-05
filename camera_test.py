"""Camera diagnose: python camera_test.py"""
import os, cv2

backends = {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF, "ANY": cv2.CAP_ANY} if os.name == "nt" else {"ANY": cv2.CAP_ANY}
found = False
for i in range(4):
    for name, b in backends.items():
        cap = cv2.VideoCapture(i, b)
        ok = cap.isOpened() and cap.read()[0]
        print(f"index {i} | {name:5s} | {'WORKS' if ok else 'fail'}")
        found = found or ok
        cap.release()
print("\\nRESULT:", "kam se kam ek camera kaam kar raha hai" if found else
      "koi camera nahi khula -> Windows camera permission / doosri app check karo")
