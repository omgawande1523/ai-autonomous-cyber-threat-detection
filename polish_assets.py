import os
import shutil

def main():
    src_img = r"C:\Users\ASUS\.gemini\antigravity-ide\brain\78c68e74-c720-49c0-b7dc-f82ddb6e2788\dashboard_mockup_1779729241860.png"
    dest_dir = r"D:\cyber_threat_detection\assets"
    dest_img = os.path.join(dest_dir, "dashboard_mockup.png")
    
    os.makedirs(dest_dir, exist_ok=True)
    
    if os.path.exists(src_img):
        shutil.copy(src_img, dest_img)
        print(f"Copied mockup image to {dest_img}")
    else:
        print(f"Source mockup image not found at {src_img}")

if __name__ == "__main__":
    main()
