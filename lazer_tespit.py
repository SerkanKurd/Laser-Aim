import cv2
import numpy as np
import math
import json
import os

SETTINGS_FILE = "settings.json"

class LaserDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.hits = []
        self.laser_was_on = False
        
        # Önce varsayılanları yükle
        self.restore_defaults(save=False)
        # Varsa dosyadan ayarları ez
        self.load_settings()

    def restore_defaults(self, save=True):
        self.brightness_threshold = 200
        self.target_scale = 1.0
        self.mirror_effect = False
        self.laser_color = 'red'
        if save:
            self.save_settings()

    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r') as f:
                    data = json.load(f)
                self.update_config(data, save=False)
            except Exception as e:
                print(f"Ayarlar yuklenirken hata: {e}")

    def save_settings(self):
        try:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(self.get_config(), f, indent=4)
        except Exception as e:
            print(f"Ayarlar kaydedilirken hata: {e}")

    def get_config(self):
        return {
            "brightness_threshold": self.brightness_threshold,
            "target_scale": self.target_scale,
            "mirror_effect": self.mirror_effect,
            "laser_color": self.laser_color
        }

    def update_config(self, data, save=True):
        if 'brightness_threshold' in data:
            self.brightness_threshold = int(data['brightness_threshold'])
        if 'target_scale' in data:
            self.target_scale = float(data['target_scale'])
        if 'mirror_effect' in data:
            self.mirror_effect = bool(data['mirror_effect'])
        if 'laser_color' in data:
            self.laser_color = str(data['laser_color'])
            
        if save:
            self.save_settings()

    def clear_hits(self):
        self.hits.clear()
        self.laser_was_on = False

    def generate_frames(self):
        if not self.cap.isOpened():
            print("Hata: Kamera açılamadı!")
            # Boş bir resim döndürelim
            blank_image = np.zeros((480, 640, 3), np.uint8)
            ret, buffer = cv2.imencode('.jpg', blank_image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if self.mirror_effect:
                frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]
            center = (width // 2, height // 2)

            # --- Hedef Tahtası Özellikleri ---
            base_radius = min(height, width) // 3
            max_radius = max(10, int(base_radius * self.target_scale))  # Sıfır veya negatif olmasını engelle
            num_rings = 5  # 5 adet iç içe daire
            ring_step = max_radius // num_rings

            # --- Lazer Tespiti ---
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            if self.laser_color == 'green':
                lower_green = np.array([40, 100, 100])
                upper_green = np.array([80, 255, 255])
                color_mask = cv2.inRange(hsv, lower_green, upper_green)
            else: # red
                lower_red1 = np.array([0, 100, 100])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([160, 100, 100])
                upper_red2 = np.array([180, 255, 255])
                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                color_mask = cv2.bitwise_or(mask1, mask2)

            v_channel = hsv[:, :, 2]
            masked_v = cv2.bitwise_and(v_channel, v_channel, mask=color_mask)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(masked_v)

            is_laser_on = max_val >= self.brightness_threshold

            # Lazer algılandıysa VE bir önceki karede yanmıyorsa (yeni "tetik" çekildiyse)
            if is_laser_on and not self.laser_was_on:
                # Vuruş noktasının ekran merkezine (hedefin merkezi) olan uzaklığını hesapla
                distance = math.hypot(max_loc[0] - center[0], max_loc[1] - center[1])

                hit_score = 0
                # İçten dışa doğru mesafeyi kontrol ederek puanı belirle
                for i in range(1, num_rings + 1):
                    if distance <= i * ring_step:
                        # En içteki daire (i=1) için 50, en dıştaki (i=5) için 10 puan
                        hit_score = (num_rings - i + 1) * 10
                        break

                # Bulunan vuruşu listeye ekle (Daire dışında kalsa bile 0 puanla karavana olarak ekler)
                self.hits.append({"pos": max_loc, "score": hit_score})

            # Bir sonraki karede kontrol edebilmek için lazerin güncel durumunu kaydet
            self.laser_was_on = is_laser_on

            # --- Çizim İşlemleri ---

            # 1. Hedef Tahtasını Çiz
            # Dıştan içe doğru çiziyoruz ki yazılar iç içe daha düzgün gözüksün
            for i in range(num_rings, 0, -1):
                radius = i * ring_step
                ring_score = (num_rings - i + 1) * 10
                cv2.circle(
                    frame, center, radius, (255, 0, 0), 2
                )  # Mavi renkte hedef halkaları

                # Halkaların üzerine puanları yazdır
                cv2.putText(
                    frame,
                    str(ring_score),
                    (center[0] - 10, center[1] - radius + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

            # Merkeze ufak bir artı (+) çiz (Tam 12'den vurma noktası - Kırmızı)
            cv2.line(
                frame,
                (center[0] - 10, center[1]),
                (center[0] + 10, center[1]),
                (0, 0, 255),
                2,
            )
            cv2.line(
                frame,
                (center[0], center[1] - 10),
                (center[0], center[1] + 10),
                (0, 0, 255),
                2,
            )

            # 2. Vuruşları Çiz ve Toplam Puanı Hesapla
            total_score = 0
            for hit in self.hits:
                pos = hit["pos"]
                score = hit["score"]
                total_score += score

                # Vuruş karavana (0 puan) ise kırmızı, isabetli ise yeşil renk
                color = (0, 255, 0) if score > 0 else (0, 0, 255)
                cv2.circle(frame, pos, 5, color, -1)

                # Vuruşun kaç puan aldığını noktanın yanına yaz
                cv2.putText(
                    frame,
                    f"+{score}",
                    (pos[0] + 10, pos[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            # Anlık lazer konumu vizörü (Kamerada sadece lazerin yerini görmek için anlık beyaz çember)
            if is_laser_on:
                cv2.circle(frame, max_loc, 8, (255, 255, 255), 2)

            # 3. Bilgileri Sol Üst Köşeye Yazdır
            cv2.putText(
                frame,
                f"Vurus Sayisi: {len(self.hits)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Toplam Puan: {total_score}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            # Web üzerinden akış sağlayacağımız için frame'i JPEG formatına çeviriyoruz.
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def get_stats(self):
        total_score = sum(hit['score'] for hit in self.hits)
        return {
            "hit_count": len(self.hits),
            "total_score": total_score
        }

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
