import cv2
import numpy as np
import math


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Hata: Kamera açılamadı!")
        return

    # Vuruşları saklayacağımız liste: [{'pos': (x,y), 'score': puan}]
    hits = []

    # Lazer parlaklık eşiği
    brightness_threshold = 200

    # Lazerin bir önceki karede yanıp yanmadığını takip eden değişken
    # Bu sayede tetiğe basılı tutulsa bile (lazer sürekli yansa bile)
    # her basışta tek bir atış olarak sayılmasını sağlayacağız.
    laser_was_on = False

    print("Lazer Vuruş Tespit Sistemi (Hedef Tahtalı) Başlatıldı.")
    print("Çıkış yapmak için 'q', ekranı temizlemek için 'c' tuşuna basın.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kameradan görüntü alınamadı.")
            break

        # Ayna etkisi
        frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]
        center = (width // 2, height // 2)

        # --- Hedef Tahtası Özellikleri ---
        max_radius = min(height, width) // 3  # Ekranın kısa kenarının 1/3'ü kadar büyük
        num_rings = 5  # 5 adet iç içe daire
        ring_step = max_radius // num_rings

        # --- Lazer Tespiti ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        v_channel = hsv[:, :, 2]
        masked_v = cv2.bitwise_and(v_channel, v_channel, mask=red_mask)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(masked_v)

        is_laser_on = max_val >= brightness_threshold

        # Lazer algılandıysa VE bir önceki karede yanmıyorsa (yeni "tetik" çekildiyse)
        if is_laser_on and not laser_was_on:
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
            hits.append({"pos": max_loc, "score": hit_score})

        # Bir sonraki karede kontrol edebilmek için lazerin güncel durumunu kaydet
        laser_was_on = is_laser_on

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
        for hit in hits:
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
            f"Vurus Sayisi: {len(hits)}",
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
        cv2.putText(
            frame,
            "Temizle: C | Cikis: Q",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        cv2.imshow("Lazer Vurus Tespiti - Hedef Modu", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            hits.clear()
            laser_was_on = False  # Temizleyince tetik mekanizmasını da sıfırla

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
