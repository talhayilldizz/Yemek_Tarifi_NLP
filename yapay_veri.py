import pandas as pd
import random

# Orijinal veri
df = pd.read_csv("datasets/tr_yemekler_temiz.csv")

# Yeni satırları tutmak için liste
new_rows = []

# Varyasyon sayısı (her yemekten kaç yeni varyasyon üretilecek)
VARYASYON_SAYISI = 5

for _, row in df.iterrows():
    food = row["foodname"]
    materials = row["materials"].split(",")
    
    for i in range(VARYASYON_SAYISI):
        temp = materials.copy()

        # %30 ihtimalle 1 malzeme çıkar
        if len(temp) > 3 and random.random() < 0.3:
            temp.remove(random.choice(temp))

        # %50 ihtimalle yeni bir malzeme ekle
        extras = ["tuz", "karabiber", "zeytinyağı", "tereyağı", "salça", "maydanoz", "yoğurt", "sarımsak", "limon", "baharat"]
        if random.random() < 0.5:
            extra = random.choice(extras)
            if extra not in temp:
                temp.append(extra)

        # %40 ihtimalle sıralamayı değiştir
        if random.random() < 0.4:
            random.shuffle(temp)

        # Yeni varyasyonu ekle
        new_rows.append({"foodname": food, "materials": ",".join(temp)})

# Yeni verileri DataFrame'e çevir
df_aug = pd.DataFrame(new_rows)

# Orijinal veriyle birleştir
df_final = pd.concat([df, df_aug], ignore_index=True)

# Dosyayı kaydet
df_final.to_csv("datasets/tr_yemekler_arttirilmis.csv", index=False)

print(f"✅ Yeni veri seti oluşturuldu: {len(df_final)} satır (orijinal {len(df)} → yeni {len(df_final)})")
print(df_final.head(10))
