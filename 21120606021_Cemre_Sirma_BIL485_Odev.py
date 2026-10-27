#21120606021 Numaralı Cemre Sırma isimli öğrenciye ait BIL485 dersi kapsamında yapılmış olan el yazısından rakam tanıma ödevidir.
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import cv2
import numpy as np
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
def turkish_summary(satir):
    satir = satir.replace("Total params", "Toplam parametre sayısı ve boyutu(KB)")
    satir = satir.replace("Trainable params", "Eğitilebilir parametre sayısı ve boyutu(KB)")
    satir = satir.replace("Non-trainable params", "Eğitilemeyen parametre sayısı ve boyutu(KB)")
    print(satir)
model.summary(print_fn=turkish_summary)
print("\nÖğrenme Safhası")
Start_Verildigi_Zaman = time.time()
Ogrenilmis_Seyler = model.fit(x_train, y_train, epochs=10, validation_split=0.2)
Finish_Yapildigi_Zaman = time.time()
Ogrenme_Zamanı = Finish_Yapildigi_Zaman - Start_Verildigi_Zaman
print(f"\nEğitim Zamanımız şu kadardır: {Ogrenme_Zamanı:.2f} saniyedir")
Loss_Degeri, Accuracy_Degeri = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTestin Loss Değeri Şudur: {Loss_Degeri:.4f}")
print(f"Testin Accuracy Değeri ise şudur: {Accuracy_Degeri:.4f}")
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(Ogrenilmis_Seyler.history['accuracy'], label='Öğrenim Sonrası Accuracy Değeri')
plt.plot(Ogrenilmis_Seyler.history['val_accuracy'], label='Validasyon Sonrası Accuracy Değeri')
plt.title('Modelin Accuracy Değeri')
plt.xlabel('Kaç Döngü Yapıldı')
plt.ylabel('Accuracy Değeri')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(Ogrenilmis_Seyler.history['loss'], label='Öğrenim Sonrası Loss Değeri')
plt.plot(Ogrenilmis_Seyler.history['val_loss'], label='Validasyon Sonrası Loss Değeri')
plt.title('Modelin Loss Değeri')
plt.xlabel('Kaç Döngü Yapıldı')
plt.ylabel('Loss Değeri')
plt.legend()
model.save('egitilmis_mlp_modeli.keras')
plt.show()
picture = cv2.imread('ogrenci_numaram.jpg', cv2.IMREAD_GRAYSCALE)
picture = cv2.medianBlur(picture, 3)
_, picture_reverse = cv2.threshold(picture, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
core = np.ones((2, 2), np.uint8)
resim_ters = cv2.erode(picture_reverse, core, iterations=2)
img_h, img_w = picture_reverse.shape
konturlar, _ = cv2.findContours(picture_reverse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
Dortgenler = [cv2.boundingRect(k) for k in konturlar]
Dortgenler = sorted(Dortgenler, key=lambda x: x[0])
Bulunan_Sayi = ""
islem_yapilmis_sayilar = []
for x_kordinati, y_kordinati, weight, height in Dortgenler:
    if 50 < height < (img_h - 20) and 15 < weight < (img_w / 2):
        sayi = picture_reverse[y_kordinati:y_kordinati + height, x_kordinati:x_kordinati + weight]
        if weight > height:
            yeni_weight = 20
            yeni_height = max(1, int((height / weight) * 20))
        else:
            yeni_height = 20
            yeni_weight = max(1, int((weight / height) * 20))
        kucuk_sayi = cv2.resize(sayi, (yeni_weight, yeni_height), interpolation=cv2.INTER_AREA)
        gecici_28 = np.zeros((28, 28), dtype=np.uint8)
        ust_taraftaki = (28 - yeni_height) // 2
        sol_taraftaki = (28 - yeni_weight) // 2
        gecici_28[ust_taraftaki:ust_taraftaki + yeni_height, sol_taraftaki:sol_taraftaki + yeni_weight] = kucuk_sayi
        M = cv2.moments(gecici_28)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            kaydirma_matrisi = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
            sayi_28x28_boyutunda = cv2.warpAffine(gecici_28, kaydirma_matrisi, (28, 28))
        else:
            sayi_28x28_boyutunda = gecici_28
        sayi_28x28_boyutunda = cv2.GaussianBlur(sayi_28x28_boyutunda, (3, 3), 0)
        islem_yapilmis_sayilar.append(sayi_28x28_boyutunda)
        model_icin_sayi = (sayi_28x28_boyutunda / 255.0).reshape(1, 28, 28)
        tahmini_deger = np.argmax(model.predict(model_icin_sayi, verbose=0))
        Bulunan_Sayi += str(tahmini_deger)
print(f"\nBulunan Sayı Şudur: {Bulunan_Sayi}")
plt.figure(figsize=(12, 2))
for i, sayinin_resmi in enumerate(islem_yapilmis_sayilar):
    plt.subplot(1, len(islem_yapilmis_sayilar), i + 1)
    plt.imshow(sayinin_resmi, cmap='gray')
    plt.axis('off')
plt.show()