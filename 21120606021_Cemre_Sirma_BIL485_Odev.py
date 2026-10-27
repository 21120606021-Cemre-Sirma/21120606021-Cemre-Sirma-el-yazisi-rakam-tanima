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
Start = time.time()
Learned = model.fit(x_train, y_train, epochs=10, validation_split=0.2)
Finish = time.time()
Time = Finish - Start
print(f"\nEğitim Zamanımız şu kadardır: {Time:.2f} saniyedir")
LossDeg, AccuracyDeg = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTestin Loss Değeri Şudur: {LossDeg:.4f}")
print(f"Testin Accuracy Değeri ise şudur: {AccuracyDeg:.4f}")
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(Learned.history['accuracy'], label='Öğrenim Sonrası Accuracy Değeri')
plt.plot(Learned.history['val_accuracy'], label='Validasyon Sonrası Accuracy Değeri')
plt.title('Modelin Accuracy Değeri')
plt.xlabel('Kaç Döngü Yapıldı')
plt.ylabel('Accuracy Değeri')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(Learned.history['loss'], label='Öğrenim Sonrası Loss Değeri')
plt.plot(Learned.history['val_loss'], label='Validasyon Sonrası Loss Değeri')
plt.title('Modelin Loss Değeri')
plt.xlabel('Kaç Döngü Yapıldı')
plt.ylabel('Loss Değeri')
plt.legend()
model.save('egitilmis_mlp_modeli.keras')
plt.show()
pict = cv2.imread('ogrenci_numaram.jpg', cv2.IMREAD_GRAYSCALE)
pict = cv2.medianBlur(pict, 3)
_, rev = cv2.threshold(pict, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
core = np.ones((2, 2), np.uint8)
rev = cv2.erode(rev, core, iterations=2)
pict_h, pict_w = rev.shape
kont, _ = cv2.findContours(rev, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rect = [cv2.boundingRect(k) for k in kont]
rect = sorted(rect, key=lambda x: x[0])
fnd = ""
prog = []
for i, j, k, l in rect:
    if 50 < l < (pict_h - 20) and 15 < k < (pict_w / 2):
        sayi = rev[j:j + l, i:i + k]
        if k > l:
            kn = 20
            ln = max(1, int((l / k) * 20))
        else:
            ln = 20
            kn = max(1, int((k / l) * 20))
        sml = cv2.resize(sayi, (kn, ln), interpolation=cv2.INTER_AREA)
        temp = np.zeros((28, 28), dtype=np.uint8)
        up = (28 - ln) // 2
        left = (28 - kn) // 2
        temp[up:up + ln, left:left + kn] = sml
        M = cv2.moments(temp)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            mtrx = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
            numb28 = cv2.warpAffine(temp, mtrx, (28, 28))
        else:
            numb28 = temp
        numb28 = cv2.GaussianBlur(numb28, (3, 3), 0)
        prog.append(numb28)
        modlnumb = (numb28 / 255.0).reshape(1, 28, 28)
        pred = np.argmax(model.predict(modlnumb, verbose=0))
        fnd += str(pred)
print(f"\nBulunan Sayı Şudur: {fnd}")
plt.figure(figsize=(12, 2))
for i, pict1 in enumerate(prog):
    plt.subplot(1, len(prog), i + 1)
    plt.imshow(pict1, cmap='gray')
    plt.axis('off')
plt.show()
