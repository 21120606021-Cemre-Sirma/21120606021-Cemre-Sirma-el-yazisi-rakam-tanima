Bu ödev 21120606021 numaralı Cemre Sırma Tarafından BIL485 dersi kapsamında hazırlanmış olan el yazısı rakam tanıma modeli oluşturma ödevidir.

Bu repoda, MNIST veri setiyle eğitilmiş MLP modelinin kodları ve kendi el yazımdan oluşan öğrenci numaramın test görselleri yer almaktadır.

Deney Sonucu Hakkında Genel Değerlendirme
Tarafınıza sunulmış olan bu el yazısı tanıma modeli tıpkı ödev isterlerinde belirtmiş olduğunuz gibi CNN mimarisi kullanılmadan ve sadece MLP mimarisi kullanılmak suretiyle hazırlanmıştır.
Tarafınıza sunulmış olan bu el yazısı tanıma modeli tıpkı ödev isterlerinde belirtmiş olduğunuz gibi önceden eğitilmiş hazır bir model kullanmamakta olup Düzleştirme ve Sıklaştırma katmanlarıyla sıfırdan eğitilmiştir.
Geliştirilen modeli kullanmak suretiyle tarafımca yapılan testlerde MNIST test seti üzerinde yaklaşık %97.5 doğruluk (accuracy) ve 0.10 kayıp (loss) değerleri elde edilmiştir.
Söz konusu bu modelin gerçek hayat koşullarında hakikaten nasıl bir davranış sergileyeceğini test etmek üzere sizin de ödev isterlerinde belirtmiş olduğunuz gibi kendi okul numarasını kalın bir keçeli kalem ile bir A4 kağıdına yazıp akabinde tarayıcı ile taratılmıştır.
Ki yine ödev isterlerinde belirtmiş olduğunuz gibi bu taramanın görselleri bu repo içerisinde tarafınıza sunulmuştur.
Geliştirilen Modelin kağıt üzerine yazmış olduğum numaraları ayırt edebilmesi için OpenCV ile Otsu eşikleme, erozyon ve ağırlık merkezi hizalaması gibi görüntü işleme metotları tarafımca kullanılmıştır.
Yapılan testin sonucunda, hazırlanmış model kağıt üzerine keçeli kalemle elle yazmış olduğum öğrenci numaramı başarıyla okumuş ve dışarıdan verdiğim görsellerde de pek ala tatmin edici performans göstermiştir. 
Dolayısıyla modelin MNIST test setindeki başarısıyla tutarlı olduğu söylenebilir.