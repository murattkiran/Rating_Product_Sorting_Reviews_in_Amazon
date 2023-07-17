
###################################################
# Rating Product & Sorting Reviews in Amazon
###################################################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

###################################################
# TASK 1: Calculate the Average Rating Based on Current Reviews and Compare it with the Existing Average Rating.
###################################################
# In the shared dataset, users have given scores and made comments on a product.
# The aim of this task is to evaluate the given scores by weighting them according to the date.
# It is necessary to compare the initial average score with the weighted rating obtained based on the date.
###################################################

###################################################
# Step 1: Read the Data Set and Calculate the Average Score of the Product.
###################################################
df = pd.read_csv("Rating Product&SortingReviewsinAmazon/amazon_review.csv")

"""def check_df(dataframe, head=5):
    print("############### Shape ################")
    print(dataframe.shape)
    print("########### Types ###############")
    print(dataframe.dtypes)
    print("########### Head ###############")
    print (dataframe.head(head))
    print ("########### Tail ###############" )
    print ( dataframe.tail(head))
    print ( "########### NA ###############" )
    print ( dataframe.isnull().sum())
    print ( "########### Quantiles ###############" )
    print ( dataframe.describe([0, 0.25, 0.50, 0.75]).T )
"""
check_df(df)

# Average Score
df["overall"].mean()   #4.587589013224822
df["overall"].value_counts()


###################################################
# Step 2: Calculate the Weighted Average Score Based on Date.
###################################################

df["day_diff"].sort_values(ascending=False).head()

df["day_diff"].describe().T

def time_based_weighted_average(dataframe, w1=30, w2=26, w3=24, w4=22):
    return dataframe.loc[dataframe["day_diff"] <= 281, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 281) & (dataframe["day_diff"] <= 431), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 431) & (dataframe["day_diff"] <= 601), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 601), "overall"].mean() * w4 / 100

time_based_weighted_average(df)
#4.689509022733296



###################################################
# Adım 3: Compare and Interpret the Average Scores for Each Time Period in the Weighted Rating.
###################################################
df.loc[df["day_diff"] <= 281, "overall"].mean()
#4.6957928802588995

df.loc[(df["day_diff"] > 281) & (df["day_diff"] <= 431), "overall"].mean()
#4.636140637775961

df.loc[(df["day_diff"] > 431) & (df["day_diff"] <= 601), "overall"].mean()
#4.571661237785016

df.loc[df["day_diff"] > 601, "overall"].mean()
#4.4462540716612375

# Değerlendirmeden itibaren geçen gün sayısı azaldıkça puan artmıştır
# bu üründe süreç içerisinde iyileştirme yapılmış olabilir ve teknolojik ürün olduğuna göre yazılım güncellemeleri
# ya da yeni sürümler üretilmiş olabilir.
# daha güncel olan yorumları daha çok ağırlıklandırdık. dolayısıyla ağırlıklı ortalamaya katkıları daha yüksek





###################################################
# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.
###################################################

###################################################
# Adım 1. helpful_no Değişkenini Üretiniz
###################################################
# Not:
# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.
#Toplam oy sayısından (total_vote) yararlı oy sayısı (helpful_yes) çıkarılarak yararlı bulunmayan oy sayılarını (helpful_no) bulunuz.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head()



###################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
###################################################
#score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayabilmek için score_pos_neg_diff,
#score_average_rating ve wilson_lower_bound fonksiyonlarını tanımlayınız.
#score_pos_neg_diff'a göre skorlar oluşturunuz. Ardından; df içerisinde score_pos_neg_diff ismiyle kaydediniz.
#score_average_rating'a göre skorlar oluşturunuz. Ardından; df içerisinde score_average_rating ismiyle kaydediniz.
# wilson_lower_bound'a göre skorlar oluşturunuz. Ardından; df içerisinde wilson_lower_bound ismiyle kaydediniz.


# score_pos_neg_diff
def score_pos_neg_diff(helpful_yes, helpful_no):
    return helpful_yes - helpful_no

df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

df.columns


# score_average_rating
def score_average_rating(helpful_yes, helpful_no):
    if helpful_yes + helpful_no == 0:
        return 0
    return helpful_yes / (helpful_yes + helpful_no)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

df.columns


# wilson_lower_bound
def wilson_lower_bound(helpful_yes, helpful_no, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    helpful_yes: int
        helpful_yes count
    helpful_no: int
        helpful_no count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = helpful_yes + helpful_no
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * helpful_yes / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df.columns



##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################
#wilson_lower_bound'a göre ilk 20 yorumu belirleyip sıralayanız. Sonuçları yorumlayınız.

df.sort_values("wilson_lower_bound", ascending=False).head(20)

# Tüm yöntemleri incelediğimizde, "score_pos_neg_diff" ve "score_average_rating" yöntemlerinin verilen puanlara bağlı olduğunu görüyoruz.
# Toplam oy sayısına göre değerlendirdiğimizde, tamamı olmasa da yüksek oy alanların "wlb" değerinin de üst sıralarda olduğunu fark ediyoruz.
# Bu hesaplamalarda en önemli faktör, sosyal kanıtı doğru bir şekilde yansıtmaktır.
# Bu nedenle, sıralama ve "wlb" değeri uyumlu görünmektedir.
# Öyle ki, kullanıcının yorumu faydalıysa, puanı en düşük olan yorum bile "wilson_lower_bound" sıralamasında üstlerde olabilir.
