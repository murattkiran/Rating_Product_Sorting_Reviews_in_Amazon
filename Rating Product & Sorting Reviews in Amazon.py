
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
# Step 3: Compare and Interpret the Average Scores for Each Time Period in the Weighted Rating.
###################################################
df.loc[df["day_diff"] <= 281, "overall"].mean()
#4.6957928802588995

df.loc[(df["day_diff"] > 281) & (df["day_diff"] <= 431), "overall"].mean()
#4.636140637775961

df.loc[(df["day_diff"] > 431) & (df["day_diff"] <= 601), "overall"].mean()
#4.571661237785016

df.loc[df["day_diff"] > 601, "overall"].mean()
#4.4462540716612375

# The score increased as the number of days since the review decreased.
# We have given more weight to more recent comments. Therefore, their contribution to the weighted average is higher.



###################################################
# Task 2: Specify 20 Reviews for the Product to be Displayed on the Product Detail Page.
###################################################

###################################################
# Step 1. Generate the helpful_no variable
###################################################
# Note:
# total_vote represents the total number of up-down votes for a review.
# up means helpful.
# There is no helpful_no variable in the data set, it must be generated over existing variables.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head()



###################################################
# Step 2. Calculate score_pos_neg_diff, score_average_rating and wilson_lower_bound Scores and add to dataframe
###################################################

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
    n = helpful_yes + helpful_no
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * helpful_yes / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df.columns


##################################################
# Step 3. Identify 20 Comments and Interpret Results.
###################################################

df.sort_values("wilson_lower_bound", ascending=False).head(20)

# When examining all the methods, we can see that "score_pos_neg_diff" and "score_average_rating" are dependent on the given scores.
# When we evaluate it according to the total number of votes, we realize that the "wlb" value of those who received the highest votes, if not all, is also in the upper ranks.
# The most important factor in these calculations is to accurately reflect social proof.
# Therefore, the sort and the "wlb" value appear to be compatible.
# In fact, if a user's review is helpful, even the review with the lowest score can be ranked high in the "wilson_lower_bound" ranking.
