import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

st.title("Movie Recommender Data Analysis")
df = pd.read_csv(r'C:\Users\User\Downloads\What_I_have_to_know\Computer_Sciences\Data_Structure_and_Programming\I3\I3_S2\PDS\PDS_Final_Project\movieDataRemoved3.csv')

st.write("Displaying dataset:")
st.dataframe(df)

st.header("Dataset information:")
st.text("""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 9836 entries, 0 to 9835
Data columns (total 9 columns):
 #   Column                Non-Null Count  Dtype 
---  ------                --------------  ----- 
 0   userID                9836 non-null   int64 
 1   userRating            9836 non-null   int64 
 2   Name_of_Movie         9836 non-null   object
 3   Genre                 9836 non-null   object
 4   Year_of_Release       9836 non-null   object
 5   Watch_time            9836 non-null   int64 
 6   Meta_Score            7868 non-null   object
 7   Votes                 7099 non-null   object
 8   Gross_Collection(M$)  7099 non-null   object
dtypes: int64(3), object(6)
memory usage: 691.7+ KB

There are a couple of things not right about the nature of the following features:
- Year_of_Release
- Watch_time
- Meta_Score
- Gross_Collection(M$)

We converted them up using a series of functions that can easily be hand coded. And secondly, we converted **** to NaN 
since NaN is more easily handled by Pandas and Numpy than string types.
""")

st.header("Missing values analysis")
st.subheader("Number of missing values in our dataset")

st.text("""
 Now we will check for any missing values in our dataset. We check for any rows whose features' values contain `NaN`. 
 Even if one of the movie's feature contains `Nan`, we will also include that into our "num_missing_rows" variables.
""")
st.image('missing_plot_whole.png')

st.image('missing_per_column.png')

st.markdown("""
As we can see so far, there are $34$ % of the entire movies that contain missing values in at least one of its features....that is a lot.
And column-wise we have $20.008133$% of movies which do have missing meta scores and $27.826352$ % of movies which have no gross collection and votes. That is a quarter of all our data set, removing those movies will heavily impact the data analysis of the features' distrbution. Moreover, we cannot just ignore missing values. So, we will resort to imputing them with either the mean, median or the mode. Here is a brief overview of when to impute missing values with the mean and median and the mode:

1. Mean: The mean is a good choice for imputing missing values when the data is normally distributed or approximately symmetric. However, if the data has extreme outliers or is skewed, the mean may not be the best choice as it can be heavily influenced by these outliers. In such cases, you may want to consider using the median instead.

2. Median: The median is a good choice for imputing missing values when the data is skewed or has extreme outliers. Since the median is not affected by extreme values, it can provide a more robust estimate of the central tendency of the data. However, if the data is symmetric and not too heavily skewed, the mean may be a better choice.

3. Mode: The mode is a good choice for imputing missing values when the data is categorical or discrete. For example, if you have a dataset of eye colors and some values are missing, you can impute the missing values with the mode (i.e., the most frequent eye color). However, if the data is continuous or numerical, the mean or median may be more appropriate.

So to know which one, let's plot the data distributions of the movies' Meta Score, Votes, and Gross Collection($). Firstly, it is obvious that since the 3 mentioned features are numerical, then imputing their instances with the `mode` is not logical so `mode` is out of the option leaving only median and mean.
""")

st.subheader("Below are the data distributions before and after square root transformation.")

st.image("2_votes.png")
st.image("2_meta.png")
st.image("2_gross.png")

st.subheader("Below are the data distributions before and after log transformation.")

st.image("log_votes.png")
st.image("log_meta.png")
st.image("log_gross.png")

st.markdown("""
### Although the skewness has gotten better, we believe that working with the transformed data will bring more challenges such as :
- transformed variables are sometimes harder to interpret
- if the measure is commonly known and understood, such as sales dollars, the multiplicative inverse of sales dollars can be harder to interpret and communicate
- making meaningful predictions often means inverting the transformed variables back to the original form. 
- sometimes, selecting the best transformation can be a process of trial and error.
- transformations can sometimes overshoot the goal and create issues in the other direction; must verify and recheck
- grouped data, values that are zero/neg required additional steps
- can risk unintended consequences in variable relationships
""")

st.markdown("""
**However, all those works are not useless. One inference that we can take from the above is that for Meta_Score, we can impute the NaN values with the mean and for Votes and Gross_Collection($), we can impute their values with the median due to their relentless skewness.**
meta_score_mean = 59
votes_median = 47591
gross_median = 16.94
""")

summary_stats = """
userID  userRating  Year_of_Release  Watch_time  Meta_Score  Votes     Gross_Collection(M$)
count   9836        9836             9836        9836        9836      9836
mean    4918        3                2001        110         58        33
std     2839        1                18          22          15        58
min     1           1                1915        45          7         0
25%     2459        2                1994        96          50        6
50%     4918        3                2006        106         59        16
75%     7377        4                2014        120         68        33
max     9836        5                2023        439         100       936
"""
st.subheader("After some clearning, we have a look at the summary statistics table")
st.text(summary_stats)



st.header("Outlier analysis")


st.markdown("""
Firstly, as Professor Sokkhey told us....in data science, there is not a one-size-fits-all method. That still holds in our case. So here are our ideas on why the above outliers occurred:
- Data entry error: data entered correctly or the IMDb's system malfunctioned.  
- Sampling error: data that we scraped may be representatives and contain biases since IMDb websites like all other movie press machines have their own biases with movie reviews.
- Jerks: there might have been people entering the data too quickly and bots might have been used although this option is not probable but not impossible.

- The following are the only columns that are worth notetaking when analysing the outliers: they are the Votes and the Meta Scores which were generated by IMDb's sites goers, in terms of biases. All the other features such as the Year of Release, the Gross Collection and the Watch Time values may be, but not probable, considered as outliers, in terms of sampling and data entry errors...etc.



**All in all, we will check later throughout the project so any outliers are not removed. yet for the time being. It's just that for now, we will perform another data cleaning process and then move on to the other stages.**
""")




st.header("Feature Engineering and EDA")
encoded_data = pd.read_csv("encoded_clean_1.csv")

st.text("Firstly, we used one-hot label encoder to separate the genre from each movie into a column of their own having the value 1 if the movie contains the genre and 0 otherwise.")
st.write("Displaying encoded data:")
st.dataframe(encoded_data)

st.image("userRating_dist.png")
st.markdown("From the look of the histogram, it can be argued that the distribution of the userRating colum values is **$Uniform$** as expected because we use the **randbetween(start, end, step, )** which uniformly and randomly generated the userRating column values.")



st.subheader("Multicollinearity")

st.image("heatmap.png")



st.markdown("According to the heatmap displayed above, only the the Gross Collection and Votes values have correlation coefficient up to 0.63 and every other correlation coefficients in the array are below 0.35 so according to that, there is no <font color='red'>multi-collinearity</font> between every other features except between Gross_Collection($) and Votes($).",unsafe_allow_html=True)


st.subheader("Watch time and year of release")
st.image("watch_time_yor.png")
st.text("We do some more basic plottings on the histograms of watch time and year of release.")


st.subheader("**Behavior of people over time**")

st.text("""In addition, it is interesting to see if the behaviour of people changes over the years. 
Therefore, one other histogram has been made to provide more insight into the number of movies rated over time. As can be obtained in the graphs below, the number of ratings increases over time, with one big outlier somewhere in 2005. An obvious cause for this is that Netflix sampled its data randomly, so that it would protect user privacy. On the right-hand side one sees the average 
ratings over the years. It must be noted that the average rating increases over time. 
Besides, the average rating becomes more stable over time. This can be explained by 
the fact that there are fewer movies rated in the early 2000’s, which causes a higher 
standard deviation in the average rating.""")
st.image("rating_yoc.png")
st.text("It seems that the general trend is positive right from 1920. But there is a small decline in the number of rating from 2020 onwards, possibly indicating that either people just forget to rate which is unlikely or they just think that most of the movies in the last few decades are not deserving of their rating.")




st.subheader("Analysis on the genres")
st.image("genre_f.png")


st.subheader("Hypothesis Testing")

st.markdown(r'''
$H_0: \mu_1 = \mu_2 = \cdots = \mu_k$ (The mean gross revenue collection for all movie genres are all equal to each other.)

$H_1:$ At least one $\mu_i$ is different (At least one movie genre has a different mean gross revenue collection than the others.)

Where:
- $H_0$ is the null hypothesis
- $\mu_i$ is the population mean gross collection of genre $i$
- $H_1$ is the alternative hypothesis
- $k$ is the number of movie genres in the dataset
''', unsafe_allow_html=True)

st.markdown("""
We group the data by genre and calculate the mean gross revenue collection for each genre. Next, we perform a **one-way ANOVA test** using the
**<font color='red'>f_oneway()</font>** function from the scipy.stats module to determine if there is a significant difference in the mean gross revenue collection between genres.
 Finally, we print the results, which include the F-statistic and p-value. The p-value tells us the probability of observing the data if the null hypothesis (that there is no significant difference in the mean gross revenue collection between genres) is true. 
 If the p-value is below a certain significance level (usually 0.05), we reject the null hypothesis and conclude that there is a significant difference in the mean gross revenue collection between genres.
 """)


st.text("""
If the hypothesis testing and ANOVA reveal that certain movie genres are statistically significant with gross collection, it means that there is a significant association between those movie genres and the revenue generated by the movie. 
This information can be useful for movie studios, producers, and investors in making decisions about which genres to invest in and produce. For example, if action movies are found to be significantly associated with higher gross collection, 
a movie studio might decide to invest more resources into producing action movies in the future. 
On the other hand, if a particular genre is found to have a negative association with gross collection, the studio might decide to reduce investments in that genre.

""")

st.text("""
Below is the result of applying the one-way ANOVA test to between Gross collection and every other variable and then sort from lowest to highest in terms of P_values.


{'Gross_Collection(M$)': 0.0,
 'Adventure': 1.4984262722187186e-220,
 'Drama': 1.5551683552504035e-103,
 'Action': 5.629765550886604e-72,
 'Animation': 2.1392303609037253e-52,
 'Sci-Fi': 1.4404980642517328e-33,
 'Family': 1.7634851590824866e-18,
 'Fantasy': 1.7933349710672943e-17,
 'Crime': 1.4954684027617998e-15,
 'Romance': 3.5801025601425383e-13,
 'Horror': 1.7419777890748926e-11,
 'Mystery': 2.349186171586003e-06,
 'Biography': 2.3234829484631096e-05,
 'History': 0.00018819608508305945,
 'War': 0.00033925090176384456,
 'Thriller': 0.0006041762480868074,
 'Music': 0.005554289805062152,
 'Film-Noir': 0.005842038131749407}


""")

st.markdown("""
**<font color='blue'>Based on the updated dictionary, all of the genres have p-values less than 0.05, which means they are all statistically significant. 
This suggests that each genre is associated with the revenue generated by a movie. However, some genres have much smaller p-values than others, indicating a stronger association. 
The genres with the smallest p-values (i.e., the most significant associations) are Action, Adventure, Animation, .... War. 
The genres with larger p-values (i.e., less significant associations) are Adult, Comedy, Musical, Sport, and Western.</font>**
""",unsafe_allow_html=True)

st.header("Feature Creation")

st.subheader("Average genre scorer")

st.text(""""
	Genre	Avg_Score
0	Musical	3.093023
0	Biography	3.091581
0	Music	3.085714
0	Sport	3.059701
0	Crime	3.043732
0	Animation	3.039749
0	Horror	3.036913
0	Mystery	3.033010
0	Fantasy	3.029872
0	Action	3.028383
0	Comedy	3.026418
0	Romance	3.026057
0	Drama	3.020866
0	Film-Noir	3.016949
0	Sci-Fi	3.008837
0	Adult	3.000000
0	War	3.000000
0	Adventure	2.970874
0	History	2.963964
0	Thriller	2.961911
0	Family	2.946188
0	Western	2.897436
0	Age_of_movie	2.872180

Based on the average scores of the movie genres, we can draw some insights. 

Firstly, it appears that musical movies, on average, received the highest scores among all genres with a score of 3.093. Biography and music genres followed closely with scores of 3.092 and 3.086, respectively. 

On the other hand, the genres that received lower scores on average include Adventure, History, Thriller, and Family with scores of 2.971, 2.964, 2.962, and 2.946, respectively. 



""")



st.subheader("Movie Success")

st.text("We will create a new column for the movie's success, which is determined by the Gross_Collection(M$) column. We could classify a movie as successful if its gross collection is above a certain threshold, such as $100 million.")

st.image("success_movies.png")



st.subheader("User and success rate")

st.text("""
The next step is to try to see any correlation or relationship between user rating and the success of each data. Since we have a binary variable (Success) and a categorical variable (userRating), we can use a stacked bar chart to visualize the relationship between them.

We can create a crosstabulation between the two variables and plot a stacked bar chart with the percentage of success and failure for each user rating category. This will help us understand how the success rate varies across different user rating categories.

""")

st.image("success_rate.png")

st.markdown("""
Looking at the <font color='red'>Unsuccess</font> stacked bars, we see that it is mostly dominated by the user with rating 1 and just a little bit of 2. But looking at the <font color='red'>Successful</font> stacked bars, we see a more uniform relationship, at least by eyeballing, each user rating shares approximately the same area on the bar. The visualization suggests that the user rating may not have a strong correlation with the success of a movie. For unsuccessful movies, there seems to be a higher proportion of low user ratings (1 or 2), while for successful movies, the user ratings are more evenly distributed. 
""",unsafe_allow_html=True)


st.header("Feature Selection")
st.subheader("Recursive Feature Elimination (Backward Elimination)")
st.text("Selected Features:['userID', 'Watch_time', 'Meta_Score', 'Votes', 'Gross_Collection(M$)', 'Action', 'Adult', 'Biography', 'Family', 'History', 'Music', 'Thriller', 'Western', 'Age_of_movie', 'Popularity']")




st.subheader("Feature Importance Using Wrapper Method")
st.text("Selected Features:['userID', 'Year_of_Release', 'Watch_time', 'Meta_Score', 'Votes', 'Gross_Collection(M$)', 'Age_of_movie', 'Popularity']")


st.subheader("Dimensionality Reduction")
st.text("PCA is a linear dimensionality reduction technique that works only with numerical variables. It transforms the original variables into new, uncorrelated variables, called principal components. Since non-numerical variables do not have a numeric representation, they cannot be directly used for PCA.")
st.text("""
# Normalize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(encoded_data_numeric)

# Perform PCA
pca = PCA()
pca.fit(data_normalized)

# Get the explained variance ratio of each principal component
explained_variance_ratio = pca.explained_variance_ratio_

We normalized the data and performed PCA on it. The reason for standardizing is the differences in the scale or measure of each feature(comparing votes and year of release is difficult to do so without standardization).
The StandardScaler is a common preprocessing step in machine learning that is used to standardize features in a dataset. 
It transforms the data such that it has zero mean and unit variance. The main purpose of using a StandardScaler is to bring all the features onto the same scale.


explained_variance_ratio
>array([0.0843049 , 0.06961097, 0.06767835, 0.05028787, 0.04662965,
       0.04410633, 0.03978857, 0.03832161, 0.03648133, 0.03631995,
       0.03514068, 0.03456936, 0.03452958, 0.03409959, 0.03380813,
       0.03282014, 0.03134928, 0.03044688, 0.02941311, 0.02811871,
       0.02518226, 0.02373246, 0.0225811 , 0.02087869, 0.01882726,
       0.01816748, 0.01516921, 0.01136381, 0.00627274])

""")

st.image("principal_components.png")