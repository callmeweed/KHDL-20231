import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_extras.metric_cards import style_metric_cards
import numpy as np
from pymongo import MongoClient
import re


def create_data():
    mongo_uri = f"mongodb://localhost:27017"

    # Tạo kết nối tới MongoDB
    client = MongoClient(mongo_uri)

    # Chọn database
    database = client["KHDL-MOVIE-PROCCESSING"]

    # Chọn collection trong database
    collection = database["movie"]
    projection = {"imdb_id": 1, "release_date": 1, "originalTitle": 1, "summary": 1, "runtimeMinutes":1, "revenue":1, "budget":1, "_id": 0}
    # Đọc dữ liệu từ collection thành một list of dictionaries
    df = list(collection.find({}, projection))
    # Tạo DataFrame từ list of dictionaries
    data = pd.DataFrame(df)
    # data = pd.read_csv('data.csv')


    data['release_date'] = data['release_date'].apply(str)
    data['release_date'] = data['release_date'].apply(convert_to_datetime)

    # Đóng kết nối
    client.close()
    return data

def merge_data_clean(data):
    mongo_uri = f"mongodb://localhost:27017"

    # Tạo kết nối tới MongoDB
    client = MongoClient(mongo_uri)

    # Chọn database
    database = client["KHDL-MOVIE-PROCCESSING"]

    # Chọn collection trong database
    df_genres = pd.DataFrame(list(database["genre"].find()))
    df_movie_genres = pd.DataFrame(list(database["movie_genre"].find()))
    df_movie_genres = pd.merge(df_movie_genres, df_genres, how='left', left_on='genres', right_on='id')
    df_movie_genres = pd.merge(df_movie_genres, data, how='right', left_on='imdb_id', right_on='imdb_id')

    df_country = pd.DataFrame(list(database["country"].find()))
    df_movie_country = pd.DataFrame(list(database["movie_country"].find()))
    df_movie_country = pd.merge(df_movie_country, df_country, how='left', left_on='country_of_origin', right_on='id')
    df_movie_country = pd.merge(df_movie_country, data, how='right', left_on='imdb_id', right_on='imdb_id')

    df_language = pd.DataFrame(list(database["language"].find()))
    df_movie_language = pd.DataFrame(list(database["movie_language"].find()))
    df_movie_language = pd.merge(df_movie_language, df_language, how='left', left_on='languages', right_on='id')
    df_movie_language = pd.merge(df_movie_language, data, how='right', left_on='imdb_id', right_on='imdb_id')

    df_company = pd.DataFrame(list(database["company"].find()))
    df_movie_company = pd.DataFrame(list(database["movie_company"].find()))
    df_movie_company = pd.merge(df_movie_company, df_company, how='left', left_on='production_companies', right_on='id')
    df_movie_company = pd.merge(df_movie_company, data, how='right', left_on='imdb_id', right_on='imdb_id')

    df_director = pd.DataFrame(list(database["director"].find()))
    df_movie_director = pd.DataFrame(list(database["movie_director"].find()))
    df_movie_director = pd.merge(df_movie_director, df_director, how='left', left_on='Director', right_on='id')
    df_movie_director = pd.merge(df_movie_director, data, how='right', left_on='imdb_id', right_on='imdb_id')

    df_star = pd.DataFrame(list(database["star"].find()))
    df_movie_star = pd.DataFrame(list(database["movie_star"].find()))
    df_movie_star = pd.merge(df_movie_star, df_star, how='left', left_on='Stars', right_on='id')
    df_movie_star = pd.merge(df_movie_star, data, how='right', left_on='imdb_id', right_on='imdb_id')

    df_writer = pd.DataFrame(list(database["writer"].find()))
    df_movie_writer = pd.DataFrame(list(database["movie_writer"].find()))
    df_movie_writer = pd.merge(df_movie_writer, df_writer, how='left', left_on='Writers', right_on='id')
    df_movie_writer = pd.merge(df_movie_writer, data, how='right', left_on='imdb_id', right_on='imdb_id')

    # Đóng kết nối
    client.close()

    return df_movie_genres, df_movie_country, df_movie_language, df_movie_company, df_movie_director, df_movie_star, df_movie_writer

# Hàm chuyển đổi chuỗi thành datetime
def convert_to_datetime(date_string):
    # Sử dụng regex để trích xuất thông tin ngày tháng năm
    if re.search(r'\d \(\w+\)', date_string):
        date_string = "January 1, " + date_string
    match = re.search(r'(\w+ \d{1,2}, \d{4})', date_string)
    if match:
        date_part = match.group(1)
        # Chuyển đổi thành định dạng datetime
        return pd.to_datetime(date_part, format='%B %d, %Y')
    else:
        return None



# Tạo DataFrame từ list of dictionaries

st.title('Film Data Exploration')

# DATE_COLUMN = 'release_date'
# DATA_URL = ('../movie_credits.csv')

# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     lowercase = lambda x: str(x).lower()
#     data.rename(lowercase, axis='columns', inplace=True)
#     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = create_data()
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')
df_movie_genres, df_movie_country, df_movie_language, df_movie_company, df_movie_director, df_movie_star, df_movie_writer = merge_data_clean(data)

# So luong phim
def card_metric(df, df_movie_genres, df_movie_country, df_movie_language, df_movie_company, df_movie_director, df_movie_star, df_movie_writer):
    col1, col2, col3 = st.columns(3)
    movie_count = df.shape[0]
    director_count = df_movie_director['id'].unique().shape[0]
    genres_count = df_movie_genres['id'].unique().shape[0]
    country_count = df_movie_country['id'].unique().shape[0]
    language_count = df_movie_language['id'].unique().shape[0]
    company_count = df_movie_company['id'].unique().shape[0]
    star_count = df_movie_star['id'].unique().shape[0]
    writer_count = df_movie_writer['id'].unique().shape[0]
    col1.metric(label="Số lượng phim", value=movie_count)
    col1.metric(label="Số lượng đạo diễn", value=director_count)
    col2.metric(label="Số lượng thể loại", value=genres_count)
    col2.metric(label="Số lượng nước sản xuất", value=country_count)
    col3.metric(label="Số lượng ngôn ngữ", value=language_count)
    col3.metric(label="Số lượng công ty sản xuất", value=company_count)
    col1.metric(label="Số lượng ngôi sao", value=star_count)
    col3.metric(label="Số lượng biên kịch", value=writer_count)

    style_metric_cards()

st.subheader("Dữ liệu được crawl từ IMDb")
st.write(data.head(5))
card_metric(data, df_movie_genres, df_movie_country, df_movie_language, df_movie_company, df_movie_director, df_movie_star, df_movie_writer)


# Số lượng phim theo doanh thu

df_2 = data.groupby(data['revenue']).agg(
    num_of_films=('imdb_id', 'count')).reset_index()
a = df_2['num_of_films'].max()
b = df_2[df_2['revenue'] != 0.0]['num_of_films'].sum()
df_2 = pd.DataFrame({'revenue': ['0.0', 'Others'], 'num_of_films': [a, b]})
st.bar_chart(df_2, x='revenue', y='num_of_films')

ratio_budget_revenue = data[(data['budget'] != 0) & (data['revenue'] != 0)]
ratio_budget_revenue['ratio(%)'] = ratio_budget_revenue['revenue']*100.0 / ratio_budget_revenue['budget'] - 100
ratio_budget_revenue['ratio(%)'] = ratio_budget_revenue['ratio(%)']
col1, col2 = st.columns(2)
with col1:
    st.write("Top 5 phim có tỉ lệ doanh thu/ngân sách cao nhất")
    st.write(ratio_budget_revenue[['originalTitle', 'budget', 'revenue', 'ratio(%)']].sort_values(by='ratio(%)',
                                                                                                  ascending=False).head(
        5))
with col2:
    st.write("Top 5 phim có tỉ lệ doanh thu/ngân sách thấp nhất")
    st.write(ratio_budget_revenue[['originalTitle', 'budget', 'revenue', 'ratio(%)']].sort_values(by='ratio(%)',
                                                                                                  ascending=True).head(
        5))
st.write(
    "==>  Dữ liệu crawl IMDb không sạch, có nhiều phim có ngân sách và doanh thu bằng 1-2$, nên ta sẽ loại bỏ các phim này trước khi đưa vào mô hình.")

st.subheader('Bộ data sau khi loại bỏ các phim có ngân sách và doanh thu không chính xác')
#drop phim có doanh thu = 0
data = data[(data['revenue'] != 0.0) & (data['budget'] != 0.0)]
data = data[(data['revenue'] >= 10000) & (data['budget'] >= 10000)]

df_movie_genres, df_movie_country, df_movie_language, df_movie_company, df_movie_director, df_movie_star, df_movie_writer = merge_data_clean(data)

card_metric(data, df_movie_genres, df_movie_country, df_movie_language, df_movie_company, df_movie_director, df_movie_star, df_movie_writer)




## dim table


# 1. Tìm hiểu về phân phối thời gian phát hành phim
# Số lượng phim được phát hành theo từng năm hoặc theo mùa (quý, tháng) -> xu hướng phát hành phim
col1, col2, col3 = st.columns(3)

st.subheader('Số lượng phim phát hành theo thời gian')
with col1:
    option = st.selectbox(
        'Đơn vị thời gian',
        ('Month', 'Quarter', 'Year'))


if option == 'Month':
    df_1 = data.groupby(data['release_date'].dt.month).agg(
        num_of_films=('imdb_id', 'count'),
        sum_of_revenue=('revenue', 'sum')).reset_index()
elif option == 'Quarter':
    df_1 = data.groupby(data['release_date'].dt.quarter).agg(
        num_of_films=('imdb_id', 'count'),
        sum_of_revenue=('revenue', 'sum')).reset_index()
else:
    df_1 = data.groupby(data['release_date'].dt.year).agg(
        num_of_films=('imdb_id', 'count'),
        sum_of_revenue=('revenue', 'sum')).reset_index()
df_1[f"{option}"] = df_1['release_date']
st.bar_chart(df_1, x=option, y='num_of_films')
st.subheader('Tổng doanh thu theo thời gian')
st.bar_chart(df_1, x=option, y='sum_of_revenue')

st.subheader('Phân loại phim theo từng thể loại')

df = df_movie_genres.groupby('name').agg(
    num_of_films=('imdb_id', 'count')).reset_index()

fig, ax = plt.subplots()
ax.pie(df['num_of_films'], labels=df['name'], autopct=lambda p: '{:.1f}%'.format(p) if p >= 5 else '',)
# ax.barh(df['name'], df['num_of_films'])
st.pyplot(fig)

st.subheader('Doanh thu, kinh phí theo từng thể loại')
df = df_movie_genres.groupby('name').agg(
    sum_of_revenue=('revenue', 'sum'),
    sum_of_budget=('budget', 'sum')).reset_index()
# Vẽ biểu đồ bar với cột 'sum_of_revenue'
fig, ax = plt.subplots()
ax.barh(df['name'], df['sum_of_revenue'], label='Revenue', color='blue', height=0.5, align='center')

# Vẽ biểu đồ bar với cột 'sum_of_budget', dịch chúng sang phải
ax.barh(df['name'], df['sum_of_budget'], label='Budget', color='orange', height=0.5, align='edge')
ax.legend()
st.pyplot(fig)

st.subheader('Phân tích ngân sách và doanh thu')

# Create a random number generator with a fixed seed for reproducibility
n_bins = 10

# Generate two normal distributions
dist1 = data[data['budget'] != 0]['budget'].dropna()
dist2 = data[data['revenue'] != 0]['revenue'].dropna()

fig1, axs1 = plt.subplots(1, sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
axs1.hist(np.log(dist1), bins=n_bins, label='Budget')
fig2, axs2 = plt.subplots(1, sharey=True, tight_layout=True)
axs2.hist(np.log(dist2), bins=n_bins, label='Revenue')

top5_budget = data[data['budget'] != 0].sort_values(by='budget', ascending=False).head(5)
top5_revenue = data[data['revenue'] != 0].sort_values(by='revenue', ascending=False).head(5)


col1, col2 = st.columns(2)
with col1:
    st.write("Top 5 phim có ngân sách cao nhất")
    st.write(top5_budget[['originalTitle', 'budget']])
    st.write("Biểu đồ histogram ngân sách (log(budget))")
    st.pyplot(fig1)


with col2:
    st.write("Top 5 phim có doanh thu cao nhất")
    st.write(top5_revenue[['originalTitle', 'revenue']])
    st.write("Biểu đồ histogram doanh thu (log(revenue))")
    st.pyplot(fig2)


st.subheader('Phân bổ số lượng phim theo thời gian bộ phim')
df = data[data['runtimeMinutes'] != 0.0]
df1 = df['runtimeMinutes'].value_counts().reset_index()
st.bar_chart(df1, x='runtimeMinutes', y='count')

st.subheader('Phân tích sự liên kết giữa thời gian bộ phim và doanh thu')
df1 = df[df['runtimeMinutes'] <= 60]['revenue']
df2 = df[(df['runtimeMinutes'] > 60) & (df['runtimeMinutes'] <= 90)]['revenue']
df3 = df[(df['runtimeMinutes'] > 90) & (df['runtimeMinutes'] <= 120)]['revenue']
df4 = df[(df['runtimeMinutes'] > 120) & (df['runtimeMinutes'] <= 150)]['revenue']
df5 = df[df['runtimeMinutes'] > 150]['revenue']
# Vẽ box plot cho từng nhóm dữ liệu
fig, axs = plt.subplots(1, sharey=True, tight_layout=True)
axs.boxplot([np.log(df1), np.log(df2), np.log(df3), np.log(df4), np.log(df5)],
            labels=['0 - 60p', '60 - 90p', '90 - 120p', '120 - 150p', '150p+'],
            vert=True)

st.pyplot(fig)

