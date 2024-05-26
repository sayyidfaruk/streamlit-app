import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set_theme(style='dark')

def create_daily_sharing_df(df):
    daily_sharing_df = df.resample(rule="D", on='dteday').agg({
        'cnt': 'sum',
        'registered': 'sum',
        'casual': 'sum',
    })

    daily_sharing_df = daily_sharing_df.reset_index()
    daily_sharing_df.rename(columns={
        'cnt': 'total',
    }, inplace=True)

    return daily_sharing_df

def create_season_sharing_df(df):
    season_sharing_df = df.groupby(by='season').agg({
        'registered': 'sum',
        'casual': 'sum',
        'cnt': 'sum',
        'temp': ['min', 'max', 'mean', ]
    })

    season_sharing_df.rename(columns={
        'cnt':'total',
    }, inplace=True)

    season_sharing_df = season_sharing_df.rename(index={1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'})
    season_sharing_df[('temp', 'mean')] *= 41
    season_sharing_df[('temp', 'min')] *= 41
    season_sharing_df[('temp', 'max')] *= 41

    return season_sharing_df

data_df = pd.read_csv('main_data.csv')

data_df.sort_values(by='dteday',inplace=True)
data_df.reset_index(inplace=True)

data_df['dteday'] = pd.to_datetime(data_df['dteday'])

min_date = data_df['dteday'].min()
max_date = data_df['dteday'].max()

with st.sidebar:
    st.image('icon.png')

    start_date, end_date = st.date_input(
        label='Rentang Waktu', min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = data_df[(data_df['dteday']>=str(start_date))&
                  (data_df['dteday']<=str(end_date))]

daily_sharing_df = create_daily_sharing_df(main_df)
season_sharing_df = create_season_sharing_df(main_df)

st.header('Bike Sharing Dashboard :sparkles:')
st.subheader('Bike sharing')

col1, col2, col3 = st.columns(3)

with col1:
    total = daily_sharing_df.total.sum()
    st.metric('Total', value=f"{total:,}")

with col2:
    registered = daily_sharing_df.registered.sum()
    st.metric('Registered', value=f"{registered:,}")
    
with col3:
    casual = daily_sharing_df.casual.sum()
    st.metric('Casual', value=f"{casual:,}")

fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(
    daily_sharing_df['dteday'],
    daily_sharing_df['total'],
    linewidth=2,
    color='#D90404',
    label='Total'
)

ax.set_title('Number of Bike sharing', loc='center',fontsize=30)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)
ax.legend()

st.pyplot(fig)

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    daily_sharing_df['dteday'],
    daily_sharing_df['total'],
    linewidth=2,
    color='#D90404',
    label='Total'
)

ax.plot(
    daily_sharing_df['dteday'],
    daily_sharing_df['registered'],
    color="#09A603",
    label='Registered'
)

ax.plot(
    daily_sharing_df['dteday'],
    daily_sharing_df['casual'],
    color="#F29F05",
    label='Casual'
)

ax.set_title('Number of Bike sharing with registered and casual', loc='center',fontsize=30)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)
ax.legend()

st.pyplot(fig)

labels = season_sharing_df.index
registered = season_sharing_df[('registered', 'sum')]
casual = season_sharing_df[('casual', 'sum')]
total = season_sharing_df[('total', 'sum')]
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(16, 8))
rects1 = ax.bar(x - width, registered, width, label='Registered', color='#F29F05')
rects2 = ax.bar(x, casual, width, label='Casual', color='#09A603')
rects3 = ax.bar(x + width, total, width, label='Total', color='#D90404')

ax.set_ylabel(None)
ax.set_title('Seasonal Sharing Data', fontsize=30)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=25)
ax.legend()

fig.tight_layout()

st.pyplot(fig)

st.subheader('Temperature')

season = st.radio(
    label="what season do you want to see?",
    options=('spring', 'summer', 'fall', 'winter'),
    horizontal=True
)

selected_season_data = season_sharing_df.loc[season]
col1, col2, col3 = st.columns(3)

with col1:
    min = selected_season_data[('temp', 'min')]
    st.metric('Min', value=f"{min:.2f} C")

with col2:
    mean = selected_season_data[('temp', 'mean')]
    st.metric('Mean', value=f"{mean:.2f} C")
    
with col3:
    max = selected_season_data[('temp', 'max')]
    st.metric('Max', value=f"{max:.2f} C")

st.caption('Copyright © Sayyid Faruk Romdoni 2024')