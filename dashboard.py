import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import geopandas
import plotly.express as px
from datetime import datetime

st.set_page_config( layout='wide' )
@st.cache(allow_output_mutation = True)
def get_data(path):
    data = pd.read_csv(path)
    #data['date'] = pd.to_datetime(data['date']).dt.strftime( '%Y-%m-%d' )
    return data

@st.cache(allow_output_mutation= True)
def get_geofile(url):
    geofile = geopandas.read_file(url)
    return geofile

def set_feature(data):
    data['price_m2'] = data['price'] / data['sqft_lot']
    return data

def overview_data(data):
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
    # by_zipcode
    regions_zip = st.sidebar.multiselect('Choose the region',
                                         data['zipcode'].unique())

    # by_columns
    regions_feature = st.sidebar.multiselect('Choose the feature',
                                             data.columns)
    # data
    st.subheader('DataSet Overview')

    # Conditonal to filter
    if (regions_zip != []) & (regions_feature != []):
        data = data.loc[data['zipcode'].isin(regions_zip), regions_feature]
    elif (regions_zip != []) & (regions_feature == []):
        data = data.loc[data['zipcode'].isin(regions_zip), :]
    elif (regions_zip == []) & (regions_feature != []):
        data = data.loc[:, regions_feature]
    else:
        data = data.copy()

    st.write(data)

    # average analysis
    st.subheader('Average Analysis')
    # Número total de imóveis por código postal
    number_houses = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df1 = pd.DataFrame(number_houses)

    # Média dos preços por código postal
    avg_price = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df2 = pd.DataFrame(avg_price)

    # Média da sala de estar por código postal
    avg_sqft_living = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df5 = pd.DataFrame(avg_sqft_living)

    # Média do preço por m² por código postal

    price_m2_zip = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()
    df7 = pd.DataFrame(price_m2_zip)

    # Criando a tabela com as analises - Merge

    df3 = pd.merge(df1, df2, how='inner', on='zipcode')
    df4 = pd.merge(df3, df5, how='inner', on='zipcode')
    df6 = pd.merge(df4, df7, how='inner', on='zipcode')
    # df7 = pd.DataFrame(df6)

    # Renaming the columns
    df6.columns = ['Zipcode', 'Number of Houses', 'Avg Price', 'Avg Living', 'Price/m2 ']

    # Writing the Data Frame
    st.write(df6)

    # descriptive analysis
    st.subheader('Descriptive Analysis')
    # Para cada coluna que seja numérica, fazer: max, min, mediana, média, std

    num_attributes = data.select_dtypes(include=[np.number])

    media = pd.DataFrame(num_attributes.apply(np.mean))
    mediana = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))
    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))

    result = pd.concat([media, mediana, std, max_, min_], ignore_index=True, axis=1)
    result.columns = ['Media', 'Mediana', 'std', 'Max', 'Min']
    result = result.drop(labels='id',axis = 0)
    st.write(result)

    return None

def portfolio_density(data,geofile):
    # Mapas com densidade de portifólio e de preço - Por Zipcode

    # MAP 1 - Density Map
    st.title('Region Overview')
    c1, c2 = st.columns((1, 1))
    c1.header('Portifolio Density')
    # Base Map - Folium
    density_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()],
                             # width = 400,
                             # height = 400,
                             default_zoom_start=8)

    # Clusters
    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in data.iterrows():
        folium.Marker(
            location=[row['lat'], row['long']],
            popup='Sold R$: {0}, on: {1}, Features: {2} sqft, {3} bedrooms, {4} bathrooms,year_built: {5}'
                .format(row['price'],
                        row['date'],
                        row['sqft_living'],
                        row['bedrooms'],
                        row['bathrooms'],
                        row['yr_built'])).add_to(marker_cluster)
    with c1:
        folium_static(density_map)

    # MAP 2 - Price Map
    c2.header('Price Density')
    df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns = ['ZIP', 'PRICE']
    # Base Map - Folium
    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]
    price_map = folium.Map(location=[data['lat'].mean(),
                                     data['long'].mean()],
                           # width = 400,
                           # height = 400,
                           default_zoom_start=8)
    price_map.choropleth(data=df,
                         geo_data=geofile,
                         columns=['ZIP', 'PRICE'],
                         key_on='feature.properties.ZIP',
                         fill_color='YlOrRd',
                         fill_opacity=0.7,
                         line_opacity=0.2,
                         legend_name='AVG PRICE')
    with c2:
        folium_static(price_map)

    return None

def commercial_distribution(data):
    # Distribuição por categorias comerciais
    st.sidebar.title('Commercial Options')
    st.title('Commercial Attributes')
    st.header('Average Price per Year built')
    # Gráfico 1 - Average price per year
    # Filters
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())
    st.sidebar.subheader('Select Max Year Built')
    f_year_built = st.sidebar.slider('Year_built', min_year_built,
                                     max_year_built,
                                     min_year_built)
    # data selection
    df = data.loc[data['yr_built'] < f_year_built]
    df = df[['price', 'yr_built']].groupby('yr_built').mean().reset_index()
    # plot
    fig1 = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig1, use_container_width=True)

    st.header('Average Price per Day')
    # Gráfico 2 - Average price per day
    # Filters
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
    min_day_built = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_day_built = datetime.strptime(data['date'].max(), '%Y-%m-%d')
    st.sidebar.subheader('Select Max Date')
    f_day_built = st.sidebar.slider('Date', min_day_built,
                                    max_day_built,
                                    min_day_built)
    # data selection
    data['date'] = pd.to_datetime(data['date'])
    df = data.loc[data['date'] < f_day_built]
    df = df[['price', 'date']].groupby('date').mean().reset_index()
    # plot
    fig2 = px.line(df, x='date', y='price')
    st.plotly_chart(fig2, use_container_width=True)

    # HISTOGRAMAS
    st.header('Price Distribution')
    st.sidebar.subheader('Select Max Price')
    # filters
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())
    # data filtering
    f_price = st.sidebar.slider('Price', min_price, max_price, avg_price)
    df = data.loc[data['price'] < f_price]
    # data plot
    fig = px.histogram(df, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    return None

def attributes_distribution(data):
    # Distribuição por categorias físicas
    st.sidebar.title('Attributes Options')
    st.title('House Attributes')

    # filters
    f_bedrooms = st.sidebar.selectbox('Max number of Bedrooms',
                                      sorted(set(data['bedrooms'].unique())))
    f_bathrooms = st.sidebar.selectbox('Max number of bathrooms',
                                       sorted(set(data['bathrooms'].unique())))
    c1, c2 = st.columns(2)

    # House per bedrooms
    c1.header('House per bedrooms')
    df = data[data['bedrooms'] < f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # House per bathrooms
    c2.header('House per bathrooms')
    df = data[data['bathrooms'] < f_bathrooms]
    fig = px.histogram(df, x='bathrooms', nbins=19)
    c2.plotly_chart(fig, use_container_width=True)

    # filters
    f_floors = st.sidebar.selectbox('Select number of floor',
                                    sorted(set(data['floors'].unique())))

    f_waterview = st.sidebar.checkbox('Only houses with waterview')

    c1, c2 = st.columns(2)

    # Houses per floor
    c1.header('House per floors')
    df = data[data['floors'] < f_floors]
    fig = px.histogram(df, x='floors', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # Houses with waterview
    c2.header('House with waterview')
    if f_waterview:
        df = data[data['waterfront'] == 1]
    else:
        df = data.copy()

    fig = px.histogram(df, x='waterfront', nbins=10)
    c2.plotly_chart(fig, use_container_width=True)

    return None



def recomendation_compra(data):
    # Recomendação de compra
    st.subheader('Buy Recommendation')
    price_by_region = data[['zipcode', 'price']].groupby('zipcode').median()
    price_by_region = pd.DataFrame(price_by_region)
    price_by_region = price_by_region.rename(columns={'price': 'mediana'})
    recomendation = pd.DataFrame(data[['id','date','zipcode', 'condition', 'price']])
    result = pd.merge(recomendation, price_by_region, how="inner", on=["zipcode", "zipcode"])

    for i in range(len(result)):

        if (result.loc[i, 'condition'] >= 3) & (result.loc[i, 'price'] < result.loc[i, 'mediana']):
             result.loc[i, 'status'] = 'Comprar'

        else:
            result.loc[i, 'status'] = 'Não Comprar'

    result_2 = result.copy()
    result_2 = result_2.drop(labels='date',axis = 1)
    st.write(result_2)
    return result

def recomendation_venda(result):
    #Recomendação de venda
    st.subheader('Sell Recommendation')
    result['month'] = result['date'].dt.month
    new_result = result.loc[result['status'] == 'Comprar']
    df_result = pd.DataFrame(new_result)
    df_result['season'] = 'NA'

    df_result.loc[df_result['month'] == 1, 'season'] = 'winter'
    df_result.loc[df_result['month'] == 2, 'season'] = 'winter'
    df_result.loc[df_result['month'] == 12, 'season'] = 'winter'

    df_result.loc[df_result['month'] == 7, 'season'] = 'summer'
    df_result.loc[df_result['month'] == 8, 'season'] = 'summer'
    df_result.loc[df_result['month'] == 9, 'season'] = 'summer'

    df_result_summer = df_result.loc[df_result['season'] == 'summer']

    df_result_winter = df_result.loc[df_result['season'] == 'winter']

    # SUMMER
    median_summer = df_result_summer[['zipcode', 'price']].groupby('zipcode').median()
    df2_result_summer = df_result_summer[['id', 'zipcode', 'price', 'season', 'condition']]
    df_summer = pd.merge(median_summer, df2_result_summer, how='inner', on=['zipcode', 'zipcode'])
    df_summer = df_summer.rename(columns={'price_x': 'median_summer', 'price_y': 'purchase_price'})

    # WINTER
    median_winter = df_result_winter[['zipcode', 'price']].groupby('zipcode').median()
    df2_result_winter = df_result_winter[['id', 'zipcode','price', 'season', 'condition']]
    df_winter = pd.merge(median_winter, df2_result_winter, how='inner', on=['zipcode', 'zipcode'])
    df_winter = df_winter.rename(columns={'price_x': 'median_winter', 'price_y': 'purchase_price'})

    # SELL_PRICE

    # Summer
    for i in range(len(df_summer)):

        if df_summer.loc[i, 'purchase_price'] < df_summer.loc[i, 'median_summer']:
            df_summer.loc[i, 'sell_price'] = 1.3 * df_summer.loc[i, 'purchase_price']
        else:
            df_summer.loc[i, 'sell_price'] = 1.1 * df_summer.loc[i, 'purchase_price']

    # Winter
    for i in range(len(df_winter)):

        if df_winter.loc[i, 'purchase_price'] < df_winter.loc[i, 'median_winter']:
            df_winter.loc[i, 'sell_price'] = 1.3 * df_winter.loc[i, 'purchase_price']
        else:
            df_winter.loc[i, 'sell_price'] = 1.1 * df_winter.loc[i, 'purchase_price']

    # LUCRO - SUMMER & WINTER

    # Summer
    for i in range(len(df_summer)):
        df_summer.loc[i, 'Lucro'] = df_summer.loc[i, 'sell_price'] - df_summer.loc[i, 'purchase_price']

    # Winter
    for i in range(len(df_winter)):
        df_winter.loc[i, 'Lucro'] = df_summer.loc[i, 'sell_price'] - df_summer.loc[i, 'purchase_price']

    # Recomendation - SUMMER & WINTER

    st.subheader('Summer')
    st.write(df_summer)

    st.subheader('Winter')
    st.write(df_winter)

    return None



if __name__ == '__main__':
    #ETL

    #Data extration
    path = 'Datasets/kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'

    data = get_data(path)
    geofile = get_geofile(url)

    #Transformation
    # Title
    st.title('House Rocket Company')
    data = set_feature(data)
    overview_data(data)
    portfolio_density(data,geofile)
    commercial_distribution(data)
    attributes_distribution(data)
    result = recomendation_compra(data)
    recomendation_venda(result)








