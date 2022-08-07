import folium
import geopandas

import numpy             as np
import pandas            as pd
import seaborn           as sns
import streamlit         as st
import plotly.express    as px
import matplotlib.pyplot as plt

from PIL              import Image
from datetime         import datetime
from folium.plugins   import MarkerCluster
from streamlit_folium import folium_static

st.set_page_config(layout='centered')


@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)
    # data['date'] = pd.to_datetime(data['date']).dt.strftime( '%Y-%m-%d' )
    return data


@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)
    return geofile


def set_feature(data):
    data['price_m2'] = data['price'] / data['sqft_lot']
    return data

def overview_data(data):
    #OVERALL INFORMATION
    title_format = '<p style="font-family:sans-serif;' \
                'font-size: 50px;' \
                'font-weight: bold;' \
                'text-align: center;' \
                '"</p> House Rocket Company</p>'
    st.markdown(title_format, unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: black;'>Welcome to House Rocket Data Report</h3>",
                unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: grey;'>General Information </h2>", unsafe_allow_html=True)
    with st.expander("Click to see overall information"):
        st.markdown("<h2 style='text-align: center; color: black;'>Profit Overview </h2>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Profit", '757M', "18%")
        c2.metric("Total Invested", '4.09B')
        c3.metric("Total returned", '4.85B')

        st.markdown("<h2 style='text-align: center; color: black;'>Overall Information </h2>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: grey;'>Total Available Properties: 21,598 </h3>",
                    unsafe_allow_html=True)
        st.markdown(
            "<h3 style='text-align: center; color: grey;'>Total Recommended Properties for Purchase: 10,579 </h3>",
            unsafe_allow_html=True)
    # - OVERVIEW DATA - START
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
    # Filters
    image = Image.open('arte.png')
    st.sidebar.image(image, caption='House Rocket Company')
    st.sidebar.header('Region and Features Options')
    # by_zipcode
    regions_zip = st.sidebar.multiselect('Choose the region',
                                         data['zipcode'].unique())

    # by_columns
    regions_feature = st.sidebar.multiselect('Choose the feature',
                                             data.columns)
    # data
    st.markdown("<h2 style='text-align: center; color: grey;'>DataSet Overview </h2>", unsafe_allow_html=True)

    # Conditonal to filter
    if (regions_zip != []) & (regions_feature != []):
        data = data.loc[data['zipcode'].isin(regions_zip), regions_feature]
    elif (regions_zip != []) & (regions_feature == []):
        data = data.loc[data['zipcode'].isin(regions_zip), :]
    elif (regions_zip == []) & (regions_feature != []):
        data = data.loc[:, regions_feature]
    else:
        data = data.copy()

    with st.expander("See DataSet"):
        st.write(data)

    # average analysis
    st.markdown("<h2 style='text-align: center; color: grey;'>Average Analysis per ZipCode </h2>",
                unsafe_allow_html=True)
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
    with st.expander("See Average Analysis"):
        st.write(df6)

    # DESCRIPTIVE ANALYSIS
    st.markdown("<h2 style='text-align: center; color: grey;'>Descriptive Analysis </h2>", unsafe_allow_html=True)

    # Price

    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Para cada coluna que seja numérica, fazer: max, min, mediana, média, std
    st.subheader('Descriptive Statistics')
    num_attributes = data[['price', 'price_m2', 'sqft_living', 'sqft_basement', 'sqft_above']]
    media = pd.DataFrame(num_attributes.apply(np.mean))
    mediana = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))
    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))

    result = pd.concat([media, mediana, std, max_, min_], ignore_index=True, axis=1)
    result.columns = ['Media', 'Mediana', 'std', 'Max', 'Min']
    result = result.T.reset_index().rename(columns={'index': "metrics"})
    result_write = result.copy().drop('price', axis=1)
    st.write(result_write)

    st.subheader('Univariate Analysis')
    # PRICE
    a1, a2, a3 = st.columns(3)
    a1.metric("", 'Price')
    a2.metric("Mean", round(result.loc[0, 'price'], 2))
    a3.metric("Median", round(result.loc[1, 'price'], 2))
    # Row B
    b1, b2, b3 = st.columns(3)
    b1.metric('Std', round(result.loc[2, 'price'], 2))
    b2.metric("Max", round(result.loc[3, 'price'], 2))
    b3.metric("Min", round(result.loc[4, 'price'], 2))

    # HISTOGRAMA - PRICE
    st.markdown("<h3 style='text-align: center; color: grey;'>Price Distribution </h3>", unsafe_allow_html=True)
    st.sidebar.header("House Attributes")
    st.sidebar.subheader('Select to see Price Distribution')
    # filters
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    # data filtering
    f_price = st.sidebar.slider('Price', min_price, max_price, max_price)
    df = data.loc[data['price'] <= f_price]
    # data plot
    fig = px.histogram(df, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    # CountPlot - view, condition, zipcode
    # Filter
    aux = data[['view', 'condition', 'zipcode']]
    st.markdown("<h4 style='text-align: center; color: grey;'>Select a Feature to see its distribution </h4>", unsafe_allow_html=True)
    count_filter = st.selectbox('', aux.columns)
    if count_filter != []:
        fig = plt.figure(figsize=(10, 4))
        sns.countplot(x=count_filter, data=aux)
        st.pyplot(fig)
    else:
        fig = plt.figure(figsize=(10, 4))
        sns.countplot(x='condition', data=aux)
        st.pyplot(fig)

    # HistPlot - yr_built
    st.markdown("<h3 style='text-align: center; color: grey;'>Histogram - Year Built </h3>", unsafe_allow_html=True)

    fig = plt.figure(figsize=(15, 10))
    sns.histplot(x='yr_built', data=data)
    st.pyplot(fig)

    return None


def attributes_distribution(data):
    # Distribuição por categorias físicas

    st.markdown("<h3 style='text-align: center; color: grey;'>House Attributes Distribution </h3>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    # House per bedrooms
    #Filter
    st.sidebar.subheader('Houses per bedroom')

    # Get nd array unique bedrooms list
    unique_bedrooms = data['bedrooms'].unique()

    # Converts nd array to dict, and then to list to pass to index of selectbox:
    unique_bedrooms_list = list(dict(enumerate(unique_bedrooms.flatten(), 0)))

    # index sorted by the last key of dictionary (grater number)
    f_bedrooms = st.sidebar.selectbox('Max Number of Bedrooms', sorted(set(data['bedrooms'].unique())),
                                      index=list(unique_bedrooms_list).index(unique_bedrooms_list[-1]))

    #Plot
    c1.subheader('House per bedrooms')
    df =  data[data['bedrooms'] <= f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # House per bathrooms
    #Filter
    st.sidebar.subheader('Houses per Bathroom')

    # Get nd array unique bathrooms list
    unique_bathrooms = data['bathrooms'].unique()

    # Converts nd array to dict, and then to list to pass to index of selectbox:
    unique_bathrooms_list = list(dict(enumerate(unique_bathrooms.flatten(), 0)))

    # index sorted by the last key of dictionary (grater number)
    f_bathrooms = st.sidebar.selectbox('Max Number of Bathrooms', sorted(set(data['bathrooms'].unique())),

                                           index=list(unique_bathrooms_list).index(unique_bathrooms_list[-1]))
    #Plot
    c2.subheader('House per bathrooms')
    df = data['bathrooms']
    fig = px.histogram(df, x='bathrooms', nbins=19)
    c2.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)

    # Houses per floor
    # Filter
    st.sidebar.subheader('Houses per Floor')

    # Get nd array unique bathrooms list
    unique_floors = data['floors'].unique()

    # Converts nd array to dict, and then to list to pass to index of selectbox:
    unique_floors_list = list(dict(enumerate(unique_floors.flatten(), 0)))

    # index sorted by the last key of dictionary (grater number)
    f_floors = st.sidebar.selectbox('Max Number of Floors', sorted(set(data['floors'].unique())),
                                    index=list(unique_floors_list).index(unique_floors_list[-1]))
    #Plot
    c1.subheader('House per floors')
    df = data['floors']
    fig = px.histogram(df, x='floors', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # Houses with waterview
    # Filter
    st.sidebar.subheader('Waterview')
    f_waterview = st.sidebar.checkbox('Only Houses with Waterview')

    if f_waterview:
        df = data[data['waterfront'] == 1]
    else:
        df = data.copy()
    #Plot
    c2.subheader('House with waterview')
    fig = px.histogram(data, x='waterfront', nbins=10)
    c2.plotly_chart(fig, use_container_width=True)

    return None

def commercial_analysis(data):
    st.header('Bivariate Analysis')
    st.markdown("<h4 style='text-align: center; color: grey;'>Filter a feature to see its Boxplot </h4>", unsafe_allow_html=True)
    # BOXPLOTS using Plotly
    # Filter
    aux = data[['bedrooms', 'floors', 'view', 'condition', 'waterfront', 'price']]
    box_filter = st.selectbox('', aux.columns)

    if box_filter != []:
        fig = plt.figure(figsize=(10, 4))
        sns.boxplot(x=box_filter, y='price', data=aux)
        st.pyplot(fig)
    else:
        fig = plt.figure(figsize=(10, 4))
        sns.boxplot(x='bedrooms', y='price', data=aux)
        st.pyplot(fig)

    # Average Price per Year built
    st.markdown("<h3 style='text-align: center; color: grey;'>Average Price per Year built </h3>", unsafe_allow_html=True)
    # Gráfico 1 - Average price per year
    df = data.loc[data['yr_built']]  # < f_year_built]
    df = df[['price', 'yr_built']].groupby('yr_built').mean().reset_index()
    # plot
    fig1 = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig1, use_container_width=True)

    return None


def portfolio_density(data, geofile):
    # Mapas com densidade de portifólio e de preço - Por Zipcode
    # MAP 1 - Density Map
    st.markdown("<h2 style='text-align: center; color: grey;'>Region Overview </h2>", unsafe_allow_html=True)
    c1, c2 = st.columns((1, 1))
    c3, c4 = st.columns((1, 1))
    c1.subheader('Portifolio Density')
    map1 = c3.checkbox('See Portifolio Density Map', )
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
    if map1:
        with c1:
            folium_static(density_map)

    # MAP 2 - Price Map
    c2.subheader('Price Density')
    map2 = c4.checkbox('See Price Density Map')
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
    if map2:
        with c2:
            folium_static(price_map)

    return None


def recomendation_compra(data):
    # Recomendação de compra
    st.markdown("<h2 style='text-align: center; color: grey;'> Business Recommendations </h2>", unsafe_allow_html=True)
    st.subheader('Buy Recommendation')
    price_by_region = data[['zipcode', 'price']].groupby('zipcode').median()
    price_by_region = pd.DataFrame(price_by_region)
    price_by_region = price_by_region.rename(columns={'price': 'mediana'})
    recomendation = pd.DataFrame(data[['id', 'date', 'zipcode', 'condition', 'price']])
    result = pd.merge(recomendation, price_by_region, how="inner", on=["zipcode", "zipcode"])

    for i in range(len(result)):

        if (result.loc[i, 'condition'] >= 3) & (result.loc[i, 'price'] < result.loc[i, 'mediana']):
            result.loc[i, 'status'] = 'Comprar'

        else:
            result.loc[i, 'status'] = 'Não Comprar'

    result_2 = result.copy()
    result_2 = result_2.drop(labels='date', axis=1)
    only_buy = st.checkbox('Only Houses to Buy')
    if only_buy:
        aux = result_2.loc[result_2['status'] == 'Comprar']
        st.write(aux)
    else:
        st.write(result_2)
    return result


def recomendation_venda(result):
    # Recomendação de venda
    st.subheader('Sell Recommendation')
    result['date'] = pd.to_datetime(result['date'])
    result['month'] = result['date'].dt.month
    new_result = result.loc[result['status'] == 'Comprar']
    df_result = pd.DataFrame(new_result)
    df_result['season'] = 'NA'

    df_result.loc[df_result['month'] == 9, 'season'] = 'fall'
    df_result.loc[df_result['month'] == 10, 'season'] = 'fall'
    df_result.loc[df_result['month'] == 11, 'season'] = 'fall'
    df_result.loc[df_result['month'] == 12, 'season'] = 'winter'
    df_result.loc[df_result['month'] == 1, 'season'] = 'winter'
    df_result.loc[df_result['month'] == 2, 'season'] = 'winter'

    df_result.loc[df_result['month'] == 6, 'season'] = 'summer'
    df_result.loc[df_result['month'] == 7, 'season'] = 'summer'
    df_result.loc[df_result['month'] == 8, 'season'] = 'summer'
    df_result.loc[df_result['month'] == 3, 'season'] = 'spring'
    df_result.loc[df_result['month'] == 4, 'season'] = 'spring'
    df_result.loc[df_result['month'] == 5, 'season'] = 'spring'

    df_result_summer = df_result.loc[(df_result['season'] == 'summer') | (df_result['season'] == 'spring')]
    df_result_winter = df_result.loc[(df_result['season'] == 'winter') | (df_result['season'] == 'fall')]


    # SUMMER
    median_summer = df_result_summer[['zipcode', 'price']].groupby('zipcode').median()
    df2_result_summer = df_result_summer[['id', 'zipcode', 'price', 'season', 'condition']]
    df_summer = pd.merge(median_summer, df2_result_summer, how='inner', on=['zipcode', 'zipcode'])
    df_summer = df_summer.rename(columns={'price_x': 'median_summer', 'price_y': 'purchase_price'})

    # WINTER
    median_winter = df_result_winter[['zipcode', 'price']].groupby('zipcode').median()
    df2_result_winter = df_result_winter[['id', 'zipcode', 'price', 'season', 'condition']]
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
        df_winter.loc[i, 'Lucro'] = df_winter.loc[i, 'sell_price'] - df_winter.loc[i, 'purchase_price']

    #Recomendation - SUMMER & WINTER
    st.subheader('Summer & Spring')
    st.write(df_summer)
    lucro_total = '$' + str(round(df_summer['Lucro'].sum()))
    st.metric('Total Profit', lucro_total)

    st.subheader('Winter & Fall')
    st.write(df_winter)
    lucro_total = '$' + str(round(df_winter['Lucro'].sum()))
    st.metric('Total Profit', lucro_total)

    #Meus dados
    st.write("\n\n\n\n\n\n"
             "by **Brunna Neri**"
             " \n\n"
             "More details: "
             "[GitHub](https://github.com/brunnaneri)"
             " \n\n"
             "Contact: [LinkedIn](https://www.linkedin.com/in/brunna-neri-74928516a)")

    return None


if __name__ == '__main__':
    # ETL

    # Data extration
    path = 'Datasets/kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
    data = get_data(path)
    geofile = get_geofile(url)

    # Transformation
    data = set_feature(data)
    overview_data(data)
    attributes_distribution(data)
    commercial_analysis(data)
    portfolio_density(data, geofile)
    result = recomendation_compra(data)
    recomendation_venda(result)
