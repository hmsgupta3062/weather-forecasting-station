import streamlit
import numpy
import pandas
import pandas.tseries.offsets
import seaborn
import matplotlib
import matplotlib.pyplot
import statsmodels
import statsmodels.tsa.stattools
import statsmodels.graphics.tsaplots
import statsmodels.tsa.arima.model
import os
import time
import warnings
warnings.filterwarnings("ignore")

streamlit.set_page_config(
    page_title="Weather Forecasting Station",
    layout="wide",
    initial_sidebar_state="expanded"
)


abs_path = ''
count = 0
data_name = ['temperature', 'humidity', 'pressure']
data_value = {"temperature_model_order": [2, 1, 1], "humidity_model_order": [1, 1, 2], "pressure_model_order": [5, 1, 4]}
data_unit = {'temperature': 'DegC', 'humidity': '%', 'pressure': 'Pa'}
container = streamlit.empty()


def about():
    project_description = 'This project is a <strong>Weather Forecasting Application</strong> which uses an IoT prototype to collect the information of the temperature, humidity and pressure from the surroundings and then sends it to the cloud in the real-time. The project uses the most suitable ML models which are able to forecast the weather data with a very low computation time and minimum possible error rate.<br><br>The IoT prototype collects the data every minute and send it to the Thingspeak Cloud Platform in the real-time. Later, the web application fetches the data from the Thingspeak Cloud Platform and then updates the dataset and after sometime, it re-trains the model and re-plots the graphs and re-forecasts the values of the temperature, humidity and pressure for the upcoming timestamps having the frequency of per minute'
    project_hardware_components = '<ul>' \
                         '<li>NodeMCU</li>' \
                         '<li>DHT11</li>' \
                         '<li>BMP280</li>' \
                         '<li>Wires and power adapters</li>' \
                         '</ul>'
    project_software_components = '<ul>' \
                                  '<li>Arduino IDE</li>' \
                                  '<li>Fritzig</li>' \
                                  '<li>Thingspeak cloud platform</li>' \
                                  '<li>Heroku cloud platform</li>' \
                                  '</ul>'
    project_workflow = 'Study of different weather forecasting systems currently developed and preparing the list of components required to build the project -> Developing an IoT system to measure the temperature, humidity and pressure values of the surroundings -> Collecting the real-time data for the duration of a month -> Visualising data and performing EDA on the data collected -> Building a forecasting model -> Developing a web application and deploy it on the cloud.'
    project_github_link = 'https://github.com/hmsgupta3062/weather-forecasting-station'
    developer_description = 'I am a final year ECE student of the Electronics and Communication Engineering Department of the ' \
                      'Manav Rachna International Institute of Research and Studies. My areas of interest ' \
                      'include the Internet of Things, Computer Vision, Deep Learning, and Web Development. I am a ' \
                      'member of Manav Rachna Innovation and Incubation Center too. I had published 2 review papers, ' \
                      '1 research paper, and 2 book chapters, 1 book and is currently authoring some other booksas well.'
    developer_contact = 'Mail: <a href="mailto:hmsgupta.3062@gmail.com">hmsgupta.3062@gmail.com</a>' \
                        '<br>' \
                        'Phone: <a href="tel:+918586803677">+91 8586803677</a>'

    streamlit.header('Knowing the Project and developer')
    streamlit.write('')
    streamlit.write('')
    streamlit.subheader('Project Description')
    streamlit.write('')
    streamlit.markdown(project_description, True)
    streamlit.write('')
    streamlit.subheader('Project Hardware Component List')
    streamlit.write('')
    streamlit.markdown(project_hardware_components, True)
    streamlit.write('')
    streamlit.subheader('Project Software Component List')
    streamlit.write('')
    streamlit.markdown(project_software_components, True)
    streamlit.write('')
    streamlit.subheader('Project Workflow')
    streamlit.write('')
    streamlit.write(project_workflow)
    streamlit.write('')
    streamlit.subheader('Project GitHub Link')
    streamlit.write('')
    streamlit.write(project_github_link)
    streamlit.write('')
    streamlit.subheader('Developer Description')
    streamlit.write('')
    streamlit.write(developer_description)
    streamlit.write('')
    streamlit.subheader('Developer Contact Details')
    streamlit.write('')
    streamlit.markdown(developer_contact, True)
    streamlit.write('')
    streamlit.write('')

def updating_data_source():
    global data_source

    timestamp = pandas.DatetimeIndex(pandas.DatetimeIndex(
        pandas.DatetimeIndex(data_source.tail(1).created_at.astype(numpy.datetime64)).strftime('%Y-%m-%d %H:%M:%S'),
        tz='GMT').tz_convert('Asia/Kolkata').strftime('%Y-%m-%d %H:%M:%S')) + pandas.tseries.offsets.Minute()
    data_2 = pandas.read_csv(
        'https://api.thingspeak.com/channels/1665242/feeds.csv?timezone=Asia%2FKolkata&status=true&location=true&start={}-{}-{}%20{}:{}:00'.format(
            timestamp.year.values[0], timestamp.month.values[0], timestamp.day.values[0], timestamp.hour.values[0], timestamp.minute.values[0]))
    data = pandas.concat((data_source, data_2), ignore_index=True)
    data.to_csv(os.path.join(abs_path, 'feeds.csv'), index=False)

def process_data(data):
    data = data.copy()
    data.loc[:10584, 'field3'] = data['field1'][:10585].values
    data.loc[:10584, 'field4'] = data['field2'][:10585].values
    data['created_at'] = pandas.DatetimeIndex(pandas.DatetimeIndex(
        pandas.DatetimeIndex(data['created_at'].astype(numpy.datetime64)).strftime('%Y-%m-%d %H:%M:%S'),
        tz='GMT').tz_convert('Asia/Kolkata').strftime('%Y-%m-%d %H:%M:%S'))
    data.drop(['field1', 'field2', 'latitude', 'longitude', 'elevation', 'status', 'entry_id'], axis=1, inplace=True)
    columns = {'field3': 'temperature', 'field4': 'humidity', 'field5': 'pressure', 'created_at': 'timestamp'}
    data.rename(columns=columns, inplace=True)
    data['temperature'].replace(0.0, numpy.nan, inplace=True)
    data['temperature'].fillna(method='ffill', inplace=True)
    data['humidity'].replace(0.0, numpy.nan, inplace=True)
    data['humidity'].fillna(method='bfill', inplace=True)
    data['pressure'].replace('\r\n\r\n', '', regex=True, inplace=True)
    data['pressure'] = data['pressure'].astype(dtype=numpy.float64)
    data['pressure'].replace([0.0, -9387.59, 878.93, 101772.86], numpy.nan, inplace=True)
    data['pressure'].fillna(value=numpy.min(data['pressure']), inplace=True)
    return data

def draw_line_plot(data):
    df = data.copy()
    df = df.set_index('timestamp')
    streamlit.line_chart(df)

def draw_correlation_plot():
    global processed_data_source

    fig, ax = matplotlib.pyplot.subplots()
    seaborn.heatmap(processed_data_source.corr(), ax=ax, annot=True, fmt=".3f")
    return fig

def adf_test(series):
    result = statsmodels.tsa.stattools.adfuller(series.dropna())
    labels = ['ADF test statistic', 'p-value', '# lags used', '# observations']
    out = pandas.Series(result[0:4], index=labels)
    for key, val in result[4].items():
        out[f'critical value ({key})'] = val
    data = pandas.DataFrame(out).rename(columns={0: "values"})
    if result[1] <= 0.05:
        return (True, data, 'There is a STRONG evidence against the null hypothesis. Therefore, REJECT the null hypothesis. Data is STATIONARY.')
    else:
        return (False, data, 'There is a WEAK evidence against the null hypothesis. Therefore, FAILED TO REJECT the null hypothesis. Data is NON-STATIONARY.')

def draw_acf_plot(data):
    fig, ax = matplotlib.pyplot.subplots()
    statsmodels.graphics.tsaplots.plot_acf(data, ax=ax, zero=False, lags=40)
    return fig

def draw_pacf_plot(data):
    fig, ax = matplotlib.pyplot.subplots()
    statsmodels.graphics.tsaplots.plot_pacf(data, ax=ax, lags=40, alpha=0.05, zero=False, method=('ols'))
    return fig

def train_model_func(data, order):
    model = statsmodels.tsa.arima.model.ARIMA(data, order=order)
    results = model.fit()
    return results

def forecast_func(model):
    global processed_data_source

    len = processed_data_source.__len__() - 1
    predictions_future = model.predict(start=len+1, end=len+10, dynamic=False)
    date = processed_data_source.tail(1).timestamp.copy()
    indexes = [(date + (pandas.tseries.offsets.Minute() * (i+1))).tolist()[0] for i in range(10)]
    data = pandas.DataFrame({'data': predictions_future, 'timestamp': indexes})
    return data

while 1:
    start_time = time.time()
    with container.container():
        streamlit.title('Weather Forecasting Station')
        streamlit.write('')
        streamlit.write('')

        # display the project and developer's details
        about()

        # update the data source and process it
        data_source = pandas.read_csv(os.path.join(abs_path, 'feeds.csv'))
        updating_data_source()
        processed_data_source = process_data(data_source)

        # display the data source
        streamlit.header('Exploring the values in the real-time')
        streamlit.write('')
        display_columns = streamlit.columns(3)
        display_columns[0].info('Temperature: {} Deg C'.format(processed_data_source.tail(1)['temperature'].values[0]))
        display_columns[1].info('Humidity: {} %'.format(processed_data_source.tail(1)['humidity'].values[0]))
        display_columns[2].info('Pressure: {} Pa'.format(processed_data_source.tail(1)['pressure'].values[0]))
        streamlit.write('')

        # displaying the dataset
        streamlit.subheader('Exploring Raw Dataset')
        streamlit.write('')
        streamlit.write(data_source.iloc[-1:-11:-1, :].astype(str))
        streamlit.write('')
        streamlit.subheader('Exploring Processed Dataset')
        streamlit.write('')
        streamlit.write(processed_data_source.iloc[-1:-11:-1, :])
        streamlit.write('')
        streamlit.subheader('Visualising Data')
        for i in data_name:
            streamlit.write('')
            streamlit.info('{} (in {}) vs Timestamp'.format(i, data_unit[i]))
            streamlit.write('')
            draw_line_plot(processed_data_source.loc[:, ['timestamp', i]])
        streamlit.write('')

        if count % 30 == 0:
            # draw the correlation plot
            streamlit.subheader('Data Correlation Plot')
            streamlit.write('')
            streamlit.write(draw_correlation_plot())
            streamlit.write('')
            streamlit.write('')

            # forecast the values
            for i in data_name:
                start_time = time.time()
                streamlit.header('Forecasting the {} values'.format(i))
                streamlit.write('')

                # testing the stationarity
                streamlit.subheader('Test for stationarity of the {} data'.format(i))
                streamlit.write('')
                streamlit.info('Test ran is the Augmented Dickey-Fuller Test.')
                streamlit.write('')
                flag, result, message = adf_test(processed_data_source[i])
                if flag:
                    streamlit.success(message)
                else:
                    streamlit.error(message)
                streamlit.write(result)
                streamlit.write('')

                # plotting the ACF plot
                streamlit.subheader('Auto-Correlation Function Plot (ACF) of the {} data'.format(i))
                streamlit.write('')
                streamlit.write(draw_acf_plot(processed_data_source[i]))
                streamlit.write('')

                # plotting the PACF plot
                streamlit.subheader('Partial Auto-Correlation Function Plot (PACF) of the {} data'.format(i))
                streamlit.write('')
                streamlit.write(draw_acf_plot(processed_data_source[i]))
                streamlit.write('')

                # training the ML model
                streamlit.subheader('ML Model to Forecast {} Data Values'.format(i.title()))
                streamlit.write('')
                streamlit.info('Model used is ARIMA model.')
                streamlit.write('')
                model = train_model_func(processed_data_source[i], data_value['{}_model_order'.format(i)])
                streamlit.write(model.summary())
                streamlit.write('')

                # forecasting the values
                streamlit.subheader('Forecasting ' + i.title() + ' Values')
                streamlit.write('')
                forecast_data = forecast_func(model)
                streamlit.write(forecast_data.astype(str))
                streamlit.write('')

                # visualise the forecasted values
                streamlit.subheader('Visualise the Forecasted {} Values'.format(i.title()))
                streamlit.write('')
                streamlit.info('{} (in {}) vs Future Timestamp values'.format(i.title(), data_unit[i]))
                streamlit.write('')
                draw_line_plot(forecast_data)
                streamlit.write('')
                streamlit.write('')

                count = 0

    time.sleep(60 - time.time() + start_time)
    count += 1