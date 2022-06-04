import datetime
import streamlit
import streamlit_option_menu
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
import json
import time
import warnings
warnings.filterwarnings("ignore")

streamlit.set_page_config(
    page_title="Weather Forecasting Station",
    layout="wide",
    initial_sidebar_state="expanded"
)

abs_path = ''
with open(os.path.join(abs_path, 'config_data.json'), 'r') as file:
    config_data = json.load(file)
config_data['updated_data_at'] = datetime.datetime.strftime(datetime.datetime.now() - (pandas.tseries.offsets.Day() * 2), '%Y-%m-%d %H:%M:%S.%f')
config_data['updated_correlation_plot_at'] = config_data['updated_data_at']
config_data['updated_temperature_model_at'] = config_data['updated_data_at']
config_data['updated_humidity_model_at'] = config_data['updated_data_at']
config_data['updated_pressure_model_at'] = config_data['updated_data_at']
with open(os.path.join(abs_path, 'config_data.json'), 'w') as file:
    json.dump(config_data, file)
# start_time = time.time()
data_first = True
correlation_first = True
temperature_model = True
humidity_model = True
pressure_model = True

def draw_line_plot(data):
    streamlit.line_chart(data)

@streamlit.cache(show_spinner=False, suppress_st_warning=True, allow_output_mutation=True)
def draw_correlation_plot():
    fig, ax = matplotlib.pyplot.subplots()
    seaborn.heatmap(processed_data_source.corr(), ax=ax, annot=True, fmt=".3f")
    return fig

@streamlit.cache(show_spinner=False, suppress_st_warning=True, allow_output_mutation=True)
def draw_acf_plot(data):
    fig, ax = matplotlib.pyplot.subplots()
    statsmodels.graphics.tsaplots.plot_acf(data, ax=ax, zero=False, lags=40)
    return fig

@streamlit.cache(show_spinner=False, suppress_st_warning=True, allow_output_mutation=True)
def draw_pacf_plot(data):
    fig, ax = matplotlib.pyplot.subplots()
    statsmodels.graphics.tsaplots.plot_pacf(data, ax=ax, lags=40, alpha=0.05, zero=False, method=('ols'))
    return fig

def updating_data_source():
    timestamp = pandas.DatetimeIndex(pandas.DatetimeIndex(
        pandas.DatetimeIndex(data_source.tail(1).created_at.astype(numpy.datetime64)).strftime('%Y-%m-%d %H:%M:%S'),
        tz='GMT').tz_convert('Asia/Kolkata').strftime('%Y-%m-%d %H:%M:%S')) + pandas.tseries.offsets.Minute()
    data_2 = pandas.read_csv(
        'https://api.thingspeak.com/channels/1665242/feeds.csv?timezone=Asia%2FKolkata&status=true&location=true&start={}-{}-{}%20{}:{}:00'.format(
            timestamp.year.values[0], timestamp.month.values[0], timestamp.day.values[0], timestamp.hour.values[0], timestamp.minute.values[0]))
    data = pandas.concat((data_source, data_2), ignore_index=True)
    data.to_csv(os.path.join(abs_path, 'feeds.csv'), index=False)

def fetch_data():
    if (datetime.datetime.now() - datetime.datetime.strptime(config_data['updated_at'], '%Y-%m-%d %H:%M:%S.%f')).total_seconds() >= 900:
        updating_data_source()
        config_data['updated_at'] = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S.%f')
        config_data['data_update_flag'] = True
        with open(os.path.join(abs_path, 'config_data.json'), 'w') as file:
            json.dump(config_data, file)
    else:
        config_data['data_update_flag'] = False
        with open(os.path.join(abs_path, 'config_data.json'), 'w') as file:
            json.dump(config_data, file)

@streamlit.cache(show_spinner=False, suppress_st_warning=True, allow_output_mutation=True)
def process_data(data_temp):
    data = data_temp.copy()
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

@streamlit.cache(show_spinner=False, suppress_st_warning=True, allow_output_mutation=True)
def process_plot_data(data):
    df = data.copy()
    df = df.set_index('timestamp')
    return df

# @streamlit.cache(show_spinner=False, suppress_st_warning=True, allow_output_mutation=True)
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

@streamlit.cache(show_spinner=False, suppress_st_warning=True, allow_output_mutation=True)
def train_model_func(data, order):
    model = statsmodels.tsa.arima.model.ARIMA(data, order=order)
    results = model.fit()
    return results

def forecast_func(model):
    len = processed_data_source.__len__() - 1
    predictions_future = model.predict(start=len+1, end=len+10, dynamic=False)
    date = processed_data_source.tail(1).timestamp.copy()
    indexes = [(date + (pandas.tseries.offsets.Minute() * (i+1))).tolist()[0] for i in range(10)]
    data = pandas.DataFrame({'data': predictions_future, 'timestamp': indexes})
    return data

def about():
    # https://github.com/hmsgupta3062/weather-forecasting-station
    about_project = 'This project is a <strong>Weather Forecasting Application</strong> which uses an IoT prototype to ' \
                    'collect the information of the temperature, humidity and pressure from the surroundings and then ' \
                    'sends it to the cloud in the real-time. The data is collected every minute. The project uses the' \
                    'most suitable ML models which are able to forecast the weather data with a very low computation time' \
                    'and minimum possible error rate.'
    about_developer = 'I am a final year ECE student of the Electronics and Communication Engineering Department of the ' \
                      'Manav Rachna International Institute of Research and Studies. My areas of interest ' \
                      'include the Internet of Things, Computer Vision, Deep Learning, and Web Development. I am a ' \
                      'member of Manav Rachna Innovation and Incubation Center too. I had published 2 review papers, ' \
                      '1 research paper, and 2 book chapters and is currently authoring a book on 8051 microcontroller.'
    streamlit.header('About the Project')
    streamlit.subheader('Project Description')
    streamlit.markdown(about_project, True)
    streamlit.subheader('Project Developer')
    streamlit.write(about_developer)

def all_data_func():
    global processed_data_source, plot_data_source, correlation_first

    streamlit.header('View All the Data Values in the real-time')
    streamlit.write('')
    streamlit.write('')
    # streamlit.subheader('Refresh Data Source')
    # streamlit.write('')
    # streamlit.button('Click Here to Refresh Data', on_click=fetch_data)
    # if config_data['data_update_flag']:
    #     streamlit.success('Data Refreshed Successfully')
    # else:
    #     streamlit.error('Data updation failed')
    # streamlit.write('')
    # streamlit.write('')
    streamlit.subheader('Exploring Raw Dataset')
    streamlit.write('')
    streamlit.write(data_source.iloc[-1:-11:-1, :].astype(str))
    streamlit.write('')
    streamlit.write('')
    streamlit.subheader('Exploring Processed Dataset')
    streamlit.write('')
    streamlit.write(processed_data_source.iloc[-1:-11:-1, :])
    streamlit.write('')
    streamlit.write('')
    streamlit.subheader('Visualising Data')
    streamlit.write('')
    streamlit.write('')
    draw_line_plot(plot_data_source['temperature'])
    streamlit.write('')
    streamlit.info('Humidity (in %) vs Timestamp')
    streamlit.write('')
    draw_line_plot(plot_data_source['humidity'])
    streamlit.write('')
    streamlit.info('Pressure (in Pa) vs Timestamp')
    streamlit.write('')
    draw_line_plot(plot_data_source['pressure'])
    streamlit.write('')
    streamlit.write('')
    if (datetime.datetime.now() - datetime.datetime.strptime(config_data['updated_correlation_plot_at'],
                                                             '%Y-%m-%d %H:%M:%S.%f')).total_seconds() >= 150 or correlation_first:
        streamlit.subheader('Correlation between Data')
        streamlit.write('')
        streamlit.write(draw_correlation_plot())
        config_data['updated_correlation_plot_at'] = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S.%f')
        with open(os.path.join(abs_path, 'config_data.json'), 'w') as file:
            json.dump(config_data, file)
        correlation_first = False

def separate_data_func(data_name, data_unit, raw_fields):
    global processed_data_source, plot_data_source, eval(data_name + '_model')
    # data_name = 'pressure'
    temp = eval(data_name + '_model')

    # if (datetime.datetime.now() - datetime.datetime.strptime(config_data['updated_data_at'], '%Y-%m-%d %H:%M:%S.%f')).total_seconds() >= 60:
    streamlit.header('Analyse the ' + data_name.title() + ' Data Model and Dataset Values')
    streamlit.write('')
    streamlit.write('')

    # streamlit.subheader('Refresh Data Source')
    # streamlit.write('')
    # streamlit.button('Click Here to Refresh Data', on_click=fetch_data)
    # if config_data['data_update_flag']:
    #     streamlit.success('Data Refreshed Successfully')
    # else:
    #     streamlit.error('Data Updation failed')
    # streamlit.write('')
    # streamlit.write('')

    streamlit.subheader('Exploring Raw ' + data_name.title() + ' Data')
    streamlit.write('')
    streamlit.write(data_source.iloc[-1:-11:-1, [0, raw_fields]].astype(str))
    streamlit.write('')
    streamlit.write('')
    streamlit.subheader('Exploring Processed ' + data_name.title() + ' Data')
    streamlit.write('')
    streamlit.write(processed_data_source.loc[:, ['timestamp', data_name]].iloc[-1:-11:-1, :])
    streamlit.write('')
    streamlit.write('')
    streamlit.subheader('Visualising Data')
    streamlit.write('')
    streamlit.info('{} (in {}) vs Timestamp'.format(data_name.title(), data_unit))
    streamlit.write('')
    draw_line_plot(plot_data_source[data_name])
    streamlit.write('')
    streamlit.write('')

    if (datetime.datetime.now() - datetime.datetime.strptime(config_data['updated_{}_model_at'.format(data_name)], '%Y-%m-%d %H:%M:%S.%f')).total_seconds() >= 150 or temp:
        streamlit.subheader('Test for stationarity')
        streamlit.write('')
        streamlit.info('Test ran is the Augmented Dickey-Fuller Test.')
        streamlit.write('')
        flag, result, message = adf_test(processed_data_source[data_name])
        if flag:
            streamlit.success(message)
        else:
            streamlit.error(message)
        streamlit.write(result)
        streamlit.write('')
        streamlit.write('')
        streamlit.subheader('Auto-Correlation Function Plot (ACF)')
        streamlit.write('')
        streamlit.write(draw_acf_plot(processed_data_source[data_name]))
        streamlit.write('')
        streamlit.write('')
        streamlit.subheader('Partial Auto-Correlation Function Plot (PACF)')
        streamlit.write('')
        streamlit.write(draw_pacf_plot(processed_data_source[data_name]))
        streamlit.write('')
        streamlit.write('')

        # if (datetime.datetime.now() - datetime.datetime.strptime(config_data['updated_{}_model_at'.format(data_name)], '%Y-%m-%d %H:%M:%S.%f')).total_seconds() >= 600:
        model = train_model_func(processed_data_source[data_name], config_data[data_name + '_model_order'])
        streamlit.subheader('ML Model to Forecast ' + data_name.title() + ' Data Values')
        streamlit.write('')
        streamlit.info('Model used is ARIMA model.')
        streamlit.write('')
        streamlit.write(model.summary())
        streamlit.write('')
        streamlit.write('')
        streamlit.subheader('View the Forecasted ' + data_name.title() + ' Values')
        streamlit.write('')
        forecast_data = forecast_func(model)
        streamlit.write(forecast_data.astype(str))
        streamlit.write('')
        streamlit.write('')
        streamlit.subheader('Visualise the Forecasted ' + data_name.title() + ' Values')
        streamlit.write('')
        plot_forecast_data = process_plot_data(forecast_data)
        streamlit.info('{} (in {}) vs Future Timestamp values'.format(data_name.title(), data_unit))
        streamlit.write('')
        draw_line_plot(plot_forecast_data)

        config_data['updated_{}_model_at'.format(data_name)] = datetime.datetime.strftime(datetime.datetime.now(),
                                                                                '%Y-%m-%d %H:%M:%S.%f')
        with open(os.path.join(abs_path, 'config_data.json'), 'w') as file:
            json.dump(config_data, file)
        temp = False

selected = streamlit_option_menu.option_menu("", ["About", 'All', 'Temperature', "Humidity", 'Pressure'], orientation='horizontal', default_index=0)
container = streamlit.empty()

while 1:
    if (datetime.datetime.now() - datetime.datetime.strptime(config_data['updated_data_at'], '%Y-%m-%d %H:%M:%S.%f')).total_seconds() >= 60 or data_first:
    # if start_time - time.time() >= 60 or first:
        data_first = False
        # start_time = time.time()
        abs_path = ''
        data_source = pandas.read_csv(os.path.join(abs_path, 'feeds.csv'))
        processed_data_source = process_data(data_source)
        plot_data_source = process_plot_data(processed_data_source)
        with open(os.path.join(abs_path, 'config_data.json'), 'r') as file:
            config_data = json.load(file)
        updating_data_source()
        config_data['updated_data_at'] = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S.%f')
        with open(os.path.join(abs_path, 'config_data.json'), 'w') as file:
            json.dump(config_data, file)

    with container.container():
        if selected == 'About':
            about()
        elif selected == 'All':
            all_data_func()
        elif selected == 'Temperature':
            # temperature_data_func()
            separate_data_func('temperature', 'DegC', 3)
        elif selected == 'Humidity':
            # humidity_data_func()
            separate_data_func('humidity', '%', 4)
        elif selected == 'Pressure':
            # pressure_data_func()
            separate_data_func('pressure', 'Pa', 5)
        else:
            pass
