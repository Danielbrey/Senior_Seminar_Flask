from numpy import datetime_as_string
import requests
from datetime import datetime, timedelta
import pandas as pd
import io
import re

def get_url(curr_day, curr_month, curr_year, location = 'campus'):
  return "https://observatory.middlebury.edu/campus/energy/archive/{}{}{}-{}.csv".format(curr_year, curr_month, curr_day, location)

def string_filter(df):
  datetimes_as_strings = df.iloc[:,0]
  datetimes_replace = datetimes_as_strings.str.replace(':', '-')
  datetimes_split = datetimes_replace.str.split('-')
  datetimes_minute = datetimes_split.apply(pd.Series)[3]
  minutes_filter = datetimes_minute.astype('int')%10 == 0
  
  return df[minutes_filter]
  
def load_day_data(curr_day, curr_month = "", curr_year = "", location = None):
  if(location):
    url = get_url(curr_day, curr_month, curr_year, location)
  else:
    url = get_url(curr_day, curr_month, curr_year)
    
  data = requests.get(url).content
  
  df = pd.read_csv(io.StringIO(data.decode('utf-8')),skiprows=1)
  return string_filter(df)
  

def get_metrics_data():
  curr_time = datetime.now()
  # Metric for past day
  today = load_day_data(curr_time.day, curr_time.month, curr_time.year)
  yesterday = load_day_data((curr_time + timedelta(days=-1)).day, curr_time.month, curr_time.year)
  
  todays_usage = int(today.sum()[2])
  yesterdays_usage = int(yesterday.head(len(today)).sum()[2])
  day_diff = round((((todays_usage / yesterdays_usage) - 1) * 100), 1)
  # Metric for past week
  all_data = load_day_data("all")
  num_datapoints = int(7 * 24 * 60 / 10) # 7 days, 24 hours, and one datapoint every ten minutes
  this_week_usage = int(all_data.tail(num_datapoints).sum()[2])
  last_week_usage = int(all_data.tail(num_datapoints*2).head(num_datapoints).sum()[2])
  week_diff = round((((this_week_usage / last_week_usage) - 1) * 100), 1)

  # Metric for past month


  return (todays_usage, day_diff, int(this_week_usage / 7), week_diff)



def get_todays_data(location = None):
  curr_time = datetime.now()
  # Metric for past day
  if(location):
    today = load_day_data(curr_time.day, curr_time.month, curr_time.year, location)
  else:
    print('check')
    today = load_day_data(curr_time.day, curr_time.month, curr_time.year)
  
  return today
