"""
@WilhemHector 12.2.2024

Script to format wind data measured by MET and lidars
and ERA5 stations to facilitate building and training a neural network
for long term wind resource prediction

To do: Check that columns names are standard
"""
import pandas as pd
import numpy as np

class Dataformatter:
    def __init__ (self):
        """
        Initialize the formatter to have access to the functions
        """
        self.heights = {"MET": [59.1, 46],
           'Lidar': [46 ,40]} # Assuming that these are standards heights for MET and lidar
    
    def hourly_format(self, wind_data):
        """ 
        Convert 10mn average data to hourly data
        and add the corresponding hour of the year

        N.B You need read file while parsing the timestamp column as a datetime object
        """
        assert pd.api.types.is_datetime64_any_dtype(wind_data['Timestamp']), "Timestamp is not a datetime object"
        wind_data.set_index('Timestamp', inplace=True) # Set indexing
        hourly_data = wind_data.resample('h').mean() # Mean from hourly data
        hourly_data.reset_index(inplace=True) # Reset index

        # Calculate hour of the year
        hourly_data['Timestamp'] = pd.to_datetime(
                                                    hourly_data['Timestamp'], utc=True)
        hourly_data['hour'] = (hourly_data['Timestamp'].dt.dayofyear - 1
                                             ) * 24 + hourly_data['Timestamp'].dt.hour
        hourly_data['Timestamp'] = hourly_data['Timestamp'].dt.tz_localize(None) # Naive timezone
        return hourly_data
   
    def adjusted_wind_helper(self, row, wdir, east_col, sw_col):
        """ 
        Helper function to adjust wind data measured by Radia Mets

        Take a given row of a dataframe, its wind direction, its SW and E 
        wind speeds.
        
        Return the adjusted wind speed value according to the following;
        If wind speed from SE, use E value, if from W use SW value
        else, take the average of the two
        """
        if row[wdir] >= 30 and row[wdir] <=60: # North easterly winds
            return row[east_col]
        elif row[wdir] >= 255 and row[wdir] <= 285: # Westerly winds
            return row[sw_col]
        else:
            return (row[east_col] + row[sw_col]) / 2
        
    def adjust_wind_met(self, wind_data):
        """
        Takes a dataframe of the wind measurements from Radia's MET
        
        Adjust the mean wind speed at 59.1m and 46m and convert 
        pressure from hPa to kPa
        """
        for height in self.heights["MET"]:
            wdirec = f'Dir 57.2m S [°]' # Assuming this is standard direction sensor height
            east_col = f'Spd {height}m E [m/s]'
            sw_col = f'Spd {height}m SW [m/s]'
            new_col = f'adjusted_wind_speed_{height}'

            # Apply adjusted function to every row
            wind_data[new_col] = wind_data.apply(
                                    lambda row: self.adjusted_wind_helper(
                                                    row, wdirec, east_col, sw_col), axis=1)
        wind_data['Pres 2m [hPa]'] = wind_data['Pres 2m [hPa]'] / 10
        return wind_data
    
    def format_era5 (self, wind_data):
        """
        Format a raw csv of dataset from an ERA5 station

        Remove all "T" in the timestamps column for proper datetime
        Convert time to Mountain time
        Truncate dataset to 2000-2024
        Add a column with the hour of the year
        """
        # Convert the timestamp to Mountain time
        wind_data['Date/time [UTC]'] = pd.to_datetime(wind_data['Date/time [UTC]'], utc= True)
        wind_data['Date/time [UTC]'] = wind_data['Date/time [UTC]'].dt.tz_convert('US/Mountain')
        wind_data['hour'] = (wind_data['Date/time [UTC]'].dt.dayofyear - 1
                                            ) * 24 + wind_data['Date/time [UTC]'].dt.hour #Hour of year
        # Truncate from 2000
        formatted_data = wind_data[wind_data['Date/time [UTC]'] >= '2000-01-01 00:00:00'].copy()
        formatted_data['Date/time [UTC]'] = formatted_data['Date/time [UTC]'].dt.tz_localize(None)
        # Rename Timestamps columns for consistency
        formatted_data.rename(columns={'Date/time [UTC]': 'Timestamp'}, inplace= True) 
        return formatted_data
    
    def concurrent_data (self, wind_data, start_date, end_date):
        """
        Takes a dataframe of wind data from ERA5, a start and end date
        
        Returns a truncated dataframe for the period of interest 
        N.B start and end date needs to be in datetime format 
        i.e. 2000-01-01 00:00:00
        """
        # Filter
        mask = (wind_data['Timestamp'] >= start_date) & (wind_data['Timestamp'] <= end_date)
        truncated_data = wind_data.loc[mask]
        return truncated_data

    def wind_shear (self, lidar_data):
        """
        Takes a Lidar dataframe, and return a dataframe
        of wind shear at different timestamps
        """
        wsp_1 = lidar_data['Spd 102m [m/s]'] # Wind speed at 102m
        wsp_0 = lidar_data['Spd 46m [m/s]'] # Reference wind speed
        lidar_data['shear'] = np.log(wsp_1/wsp_0)/ np.log(102/46) # Add shear column
        return lidar_data[['Timestamp', 'shear']]
    
    def hour3_format (self, wind_data):
        """ 
        Convert to 3 hour average data
        N.B You need read file with parse_dates=['Timestamp'] as an argument
        """
        assert pd.api.types.is_datetime64_any_dtype(wind_data['Timestamp']), "Timestamp is not a datetime object"
        wind_data.set_index('Timestamp', inplace=True) # Set indexing
        hour3_data = wind_data.resample('6h').mean() # Mean from hourly data
        hour3_data.reset_index(inplace=True) # Reset index
        return hour3_data
    
    def scale_training_data (self, wind_data, scale_up, scale_mid=1):
        """ 
        Scale the wind speed training data based on how far it its from th mean
        """
        mean = np.mean(wind_data['adjusted_wind_speed_59.1'])
        std = np.std(wind_data['adjusted_wind_speed_59.1'])
        high_threshold = mean + std # Reference for high values
        mid_threshold = mean - std

        # Scale data
        scaled_data = wind_data.copy()
        high_ix = wind_data['adjusted_wind_speed_59.1'] > high_threshold
        mid_ix = (wind_data['adjusted_wind_speed_59.1'] < mid_threshold) # & (wind_data['adjusted_wind_speed_59.1'] <= high_threshold)

        scaled_data.loc[high_ix, 'adjusted_wind_speed_59.1'] *= scale_up # We are scaling everything above the std the same for now
        scaled_data.loc[mid_ix, 'adjusted_wind_speed_59.1'] *= scale_mid

        return scaled_data


        