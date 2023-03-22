__all__ = ['datetime_extraction']


def datetime_extraction(feature, extract, mode=0, name=None, config=None):
    """Extracts date and time features from datetime variables.
    
    The transformer supports the extraction of the following features:
    - 'year'
    - 'month'
    - 'day'
    - 'hour'
    - 'minute'
    - 'second'
    - 'date'
    - 'time'
    - 'weekday'
    - 'yearday'
    - 'quarter'
    - 'week'
    - 'days_in_month'
    - 'month_start'
    - 'month_end'
    - 'quarter_start'
    - 'quarter_end'
    - 'year_start'
    - 'year_end'
    - 'leap_year'
    
    Args:
        feature: pd.Series with datetime format, sample feature.
        extract: The list of date features to extract. 
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.        
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid.
    Returns:
        Refer to params `mode` explanation.
    """
    if config is None:
        if name is None:
            name = feature.name
        if isinstance(extract, str):
            extract = [extract]
        config = {'param':{'extract':extract, 'name':[f'{name}_{i}' for i in extract]},
                  'type':'datetime_extraction', 'variable':feature.name}
    if mode==2:
        return config
    else:
        var = ['year', 'month', 'day', 'hour', 'minute', 'second', 'date', 'time',
               'weekday', 'yearday', 'quarter', 'week', 'days_in_month',
               'month_start', 'month_end', 'quarter_start', 'quarter_end', 'year_start', 'year_end',
               'leap_year' ]
        if isinstance(extract, str):
            extract = [extract]
        for i in extract:
            assert i in var, f'extract variable `{i}` error, must be one of {var}.'
        data = []
        for i in extract:
            if i=='year':
                data.append(feature.dt.year)
            elif i=='month':
                data.append(feature.dt.month)
            elif i=='day':
                data.append(feature.dt.day)
            elif i=='hour':
                data.append(feature.dt.hour)
            elif i=='minute':
                data.append(feature.dt.minute)
            elif i=='second':
                data.append(feature.dt.second)
            elif i=='date':
                data.append(feature.dt.date)
            elif i=='time':
                data.append(feature.dt.time)
            elif i=='weekday':
                data.append(feature.dt.day_of_week)
            elif i=='yearday':
                data.append(feature.dt.day_of_year)
            elif i=='quarter':
                data.append(feature.dt.quarter)
            elif i=='week':
                data.append(feature.dt.isocalendar().week)
            elif i=='days_in_month':
                data.append(feature.dt.days_in_month)
            elif i=='month_start':
                data.append(feature.dt.is_month_start)
            elif i=='month_end':
                data.append(feature.dt.is_month_end)
            elif i=='quarter_start':
                data.append(feature.dt.is_quarter_start)
            elif i=='quarter_end':
                data.append(feature.dt.is_quarter_end)
            elif i=='year_start':
                data.append(feature.dt.is_year_start)
            elif i=='year_end':
                data.append(feature.dt.is_year_end)
            elif i=='leap_year':
                data.append(feature.dt.is_leap_year)
            
        data = pd.concat(data, axis=1)
        data.columns = config['param']['name']
        return data if mode else (data, config)