__all__ = ['datetime_extraction', 'datetime_transform']


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


def datetime_transform(feature, format_in, format_out='timestamp', mode=0, name=None, config=None):
    """Time format transform.
    
    The transformer supports the time format of the following features:
    
    - 'timenumber:s', 's' id unit, one of ['s', 'ms', 'us', 'ns']:
        1679455111, 1679455111.0958722
    - 'strtime:format', format is "%Y-%m-%d %H:%M:%S":
        %y Two-digit Year Representation (00-99)
        %Y Four-digit Year Representation (000-9999)
        %m month (01-12)
        %d One day in months (0-31)
        %h Hours in 24-hour clock (0-23)
        %I 12-hour Hours (01-12)
        %M minutes (00=59)
        %S seconds (00-59)
        %a local simplified week name
        %A local full week name
        %b Local Simplified Month Name
        %B Local full month name
        %c local corresponding date representation and time representation
        %j One day in year (001-366)
        %p equivalent of local a.m. or p.m.
        %U Number of weeks in a year (00-53) Sunday is the beginning of the week.
        %w week (0-6), Sunday is the beginning of the week.
        %W Number of weeks in a year (00-53) Monday is the beginning of the week.
        %x local corresponding date representation
        %X local corresponding time representation
        %Z Name of the current time zone
        eg:
        - "%Y-%m-%d %H:%M:%S": '2023-01-01 02:03:56'
        - "%A %Y-%m %H:%M:%S": 'Thursday 2023-01 02:03:56'
    - 'timestamp':
        Timestamp(2023-01-01 02:03:56)
    - 'datetime'
        datetime.datetime(2023, 3, 23, 8, 9, 59, 733760)
    
    Args:
        feature: pd.Series with datetime format, sample feature.
        format_in: input time format.
        format_out: output time format.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.        
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid.
    Returns:
        Refer to params `mode` explanation.
    """
    if config is None:
        config = {'param':{'format_in':format_in, 'format_out':format_out,
                          'name':feature.name if name is None else name},
                  'type':'datetime_transform', 'variable':feature.name}
    if mode==2:
        return config
    else:
        format_in_mode = format_in.split(':')[0]
        format_in_param = format_in[len(format_in_mode)+1:]
        format_out_mode = format_out.split(':')[0]
        format_out_param = format_out[len(format_out_mode)+1:]

        if format_in_mode=='timenumber':
            if format_in_param=='':
                format_in_param = 's'
            t = pd.to_datetime(feature, unit=format_in_param)
        elif format_in_mode=='strtime':
            if format_in_param=='':
                format_in_param = None
            t = pd.to_datetime(feature, format=format_in_param)
        elif format_in_mode=='datetime':
            t = pd.to_datetime(feature)
        elif format_in_mode=='timestamp':
            t = feature
        else:
            raise ValueError('`format_in` value error.')

        if format_out_mode=='timenumber':
            t = t.map(lambda x: x.timestamp())
        elif format_out_mode=='strtime':
            if format_out_param=='':
                format_out_param = "%Y-%m-%d %H:%M:%S"
            t = t.dt.strftime(format_out_param)
        elif format_out_mode=='datetime':
            t = t.map(lambda x: x.to_pydatetime())
        elif format_out_mode=='timestamp':
            pass
        else:
            raise ValueError('`format_out` value error.')
        return t.rename(config['param']['name']) if mode else (t.rename(config['param']['name']), config)
 
