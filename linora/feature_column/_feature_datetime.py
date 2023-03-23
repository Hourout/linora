from linora.feature_column._datetime import *


class FeatureDatetime(object):
    def __init__(self):
        self.pipe = {}
        
    def datetime_extraction(self, variable, extract, name=None, keep=True):
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
            variable: str, feature variable name.
            extract: The list of date features to extract. 
            name: str, output feature name, if None, name is feature.name .
            keep: If the `name` is output only once in the calculation, the `name` will be kept in the final result.
        """
        config = {'param':{'extract':extract, 'name':variable if name is None else name},
                  'type':'datetime_extraction', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    def datetime_transform(self, variable, format_in, format_out='timestamp', name=None, keep=True):
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
            variable: str, feature variable name.
            format_in: input time format.
            format_out: output time format.
            name: str, output feature name, if None, name is feature.name .
            keep: If the `name` is output only once in the calculation, the `name` will be kept in the final result.
        """
        config = {'param':{'format_in':format_in, 'format_out':format_out,
                           'name':variable if name is None else name},
                  'type':'datetime_transform', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self

