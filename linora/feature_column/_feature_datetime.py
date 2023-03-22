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