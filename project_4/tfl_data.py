# -----------------------Tom Keane CID: 01788365----------------------
import requests
from datetime import datetime
api_log = []


def __limit_check():
    limit_reached = False
    n = len(api_log)
    if n >= 5:
        tdelta = datetime.now() - api_log[n - 6]
        if tdelta.seconds < 300:
            limit_reached = True
    return limit_reached


def get_line_severity(line_id=''):
    # Takes as input a name of a tube line and returns its current status
    if type(line_id) != str:
        raise TypeError("line_id must be a string")
    if not __limit_check():
        api_log.append(datetime.now())
        response = requests.get('https://api.tfl.gov.uk/Line/' + line_id + '/Status')
        api_output = response.json()
        if type(api_output) == list:
            status = api_output[0]['lineStatuses'][0]['statusSeverityDescription']
        else:
            raise ValueError(api_output["message"])
    else:
        raise QuotaError("Can’t make an API call now due to quota limit, try again in a few minutes.")
    return status


def get_air_quality(is_future):
    """ Takes a boolean is_future, if it is True returns one word summary of the
    future forecast of air quality as determined by TfL, if false returns the one
    word summary of current state of air quality as determined by TfL. """
    if type(is_future) != bool:
        raise TypeError("is_future must be a Boolean")
    if not __limit_check():
        api_log.append(datetime.now())
        response = requests.get('https://api.tfl.gov.uk/AirQuality')
        api_output = response.json()
        air_quality = api_output["currentForecast"][is_future]["forecastBand"]
    else:
        raise QuotaError("Can’t make an API call now due to quota limit, try again in a few minutes.")
    return air_quality


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class QuotaError(Error):
    """Exception raised when API quota is reached.
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
