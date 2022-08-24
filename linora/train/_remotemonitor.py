import json
import logging
import requests

from linora.utils._config import Config

__all__ = ['RemoteMonitor']


class RemoteMonitor():
    """Callback used to stream events to a server.

    Requires the `requests` library.
    Events are sent to `root + '/publish/epoch/end/'` by default. 
    Calls are HTTP POST, with a `data` argument which is a JSON-encoded dictionary of event data.
    If `send_as_json=True`, the content type of the request will be `"application/json"`.
    Otherwise the serialized JSON will be sent within a form.
    
    Args:
        root: String; root url of the target server.
        path: String; path relative to `root` to which the events will be sent.
        field: String; JSON field under which the data will be stored.
            The field is used only if the payload is sent within a form
            (i.e. send_as_json is set to False).
        headers: Dictionary; optional custom HTTP headers.
        send_as_json: Boolean; whether the request should be sent as `"application/json"`.
    """

    def __init__(self, root='http://localhost:9000', path='/publish/epoch/end/', 
                 field='data', headers=None, send_as_json=False):
        self._params = Config()
        self._params.root = root+path
        self._params.field = field
        self._params.headers = headers
        self._params.send_as_json = send_as_json
        self._params.name = 'RemoteMonitor'

    def _update(self, batch, log):
        """update log.
        
        Args:
            batch: Integer, index of batch.
            log: dict, name and value of loss or metrics;
        """
        send = {'batch':batch}
        for k, v in log.items():
            send[k] = v.item() if isinstance(v, (np.ndarray, np.generic)) else v
        try:
            if self.send_as_json:
                requests.post(self._params.root, json=send, headers=self._params.headers)
            else:
                requests.post(self._params.root, {self._params.field:json.dumps(send)}, headers=self._params.headers)
        except requests.exceptions.RequestException:
            logging.warning(f'Warning: could not reach RemoteMonitor root server at {self.root}')
