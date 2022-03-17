import requests

__all__ = ['BotFeiShu']

class BotFeiShu():
    def __init__(self, webhook):
        self.webhook = webhook
        
    def send_text(self, msg):
        t = requests.post(self.webhook, json={"msg_type": "text", "content":{"text": msg}})
        return t.content.decode()