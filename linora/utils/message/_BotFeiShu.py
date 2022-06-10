import requests

__all__ = ['BotFeiShu']


class BotFeiShu():
    """send message by Feishu
    
    Args:
        webhook: Feishu robot url
    """
    def __init__(self, webhook):
        self.webhook = webhook
        
    def send_text(self, msg):
        """send message
        
        Args:
            msg: str, message
        Return:
            request post information.
        """
        t = requests.post(self.webhook, json={"msg_type": "text", "content":{"text": msg}})
        return t.content.decode()