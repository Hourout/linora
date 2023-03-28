import re
import json
import time
from urllib.parse import quote_plus
import hmac
import base64
import hashlib
import queue

import requests

from linora.utils._config import Config
from linora.utils._logger import Logger

__all__ = ['BotDingTalk']


class BotDingTalk():
    """Dingding group custom robot
    
    each robot can send up to 20 messages per minute, 
    supports three message types: text (text), link (link), and markdown.
    """
    def __init__(self, webhook, secret=None, pc_slide=False, fail_notice=False, logger=None, verbose=0):
        """
        Args:
            webhook: Dingding group custom robot webhook address.
            secret: The secret key that needs to be passed in when "Signing" is checked on the robot security settings page.
            pc_slide: The message link opening method, the default is False to open the browser, 
                      when set to True, the sidebar on the PC side is opened.
            fail_notice: message sending failure reminder, the default is False to not remind, 
                         the developer can judge and deal with it according to the returned message sending result.
            logger: Logger object, la.utils.Logger() class.
            verbose: Verbosity mode, 0 (silent), 1 (verbose).
        """
        self._params.Config()
        self._params.headers = {'Content-Type': 'application/json; charset=utf-8'}
        self._params.queue = queue.Queue(20)
        self._params.webhook = webhook
        self._params.secret = secret
        self._params.pc_slide = str.lower(pc_slide)
        self._params.fail_notice = fail_notice
        self._params.time_start = time.time()
        if self._params.secret is not None and self._params.secret.startswith('SEC'):
            self._update_webhook()
        if logger is None:
            logger = Logger()
        self._params.logger = logger
        self.params.verbose = verbose
            
    def _update_webhook(self):
        timestamp = round(self._params.time_start * 1000)
        string_to_sign = f'{timestamp}\n{self._params.secret}'
        hmac_code = hmac.new(self._params.secret.encode(), string_to_sign.encode(), digestmod=hashlib.sha256).digest()
        
        sign = quote_plus(base64.b64encode(hmac_code))
        if 'timestamp'in self.webhook:
            self._params.webhook = '{}&timestamp={}&sign={}'.format(self._params.webhook[:self._params.webhook.find('&timestamp')], str(timestamp), sign)
        else:
            self._params.webhook = '{}&timestamp={}&sign={}'.format(self._params.webhook, str(timestamp), sign)
            
    def _format_url(self, url):
        return f'dingtalk://dingtalkclient/page/link?url={quote_plus(url)}&pc_slide={self._params.pc_slide}'
    
    def _assert_msg(self, msg):
        assert len(msg.strip())>0, f"`{msg}` msg cannot be empty."

    def send_text(self, text, is_at_all=False, at_mobiles=[], at_dingtalk_ids=[], msg_add_mobile=True):
        """send text message.
        
        Args:
            text: message content.
            is_at_all: true is at all people, false is not at all people.
            at_mobiles: list of at people mobiles.
            at_dingtalk_ids: list of at people dingtalk ids.
            msg_add_mobile: message auto add to mobile.
        """
        self._assert_msg(text)
        data = {"msgtype": "text", "text":{"content": text}, "at": {}}
        if is_at_all:
            data["at"]["isAtAll"] = 'true'
        if at_mobiles:
            at_mobiles = list(map(str, at_mobiles))
            data["at"]["atMobiles"] = at_mobiles
            if msg_add_mobile:
                data["text"]["content"] = text + '\n@' + '@'.join(at_mobiles)
        if at_dingtalk_ids:
            data["at"]["atDingtalkIds"] = list(map(str, at_dingtalk_ids))
        return self._post(data)

    def send_image(self, image_url):
        """send image message.
        
        Args:
            image_url: image url.
        """
        self._assert_msg(image_url)
        data = {"msgtype":"image", "image":{"picURL":image_url}}
        return self._post(data)

    def send_link(self, text, title, message_url, image_url=''):
        """send link message.
        
        Args:
            text: message content, if it's too long, only part of it will be displayed.
            title: message title.
            message_url: link url, click the url to which the message is redirected.
            image_url: image url.
        """
        self._assert_msg(title)
        self._assert_msg(text)
        self._assert_msg(message_url)
        data = {"msgtype": "link",
                "link": {"text":text, "title":title, "picUrl":image_url, "messageUrl":self._format_url(message_url)}}
        return self._post(data)

    def send_markdown(self, text, title, is_at_all=False, at_mobiles=[], at_dingtalk_ids=[], msg_add_mobile=True):
        """send markdown message.
        
        Args:
            text: Message content in markdown format.
            title: The display content revealed by the above-the-fold session.
            is_at_all: true is at all people, false is not at all people.
            at_mobiles: list of at people mobiles.
            at_dingtalk_ids: list of at people dingtalk ids.
            msg_add_mobile: message auto add to mobile.
        """
        self._assert_msg(title)
        self._assert_msg(text)
        text = re.sub(r'(?<!!)\[.*?\]\((.*?)\)', lambda m: m.group(0).replace(m.group(1), self._format_url(m.group(1))), text)
        data = {"msgtype": "markdown", "markdown": {"title": title, "text": text}, "at": {}}
        if is_at_all:
            data["at"]["isAtAll"] = 'true'
        if at_mobiles:
            at_mobiles = list(map(str, at_mobiles))
            data["at"]["atMobiles"] = at_mobiles
            if is_auto_at:
                data["markdown"]["text"] = text + '\n@' + '@'.join(at_mobiles)
        if at_dingtalk_ids:
            data["at"]["atDingtalkIds"] = list(map(str, at_dingtalk_ids))
        return self._post(data)

    def send_action_card(self, title, text, btns, btn_orientation=0, hide_avatar=0):
        """send ActionCard message.
        
        Args:
            title: Display content from the above-the-fold session.
            text: message in markdown format
            btns: button list, when the number of buttons is 1, the overall jump to ActionCard type;
                  when the number of buttons is greater than 1, the independent jump to ActionCard type.
            btn_orientation: 0: Buttons are arranged vertically, 1: Buttons are arranged horizontally (optional).
            hide_avatar: 0: normal sender avatar, 1: hide sender avatar (optional).
        """
        if len(btns) == 1:
            data = {"msgtype": "actionCard",
                    "actionCard": {
                        "title":title,
                        "text":text,
                        "hideAvatar":hide_avatar,
                        "btnOrientation":btn_orientation,
                        "singleTitle":btns[0]["title"],
                        "singleURL":btns[0]["actionURL"]}}
        else:
            data = {"msgtype": "actionCard",
                    "actionCard": {
                        "title":title,
                        "text":text,
                        "hideAvatar":hide_avatar,
                        "btnOrientation":btn_orientation,
                        "btns":btns}}
        return self._post(data)

    def send_feed_card(self, links):
        """send FeedCard message.
        
        Args:
            links: a list of FeedLink isinstance or CardItem isinstance.
        """
        assert isinstance(links, list), '`links` format error, should be list.'
        link_list = []
        for link in links:
            assert isinstance(link, FeedLink) or isinstance(link, CardItem), f'`{link}` format error'
            link = link.get_data()
            link['messageURL'] = link['messageURL']
            link_list.append(link)
        data = {"msgtype": "feedCard", "feedCard": {"links": link_list}}
        return self._post(data)

    def make_card(self, title, url, image_url=None):
        """make FeedCard or ActionCard.
        
        Args:
            title: message name.
            url: URL triggered by click.
            image_url: default make ActionCard, if image_url is not None, make FeedCard.
        """
        self._assert_msg(title)
        self._assert_msg(url)
        if image_url is None:
            return {"title":title, "actionURL":self._format_url(url)}
        else:
            self._assert_msg(image_url)
            return {"title":title, "messageURL":self._format_url(url), "picURL":image_url}
        
    def _post(self, data):
        time_now = time.time()
        
        if time_now - self._params.time_start >= 3600 and self._params.secret is not None and self._params.secret.startswith('SEC'):
            self._params.time_start = time_now
            self._update_webhook()

        self._params.queue.put(time_now)
        if self._params.queue.full():
            elapse_time = time_now - self._params.queue.get()
            if elapse_time < 60:
                sleep_time = int(60 - elapse_time) + 1
                self._log(f'钉钉官方限制机器人每分钟最多发送20条，当前发送频率已达限制条件，休眠 {sleep_time}s')
                time.sleep(sleep_time)

        try:
            post_data = json.dumps(data)
            response = requests.post(self.webhook, headers=self.headers, data=post_data)
        except requests.exceptions.HTTPError as exc:
            self._log(f'message send fail, HTTP error: {exc.response.status_code}, reason: {exc.response.reason}')
            raise
        except requests.exceptions.ConnectionError:
            self._log("message send fail，HTTP connection error!")
            raise
        except requests.exceptions.Timeout:
            self._log("message send fail，Timeout error!")
            raise
        except requests.exceptions.RequestException:
            self._log("message send fail, Request Exception!")
            raise
        else:
            try:
                result = response.json()
            except JSONDecodeError:
                self._log("服务器响应异常，状态码：%s，响应内容：%s" % (response.status_code, response.text))
                return {'errcode': 500, 'errmsg': '服务器响应异常'}
            else:
                self._log('发送结果：%s' % result)
                # 消息发送失败提醒（errcode 不为 0，表示消息发送异常），默认不提醒，开发者可以根据返回的消息发送结果自行判断和处理
                if self.fail_notice and result.get('errcode', True):
                    time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    error_data = {
                      "msgtype": "text",
                      "text": {
                        "content": "[注意-自动通知]钉钉机器人消息发送失败，时间：%s，原因：%s，请及时跟进，谢谢!" % (
                          time_now, result['errmsg'] if result.get('errmsg', False) else '未知异常')
                        },
                      "at": {
                        "isAtAll": False
                        }
                      }
                    self._log("消息发送失败，自动通知：%s" % error_data)
                    requests.post(self.webhook, headers=self.headers, data=json.dumps(error_data))
                return result

    def _log(self, msg):
        if self._params.logger.params.log_file!='':
            self._params.logger.write(msg)
        if self._params.verbose:
            self._params.logger.info(msg)
