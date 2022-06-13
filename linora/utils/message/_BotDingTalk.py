import re
import sys
import json
import time
from urllib.parse import quote_plus
import hmac
import base64
import hashlib
import queue

import requests

from linora.utils._config import Config

class BotDingTalk():
    """Dingding group custom robot
    
    each robot can send up to 20 messages per minute, 
    supports three message types: text (text), link (link), and markdown.
    """
    def __init__(self, webhook, secret=None, pc_slide=False, fail_notice=False):
        """
        Args:
            webhook: Dingding group custom robot webhook address.
            secret: The secret key that needs to be passed in when "Signing" is checked on the robot security settings page.
            pc_slide: The message link opening method, the default is False to open the browser, 
                      when set to True, the sidebar on the PC side is opened.
            fail_notice: message sending failure reminder, the default is False to not remind, 
                         the developer can judge and deal with it according to the returned message sending result.
        """
#         super(BotDingtalk, self).__init__()
        self._params.Config()
        self._params.headers = {'Content-Type': 'application/json; charset=utf-8'}
        self._params.queue = queue.Queue(20)  # 钉钉官方限流每分钟发送20条信息
        self._params.webhook = webhook
        self._params.secret = secret
        self._params.pc_slide = str.lower(pc_slide)
        self._params.fail_notice = fail_notice
        self._params.start_time = time.time()  # 加签时，请求时间戳与请求时间不能超过1小时，用于定时更新签名
        if self._params.secret is not None and self._params.secret.startswith('SEC'):
            self._update_webhook()
            
    def _update_webhook(self):
        timestamp = round(self._params.start_time * 1000)
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

    def send_action_card(self, action_card):
        """
        ActionCard类型
        :param action_card: 整体跳转ActionCard类型实例或独立跳转ActionCard类型实例
        :return: 返回消息发送结果
        """
        if isinstance(action_card, ActionCard):
            data = action_card.get_data()
            
            if "singleURL" in data["actionCard"]:
                data["actionCard"]["singleURL"] = self._format_url(data["actionCard"]["singleURL"])
            elif "btns" in data["actionCard"]:
                for btn in data["actionCard"]["btns"]:
                    btn["actionURL"] = self._format_url(btn["actionURL"])
            
            logging.debug("ActionCard类型：%s" % data)
            return self._post(data)
        else:
            logging.error("ActionCard类型：传入的实例类型不正确，内容为：{}".format(str(action_card)))
            raise TypeError("ActionCard类型：传入的实例类型不正确，内容为：{}".format(str(action_card)))

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
            link['messageURL'] = self._format_url(link['messageURL'])
            link_list.append(link)
        data = {"msgtype": "feedCard", "feedCard": {"links": link_list}}
        return self._post(data)

    def _post(self, data):
        now = time.time()
        
        # 钉钉自定义机器人安全设置加签时，签名中的时间戳与请求时不能超过一个小时，所以每个1小时需要更新签名
        if now - self._params.start_time >= 3600 and self._params.secret is not None and self._params.secret.startswith('SEC'):
            self._params.start_time = now
            self._update_webhook()

        # 钉钉自定义机器人现在每分钟最多发送20条消息
        self._params.queue.put(now)
        if self._params.queue.full():
            elapse_time = now - self._params.queue.get()
            if elapse_time < 60:
                sleep_time = int(60 - elapse_time) + 1
                logging.debug('钉钉官方限制机器人每分钟最多发送20条，当前发送频率已达限制条件，休眠 {}s'.format(str(sleep_time)))
                time.sleep(sleep_time)

        try:
            post_data = json.dumps(data)
            response = requests.post(self.webhook, headers=self.headers, data=post_data)
        except requests.exceptions.HTTPError as exc:
            logging.error("消息发送失败， HTTP error: %d, reason: %s" % (exc.response.status_code, exc.response.reason))
            raise
        except requests.exceptions.ConnectionError:
            logging.error("消息发送失败，HTTP connection error!")
            raise
        except requests.exceptions.Timeout:
            logging.error("消息发送失败，Timeout error!")
            raise
        except requests.exceptions.RequestException:
            logging.error("消息发送失败, Request Exception!")
            raise
        else:
            try:
                result = response.json()
            except JSONDecodeError:
                logging.error("服务器响应异常，状态码：%s，响应内容：%s" % (response.status_code, response.text))
                return {'errcode': 500, 'errmsg': '服务器响应异常'}
            else:
                logging.debug('发送结果：%s' % result)
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
                    logging.error("消息发送失败，自动通知：%s" % error_data)
                    requests.post(self.webhook, headers=self.headers, data=json.dumps(error_data))
                return result


class ActionCard(object):
    """
    ActionCard类型消息格式（整体跳转、独立跳转）
    """
    def __init__(self, title, text, btns, btn_orientation=0, hide_avatar=0):
        """
        ActionCard初始化
        :param title: 首屏会话透出的展示内容
        :param text: markdown格式的消息
        :param btns: 按钮列表：（1）按钮数量为1时，整体跳转ActionCard类型；（2）按钮数量大于1时，独立跳转ActionCard类型；
        :param btn_orientation: 0：按钮竖直排列，1：按钮横向排列（可选）
        :param hide_avatar: 0：正常发消息者头像，1：隐藏发消息者头像（可选）
        """
        super(ActionCard, self).__init__()
        self.title = title
        self.text = text
        self.btn_orientation = btn_orientation
        self.hide_avatar = hide_avatar
        btn_list = []
        for btn in btns:
            if isinstance(btn, CardItem):
                btn_list.append(btn.get_data())
        if btn_list:
            btns = btn_list  # 兼容：1、传入CardItem示例列表；2、传入数据字典列表
        self.btns = btns

    def get_data(self):
        """
        获取ActionCard类型消息数据（字典）
        :return: 返回ActionCard数据
        """
        if all(map(is_not_null_and_blank_str, [self.title, self.text])) and len(self.btns):
            if len(self.btns) == 1:
                # 整体跳转ActionCard类型
                data = {
                        "msgtype": "actionCard",
                        "actionCard": {
                            "title": self.title,
                            "text": self.text,
                            "hideAvatar": self.hide_avatar,
                            "btnOrientation": self.btn_orientation,
                            "singleTitle": self.btns[0]["title"],
                            "singleURL": self.btns[0]["actionURL"]
                        }
                }
                return data
            else:
                # 独立跳转ActionCard类型
                data = {
                    "msgtype": "actionCard",
                    "actionCard": {
                        "title": self.title,
                        "text": self.text,
                        "hideAvatar": self.hide_avatar,
                        "btnOrientation": self.btn_orientation,
                        "btns": self.btns
                    }
                }
                return data
        else:
            logging.error("ActionCard类型，消息标题或内容或按钮数量不能为空！")
            raise ValueError("ActionCard类型，消息标题或内容或按钮数量不能为空！")


class FeedLink(object):
    """
    FeedCard类型单条消息格式
    """
    def __init__(self, title, message_url, pic_url):
        """
        初始化单条消息文本
        :param title: 单条消息文本
        :param message_url: 点击单条信息后触发的URL
        :param pic_url: 点击单条消息后面图片触发的URL
        """
        super(FeedLink, self).__init__()
        self.title = title
        self.message_url = message_url
        self.pic_url = pic_url

    def get_data(self):
        """
        获取FeedLink消息数据（字典）
        :return: 本FeedLink消息的数据
        """
        if all(map(is_not_null_and_blank_str, [self.title, self.message_url, self.pic_url])):
            data = {
                    "title": self.title,
                    "messageURL": self.message_url,
                    "picURL": self.pic_url
            }
            return data
        else:
            logging.error("FeedCard类型单条消息文本、消息链接、图片链接不能为空！")
            raise ValueError("FeedCard类型单条消息文本、消息链接、图片链接不能为空！")


class CardItem(object):
    """
    ActionCard和FeedCard消息类型中的子控件
    
    注意：
    1、发送FeedCard消息时，参数pic_url必须传入参数值；
    2、发送ActionCard消息时，参数pic_url不需要传入参数值；
    """

    def __init__(self, title, url, pic_url=None):
        """
        CardItem初始化
        @param title: 子控件名称
        @param url: 点击子控件时触发的URL
        @param pic_url: FeedCard的图片地址，ActionCard时不需要，故默认为None
        """
        super(CardItem, self).__init__()
        self.title = title
        self.url = url
        self.pic_url = pic_url

    def get_data(self):
        """
        获取CardItem子控件数据（字典）
        @return: 子控件的数据
        """
        if all(map(is_not_null_and_blank_str, [self.title, self.url, self.pic_url])):
            # FeedCard类型
            data = {
                "title": self.title,
                "messageURL": self.url,
                "picURL": self.pic_url
            }
            return data
        elif all(map(is_not_null_and_blank_str, [self.title, self.url])):
            # ActionCard类型
            data = {
                "title": self.title,
                "actionURL": self.url
            }
            return data
        else:
            logging.error("CardItem是ActionCard的子控件时，title、url不能为空；是FeedCard的子控件时，title、url、pic_url不能为空！")
            raise ValueError("CardItem是ActionCard的子控件时，title、url不能为空；是FeedCard的子控件时，title、url、pic_url不能为空！")


if __name__ == '__main__':
    import doctest
    doctest.testmod()