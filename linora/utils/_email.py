import os
import poplib
import smtplib
from base64 import b64encode
from email.encoders import encode_base64
from email.header import Header
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from linora.utils._config import Config
from linora.utils._email_utils import parse_mail, parse_headers

__all__ = ['EMail']


poplib._MAXLINE = 4096
mail_personal = {
    '163.com': {
        'smtp_host': 'smtp.163.com',
        'smtp_port': 994,
        'smtp_ssl': True,
        'smtp_tls': False,
        'pop_host': 'pop.163.com',
        'pop_port': 995,
        'pop_ssl': True,
        'pop_tls': False,
        'imap_host': 'imap.163.com',
        'imap_port': 993,
        'imap_ssl': True,
        'imap_tls': False
    },
    '126.com': {
        'smtp_host': 'smtp.126.com',
        'smtp_port': 994,
        'smtp_ssl': True,
        'smtp_tls': False,
        'pop_host': 'pop.126.com',
        'pop_port': 995,
        'pop_ssl': True,
        'pop_tls': False,
        'imap_host': 'imap.126.com',
        'imap_port': 993,
        'imap_ssl': True,
        'imap_tls': False
    },
    'yeah.net': {
        'smtp_host': 'smtp.yeah.net',
        'smtp_port': 994,
        'smtp_ssl': True,
        'smtp_tls': False,
        'pop_host': 'pop.yeah.net',
        'pop_port': 995,
        'pop_ssl': True,
        'pop_tls': False,
        'imap_host': 'imap.yeah.net',
        'imap_port': 993,
        'imap_ssl': True,
        'imap_tls': False
    },
    'qq.com': {
        'smtp_host': 'smtp.qq.com',
        'smtp_port': 465,
        'smtp_ssl': True,
        'smtp_tls': False,
        'pop_host': 'pop.qq.com',
        'pop_port': 995,
        'pop_ssl': True,
        'pop_tls': False,
    },
    'gmail.com': {
        'smtp_host': 'smtp.gmail.com',
        'smtp_port': 465,
        'smtp_ssl': True,
        'smtp_tls': False,
        'pop_host': 'pop.gmail.com',
        'pop_port': 995,
        'pop_ssl': True,
        'pop_tls': False,
    },
    'sina.com': {
        'smtp_host': 'smtp.sina.com',
        'smtp_port': 465,
        'smtp_ssl': True,
        'smtp_tls': False,
        'pop_host': 'pop.sina.com',
        'pop_port': 995,
        'pop_ssl': True,
        'pop_tls': False,
    },
    'outlook.com': {
        'smtp_host': 'smtp-mail.outlook.com',
        'smtp_port': 587,
        'smtp_ssl': False,
        'smtp_tls': True,
        'pop_host': 'pop.outlook.com',
        'pop_port': 995,
        'pop_ssl': True,
        'pop_tls': False,
    },
    'hotmail.com': {
        'smtp_host': 'smtp.office365.com',
        'smtp_port': 587,
        'smtp_ssl': False,
        'smtp_tls': True,
        'pop_host': 'outlook.office365.com',
        'pop_port': 995,
        'pop_ssl': True,
        'pop_tls': False,
    },
}

mail_enterprise = {
    'qq': {
        'smtp_host': 'smtp.exmail.qq.com',
        'smtp_port': 465,
        'smtp_ssl': True,
        'smtp_tls': False,
        'pop_host': 'pop.exmail.qq.com',
        'pop_port': 995,
        'pop_ssl': True,
        'pop_tls': False
    },
    'ali': {
        'smtp_host': 'smtp.mxhichina.com',
        'smtp_port': 465,
        'smtp_ssl': True,
        'smtp_tls': False,
        'pop_host': 'pop3.mxhichina.com',
        'pop_port': 995,
        'pop_ssl': True,
        'pop_tls': False
    },
    '163': {
        'smtp_host': 'smtp.qiye.163.com',
        'smtp_port': 994,
        'smtp_ssl': True,
        'smtp_tls': False,
        'pop_host': 'pop.qiye.163.com',
        'pop_port': 995,
        'pop_ssl': True,
        'pop_tls': False
    },
    'google': {
        'smtp_host': 'smtp.gmail.com',
        'smtp_port': 465,
        'smtp_ssl': True,
        'smtp_tls': False,
        'pop_host': 'pop.gmail.com',
        'pop_port': 995,
        'pop_ssl': True,
        'pop_tls': False,
    },
}

mail_default = {
    'smtp_host': 'smtp.',
    'smtp_port': 465,
    'smtp_ssl': True,
    'smtp_tls': False,
    'pop_host': 'pop.',
    'pop_port': 995,
    'pop_ssl': True,
    'pop_tls': False,
    'imap_host': 'imap.',
    'imap_port': 993,
    'imap_ssl': True,
    'imap_tls': False
}

class EMail():
    def __init__(self, username, password, smtp_host=None, smtp_port=None, smtp_ssl=None, smtp_tls=None,
                 pop_host=None, pop_port=None, pop_ssl=None, pop_tls=None, enterprise=None):
        """ Return MailServer instance, it implements all SMTP and POP functions.
        
        The module is inspired by zmail.
        
        Args:
            username: email name.
            password: email password.
            smtp_host: email smtp host.
            smtp_port: email smtp port.
            smtp_ssl: email smtp ssl.
            smtp_tls: email smtp tls.
            pop_host: email pop host.
            pop_port: email pop port.
            pop_ssl: email pop ssl.
            pop_tls: email pop tls.
            enterprise: Shortcut for use enterprise mail,if specified, enterprise mail configs will replace all inner auto-generate configs. support ['qq', 'ali', '163', 'google']
        """
        self._params = Config()
        self._params.mail_personal = mail_personal
        self._params.mail_enterprise = mail_enterprise
        self._params.mail_default = mail_default
        self._params.username = username
        self._params.password = password
        
        user_define_config = {
            'smtp_host': smtp_host,
            'smtp_port': smtp_port,
            'smtp_ssl': smtp_ssl,
            'smtp_tls': smtp_tls,
            'pop_host': pop_host,
            'pop_port': pop_port,
            'pop_ssl': pop_ssl,
            'pop_tls': pop_tls}
        mail_config = self._check_supported_server(username, enterprise)
        mail_config.update({k: v for k, v in mail_config.items() if v is not None})
        self._params.mail_config = {k: v for k, v in mail_config.items() if 'imap' not in k}
        self._params.mime = None
        self._params.timeout = 60
        
    def send_mail(self, recipients, mail, cc=None, timeout=None, auto_add_from=True, auto_add_to=True):
        """"Send email.
        
        Args:
            recipients: recipient user email, can either be str or a list of str.
            mail: dict, mail content.
            cc: Mail cc object.
            timeout: if is not None, it will replace server's timeout.
            auto_add_from: If set to True, when the key ' 'from' (case-insensitive) not in mail(For send), the default 'from' will automatically added to mail.
            auto_add_to: If set to True, when the key 'to' (case-insensitive) not in mail(For send), the default 'to' will automatically added to mail.
        """
        if timeout is not None:
            self._params.timeout = timeout
        if auto_add_from and mail.get('From') is None:
            if self._params.mime is None:
                self._make_mime(mail)
            self._params.mime['From'] = self._make_address_header([self._params.username])
        
        recipients = recipients if  isinstance(recipients, list) else [recipients]
        if auto_add_to and mail.get('To') is None:
            if self._params.mime is None:
                self._make_mime(mail)
            self._params.mime['To'] = self._make_address_header(recipients)
            
        if cc is not None:
            cc = cc if isinstance(cc, list) else [cc]
            recipients.extend(cc)
            if self._params.mime is None:
                self._make_mime(mail)
            self._params.mime['Cc'] = self._make_address_header(cc)
        
        recipients = [i[1] if isinstance(i, tuple) else i for i in recipients]
        self._login()
        self._params.server.timeout = self._params.timeout
        self._params.server.sendmail(self._params.username, recipients, self._params.mime.as_string())
        self._logout()
        self._params.mime = None
        return True
    
    def delete(self, which):
        """Delete mail.
        
        Args:
            which: int, which is a int number that represent mail's position in mailbox.
                The which must between 1 and message count(return from Mail.stat())
        """
        self._login(send=False)
        self._params.server.dele(which)
        self._logout(send=False)
        return True

    def stat(self):
        """Get mailbox status.
        The result is a tuple of 2 integers: (message count, mailbox size).
        """
        self._login(send=False)
        status = self._params.server.stat()
        self._logout(send=False)
        return status

    def get_mail(self, which=None, subject=None, sender=None, start_time=None, end_time=None, 
                  start_index=None, end_index=None):
        """Get a list of mails from mailbox."""
        if which is not None:
            if not isinstance(which, (list, tuple)):
                which = [which]
        elif subject or sender or start_time or end_time or start_index or end_index:
            headers = self.get_headers(start_index, end_index)

            which = []
            for header in headers:
                mail_subject = header.get('Subject')
                mail_sender = header.get('From')
                mail_date = header.get('date')
                ok = True
                if subject is not None:
                    if mail_subject is None or subject not in mail_subject:
                        ok = False
                if sender is not None:
                    if mail_sender is None or sender not in mail_sender:
                        ok = False
                if start_time is not None:
                    if mail_date is None or start_time > mail_date:
                        ok = False
                if end_time is not None:
                    if mail_date is None or end_time < mail_date:
                        ok = False
                if ok:
                    which.append(header['id'])
            which.sort()
        else:
            which = [self.stat()[0]]
        self._login(send=False)
        mail = [parse_mail(self._params.server.retr(i)[1], i) for i in which]
        self._logout(send=False)
        return mail



    def get_headers(self, start_index=None, end_index=None):
        """Get mails headers."""
        self._login(send=False)
        end = self._params.server.stat()[0]
        if start_index is None:
            start_index = 1
        if end_index is None:
            end_index = end
        index = range(min(start_index, end_index), max(start_index, end_index)+1)
        header = [self._params.server.top(i, 0)[1] for i in index]
        headers = []
        for i, mail_header in enumerate(header):
            _, _headers, *__ = parse_headers(mail_header)
            _headers.update(id=index[i])
            try:
                headers.append({j:_headers[j] for j in {'date', 'Subject','From', 'id'}})
            except:
                print(_headers)
        self._logout(send=False)
        return headers

    def check_available(self):
        """check email connection status."""
        try:
            self._login()
            self._logout()
            check = {'stmp server':True, 'stmp info':'ok'}
        except Exception as e:
            check = {'stmp server':False, 'stmp info':str(e)}
        try:
            self._login(send=False)
            self._logout(send=False)
            check.update({'pop server':True, 'pop info':'ok'})
        except Exception as e:
            check.update({'pop server': False, 'pop info':str(e)})
        return check
        
    def _check_supported_server(self, mail_address, config=None):
        provider = mail_address.split('@')[1]

        if config is not None:
            if config in self._params.mail_enterprise:
                return self._params.mail_enterprise[config]
            else:
                raise ValueError(f'Only supported enterprise server [{str(list(self._params.mail_enterprise))[1:-1]}].')

        if provider in self._params.mail_personal:
            return self._params.mail_personal[provider]
        else:
            config = self._params.mail_default.copy()
            config['smtp_host'] += provider
            config['pop_host'] += provider
            config['imap_host'] += provider
            return config
        
    def _login(self, send=True):
        if send:
            if self._params.mail_config['smtp_ssl']:
                self._params.server = smtplib.SMTP_SSL(self._params.mail_config['smtp_host'], self._params.mail_config['smtp_port'], 'email.local', timeout=self._params.timeout)
            else:
                self._params.server = smtplib.SMTP(self._params.mail_config['smtp_host'], self._params.mail_config['smtp_port'], 'email.local', timeout=self._params.timeout)
            if self._params.mail_config['smtp_tls']:
                self._params.server.ehlo()
                self._params.server.starttls()
                self._params.server.ehlo()
            self._params.server.login(self._params.username, self._params.password)
        else:
            if self._params.mail_config['pop_ssl']:
                self._params.server = poplib.POP3_SSL(self._params.mail_config['pop_host'], self._params.mail_config['pop_port'], timeout=self._params.timeout)
            else:
                self._params.server = poplib.POP3(self._params.mail_config['pop_host'], self._params.mail_config['pop_port'], timeout=self._params.timeout)
            if self._params.mail_config['pop_tls']:
                self._params.server.stls()
            self._params.server.user(self._params.username)
            self._params.server.pass_(self._params.password)
                
    def _logout(self, send=True):
        if send:
            try:
                code, message = self._params.server.docmd("QUIT")
                if code != 221:
                    raise smtplib.SMTPResponseException(code, message)
            except smtplib.SMTPServerDisconnected:
                pass
            finally:
                self._params.server.close()
        else:
            self._params.server.quit()
        
    def _make_mime(self, mail):
        mime = MIMEMultipart(boundary=None)
        for k, v in mail.items():
            _k = k.lower()
            if _k in ('subject', 'from'):
                if isinstance(v, str):
                    mime[_k.capitalize()] = v
                else:
                    raise InvalidArguments('{} can only be str! Got {} instead.'.format(_k.capitalize(), type(v)))
#             elif _k == 'to':
                
#                 if not all([(i in mail) for i in
#                     ('from', 'to', 'subject', 'raw_headers', 'charsets', 'headers',
#                      'date', 'id', 'raw', 'attachments', 'content_text', 'content_html')]):
#                     warnings.warn("Header 'to' is invalid and unused,if you want to add address name "
#                                   "use tuple (address,name) instead.", category=DeprecationWarning, stacklevel=4)
#             elif _k in ('attachments', 'content_text', 'content_html', 'headers'):
#                 pass
#             else:
#                 if not self._is_resend_mail():
                    # Remove resend warnings.
#                     warnings.warn("Header '{}' is invalid and unused,if you want to add extra headers "
#                                   "use 'headers' instead.".format(str(_k)), category=DeprecationWarning, stacklevel=4)

        # Set extra headers.
        if mail.get('headers') and isinstance(mail['headers'], dict):
            for k, v in mail['headers'].items():
                mime[k] = v

        # Set HTML content.
        if mail.get('content_html') is not None:
            html = mail['content_html'] if isinstance(mail['content_html'], list) else [mail['content_html']]
            for i in html:
                mime.attach(MIMEText('{}'.format(i), 'html', 'utf-8'))

        # Set TEXT content.
        if mail.get('content_text') is not None:
            text = mail['content_text'] if isinstance(mail['content_text'], list) else [mail['content_text']]
            for i in text:
                mime.attach(MIMEText('{}'.format(i), 'plain', 'utf-8'))

        # Set attachments.
        if mail.get('attachments'):
            attachments = mail['attachments'] if isinstance(mail['attachments'], list) else [mail['attachments']]
            for attachment in attachments:
                if isinstance(attachment, str):
                    if not os.path.exists(attachment):
                        raise FileExistsError(f"The file {attachment} doesn't exist.")
                    name = os.path.split(attachment)[1]
                    encoded_name = Header(name).encode()
                    with open(attachment, 'rb') as f:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(f.read())
                        part['Content-Disposition'] = f'attachment;filename="{encoded_name}"'
                        encode_base64(part)
                    mime.attach(part)
                elif isinstance(attachment, tuple):
                    name, raw = attachment
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(raw)
                    part['Content-Disposition'] = f'attachment;filename="{name}"'
                    encode_base64(part)
                    mime.attach(part)
                else:
                    raise InvalidArguments(f'Attachments excepted str or tuple got {type(attachment)} instead.')
        self._params.mime = mime
        
    def _make_address_header(self, address_list):
        """Used for make 'To' 'Cc' 'From' header."""
        res = []
        for address in address_list:
            if isinstance(address, tuple):
                assert len(address) == 2, 'Only two arguments!'
                name, rel_address = address
                name = '' if not name else f"=?utf-8?b?{b64encode(name.encode('utf-8')).decode('ascii')}?="
                res.append(f'{name} <{rel_address}>')
            elif isinstance(address, str):
                res.append(f'<{address}>')
            else:
                raise InvalidArguments(f'Email address can only be tuple or str.Get {type(address)} instead.')
        return ', '.join(res)
    
    