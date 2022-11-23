import datetime
import re
from base64 import b64decode
from datetime import timedelta, timezone, tzinfo
from email.header import decode_header
from quopri import decodestring
from urllib.parse import unquote
from collections import OrderedDict
from collections.abc import Mapping, MutableMapping


class CaseInsensitiveDict(MutableMapping):
    """A case-insensitive ``dict``-like object.
    Inspired by requests.structures
    _store -> 'key.lower()' : ('key','value')
    """

    def __init__(self, data=None, **kwargs):
        self._store = OrderedDict()
        if data is None:
            data = {}
        self.update(data, **kwargs)

    def __setitem__(self, key, value):
        self._store[key.lower()] = (key, value)

    def __getitem__(self, key):
        return self._store[key.lower()][1]

    def __delitem__(self, key):
        del self._store[key.lower()]

    def __iter__(self):
        return (casedkey for casedkey, _ in self._store.values())

    def __len__(self):
        return len(self._store)

    def lower_items(self):
        """Like iteritems(), but with all lowercase keys."""
        return (
            (lowerkey, value[1])
            for (lowerkey, value)
            in self._store.items()
        )

    def __eq__(self, other):
        if isinstance(other, Mapping):
            other = CaseInsensitiveDict(other)
        else:
            return NotImplemented
        # Compare insensitively
        return dict(self.lower_items()) == dict(other.lower_items())

    def copy(self):
        """Shallow copy."""
        return CaseInsensitiveDict(self._store.values())

    def __repr__(self):
        return str(dict(self.items()))


TYPE_MULTIPART = 'multipart'
TYPE_TEXT_PLAIN = ('text', 'plain')
TYPE_TEXT_HTML = ('text', 'html')
DATE_PATTERN_1 = re.compile(r'(\w+),\s+([0-9]+)\s+(\w+)\s+([0-9]+)\s+([0-9]+):([0-9]+):([0-9]+)\s+(.+)')
DATE_PATTERN_2 = re.compile(r'([0-9]+)\s+([\w]+)\s+([0-9]+)\s+([0-9]+):([0-9]+):([0-9]+)\s+(.+)')
TIMEZONE_PATTERN = re.compile(re.compile(r"([+\-])([0-9])?([0-9])?([0-9])?([0-9])?"))
TIMEZONE_MINUTE_OFFSET = (600, 60, 10, 1)
FILENAME_PATTERN = re.compile(re.compile(r"([^']+)'([^']*)'(.+)"))
MONTH_TO_INT = CaseInsensitiveDict({
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr': 4,
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Aug': 8,
    'Sep': 9,
    'Oct': 10,
    'Nov': 11,
    'Dec': 12,
})
HEADER_VALUE_STRIP = '\r\n "'



def recursive_decode(_bytes, encodings):
    """Recursive decode bytes to str."""
    for encoding in encodings:
        try:
            return _bytes.decode(encoding)
        except UnicodeDecodeError:
            pass
    return None


def remove_line_feed_and_whitespace(_value):
    _value = _value.strip(HEADER_VALUE_STRIP)

    while r'\r\n' == _value[:4]:
        _value = _value[4:]
    while r'\r\n' == _value[-4:]:
        _value = _value[:-4]

    return _value


def parse_header_value(bvalue, encodings):
    """Parse mail-specified-header to real value."""
    value = recursive_decode(bvalue, encodings)

    # Decode header without converting charset.
    if value is not None:
        decoded_value = ''
        for _value, _charset in decode_header(value):
            if _charset is not None:
                try:
                    decoded_value += _value.decode(_charset)
                except UnicodeDecodeError:
                    break
            else:
                if isinstance(_value, bytes):
                    decoded_value += _value.decode('utf-8')
                elif isinstance(_value, str):
                    decoded_value += _value
        return decoded_value
    return None


def _fmt_date_tz(tz):
    match_groups = TIMEZONE_PATTERN.match(tz).groups()
    _minute_offset = 0
    if match_groups[0] == '+':
        for i, v in enumerate(match_groups[1:]):
            if v is None:
                continue
            _minute_offset += int(TIMEZONE_MINUTE_OFFSET[i] * int(v))
        return timezone(timedelta(minutes=_minute_offset))
    elif match_groups[0] == '-':
        for i, v in enumerate(match_groups[1:]):
            if v is None:
                continue
            _minute_offset += int(TIMEZONE_MINUTE_OFFSET[i] * int(v))
        return timezone(-timedelta(minutes=_minute_offset))
    return None


def fmt_date(date_as_string):
    """Convert mail header Date to datetime object."""
    match_one = DATE_PATTERN_1.fullmatch(date_as_string)
    match_two = DATE_PATTERN_2.fullmatch(date_as_string)
    if match_one:
        week, day, month_as_string, year, hour, minute, second, time_zone = match_one.groups()
        month = MONTH_TO_INT[month_as_string]
        tz = _fmt_date_tz(time_zone)
        return datetime.datetime(int(year), month, int(day),
                                 int(hour), int(minute), int(second), tzinfo=tz)
    elif match_two:
        day, month_as_string, year, hour, minute, second, time_zone = match_two.groups()
        month = MONTH_TO_INT[month_as_string]
        tz = _fmt_date_tz(time_zone)
        return datetime.datetime(int(year), month, int(day),
                                 int(hour), int(minute), int(second), tzinfo=tz)
    return None


def _get_sub_charset(raw_headers):
    """Hardcode for some invalid mail-encoding."""
    for k, _ in raw_headers:
        if b'X-QQ' in k:
            return ['gbk']
    return []


def parse_headers(lines):
    headers = CaseInsensitiveDict()
    raw_headers = []
    unknown_value_headers = []

    lines_idx = 0
    line = lines[0]
    line_count = len(lines)

    while lines:
        if line in (b'', b'\r\n', b'\n'):
            break
        try:
            bname, bvalue = line.split(b':', 1)
        except:
            raise ValueError('Invalid header:' + str(line))
        bname = bname.strip(b' \t')
        bvalue = bvalue.lstrip()

        # next line
        lines_idx += 1
        if lines_idx <= line_count - 1:
            line = lines[lines_idx]
        else:
            bvalue = bvalue.strip()
            raw_headers.append((bname, bvalue))
            # Parse header.
            name = recursive_decode(bname, ('utf-8',))
            if name is not None:
                value = parse_header_value(bvalue, ('utf-8',))
                if value is not None:
                    headers[name] = value
                else:
                    unknown_value_headers.append((name, bvalue))
            else:
                raise ValueError('Invalid header name {}'.format(str(bname)))
            break

        # consume continuation lines
        continuation = line and line[0] in (32, 9)  # (' ', '\t')

        if continuation:
            bvalue = [bvalue]
            while continuation:
                bvalue.append(line.strip(b' \t'))
                # next line
                lines_idx += 1
                if lines_idx < line_count:
                    line = lines[lines_idx]
                    continuation = line and line[0] in (32, 9)  # (' ', '\t')
                else:
                    line = b''
                    break
            bvalue = b''.join(bvalue)

        bvalue = bvalue.strip()
        raw_headers.append((bname, bvalue))
        # Parse header.
        name = recursive_decode(bname, ('utf-8',))
        if name is not None:
            value = parse_header_value(bvalue, ('utf-8',))
            if value is not None:
                headers[name] = value
            else:
                unknown_value_headers.append((name, bvalue))
        else:
            raise ValueError('Invalid header {}'.format(str(bname)))

    # Parse Content-Type
    try:
        content_type, *extra_pair = headers['content-type'].split(';')
    except Exception as e:
        content_type = 'application/octet-stream'
        extra_pair = []
    main_type, sub_type = content_type.split('/')

    # Remove whitespace and get lower type.
    main_type, sub_type = main_type.replace(' ', '').lower(), sub_type.replace(' ', '').lower()

    # Get extra key values.
    extra_kv = CaseInsensitiveDict()
    for pair in extra_pair:
        if pair:
            try:
                _k, _v = pair.split('=', 1)
                _k = remove_line_feed_and_whitespace(_k)
                _v = remove_line_feed_and_whitespace(_v)
                extra_kv[_k] = _v
            except Exception as e:
                continue

    # Detect charsets
    charsets = []
    main_charset = extra_kv.get('charset')
    if main_charset is None:
        main_charset = 'utf-8'
    else:
        main_charset = main_charset.lower()
    charsets.append(main_charset)

    sub_charset = _get_sub_charset(raw_headers)  # type:list
    for charset in sub_charset:
        if charset not in charsets:
            charsets.append(charset)

    # Re-parses unknown headers
    for name, bvalue in unknown_value_headers:
        value = recursive_decode(bvalue, charsets)
        if value is not None:
            headers[name] = value
    # Parse Date and convert Date to DateTimeObject.
    if headers.get('date'):
        headers['date'] = fmt_date(headers['date'])
    return raw_headers, headers, lines_idx, main_type, sub_type, charsets, extra_kv


def multiple_part_decode(lines, boundary):
    content_text = []
    content_html = []
    attachments = []
    boundary = boundary.encode('ascii')

    # Split to parts.
    parts = []
    part_index = []
    for _idx, line in enumerate(lines):
        if boundary in line:
            part_index.append(_idx)
    if not part_index:
        raise ValueError('Can not find boundary on this mail.boundary{}'.format(boundary.decode("ascii")))
    else:
        _len = len(part_index)
        for idx_idx, idx in enumerate(part_index):
            if idx_idx + 1 <= _len - 1:
                parts.append(lines[idx + 1:part_index[idx_idx + 1]])

    for part in parts:
        parsed_part = parse(part)  # Recursive call
        if parsed_part['content_text']:
            content_text += parsed_part['content_text']
        if parsed_part['content_html']:
            content_html += parsed_part['content_html']
        if parsed_part['attachments']:
            attachments += parsed_part['attachments']

    return content_text, content_html, attachments


def parse_one_part_body(headers, body, main_type, sub_type, transfer_encoding, charsets, extra_kv):
    """Parse non-multiple-part body"""
    transfer_encoding = transfer_encoding.lower()
    content_text = None  # type:None or str
    content_html = None  # type:None or str
    attachment = None  # type:None or tuple

    content_disposition = headers.get('content-disposition')  # type:None or str

    # Is attachment.
    if content_disposition is not None and content_disposition.find('attachment') == 0:
        raw_attachment = _decode_one_part_body(body, transfer_encoding, charsets, _need_decode=False)
        if content_disposition:
            _extra_kv = CaseInsensitiveDict()
            content_disposition_extra_parts = content_disposition.split(';')
            for part in content_disposition_extra_parts:
                if '=' not in part:
                    continue
                try:
                    _k, _v = part.split('=', 1)
                    _k = remove_line_feed_and_whitespace(_k)
                    _v = remove_line_feed_and_whitespace(_v)
                    _extra_kv[_k] = _v
                except Exception as e:
                    continue
            filename = _extra_kv.get('filename')
            if filename is None and _extra_kv.get('filename*'):  # RFC5987 and ignore language tags
                match = FILENAME_PATTERN.fullmatch(_extra_kv.get('filename*'))
                if match:
                    _encoding, _language_tags, _name = match.groups()
                    filename = unquote(_name, _encoding)
        else:
            filename = None
        attachment_name = filename or extra_kv.get('name') or headers.get('subject') or 'Untitled'
        attachment = (attachment_name, raw_attachment)
    # Is text/plain
    elif (main_type, sub_type) == TYPE_TEXT_PLAIN:
        decoded_body = _decode_one_part_body(body, transfer_encoding, charsets)

        if decoded_body:
            content_text = decoded_body
    # Is text/html
    elif (main_type, sub_type) == TYPE_TEXT_HTML:
        decoded_body = _decode_one_part_body(body, transfer_encoding, charsets)
        if decoded_body:
            content_html = decoded_body
    else:  # All other type regard as attachment.
        raw_attachment = _decode_one_part_body(body, transfer_encoding, charsets, _need_decode=False)
        if content_disposition:
            _extra_kv = CaseInsensitiveDict()
            content_disposition_extra_parts = content_disposition.split(';')
            for part in content_disposition_extra_parts:
                if '=' not in part:
                    continue
                try:
                    _k, _v = part.split('=', 1)
                    _k = remove_line_feed_and_whitespace(_k)
                    _v = remove_line_feed_and_whitespace(_v)
                    _extra_kv[_k] = _v
                except Exception as e:
                    continue
            filename = _extra_kv.get('filename')
            if filename is None and _extra_kv.get('filename*'):  # RFC5987 and ignore language tags
                match = FILENAME_PATTERN.fullmatch(_extra_kv.get('filename*'))
                if match:
                    _encoding, _language_tags, _name = match.groups()
                    filename = unquote(_name, _encoding)
        else:
            filename = None
        attachment_name = filename or extra_kv.get('name') or headers.get('subject') or 'Untitled'
        attachment = (attachment_name, raw_attachment)

    return content_text, content_html, attachment


def _decode_one_part_body(lines, transfer_encoding, charsets, _need_decode=True):
    """Decode transfer-encoding then decode raw value to string."""
    if transfer_encoding == 'quoted-printable':
        decoded_bytes = decodestring(b'\r\n'.join(lines))
        if _need_decode:
            return recursive_decode(decoded_bytes, charsets)
        else:
            return b'\r\n'.join(lines)
    elif transfer_encoding == 'base64':
        decoded_bytes = b64decode(b''.join(lines))
        if _need_decode:
            return recursive_decode(decoded_bytes, charsets)
        else:
            return decoded_bytes
    elif transfer_encoding in ('binary', '8bit', '7bit'):
        if _need_decode:
            return recursive_decode(b'\r\n'.join(lines), charsets)
        else:
            return b'\r\n'.join(lines)
    else:
        raise ValueError('Invalid transfer-encoding {}'.format(transfer_encoding))


def parse(lines):
    """Decode a multiple-part or Non-multiple-part mail to ParsedMail(as CaseInsensitiveDict)."""
    content_text = []
    content_html = []
    attachments = []

    raw_headers, headers, eof_idx, main_type, sub_type, charsets, extra_kv = parse_headers(lines)

    body = lines[eof_idx + 1:]

    if main_type == TYPE_MULTIPART:  # Include recursive call
        boundary = extra_kv.get('boundary')
        if boundary is None:
            raise ValueError('Can not find boundary in multiple-part mail.')

        _content_text, _content_html, _attachment = multiple_part_decode(body, boundary)

        if _content_text:
            content_text += _content_text
        if _content_html:
            content_html += _content_html
        if _attachment:
            attachments += _attachment

    else:  # Recursive exit
        transfer_encoding = headers.get('content-transfer-encoding')

        if transfer_encoding is not None:
            transfer_encoding = transfer_encoding.lower()
        else:
            # Default transfer_encoding.
            transfer_encoding = '8bit'

        _content_text, _content_html, _attachment = parse_one_part_body(headers, body, main_type, sub_type,
                                                                        transfer_encoding, charsets, extra_kv)
        if _content_text:
            content_text.append(_content_text)
        if _content_html:
            content_html.append(_content_html)
        if _attachment:
            attachments.append(_attachment)

    mail = dict()
    mail['content_text'] = content_text
    mail['content_html'] = content_html
    
#     attachments = [i for i in attachments if i[0]!='Untitled'] 
    mail['attachments'] = attachments
    
#     mail['headers'] = headers
#     mail['raw_headers'] = raw_headers
#     mail['charsets'] = charsets

    mail['subject'] = headers.get('Subject')
    mail['date'] = headers.get('Date')
    mail['from'] = headers.get('From')
    mail['to'] = headers.get('To')
    mail['cc'] = headers.get('Cc')
    return mail


def parse_mail(lines, which):
    """A wrapper for parse mail."""
    mail = parse(lines)
    mail['id'] = which
    if mail['to'] is not None:
        mail['to'] = [i.strip() for i in str(mail['to']).split(',')]
    if mail['cc'] is not None:
        mail['cc'] = [i.strip()[1:-1] for i in str(mail['cc']).split(',')]
#     parsed_mail['raw'] = lines
    return mail