import time
import datetime

import requests
import pandas as pd

__all__ = ['weather']

city_dict = {}

def _get_tianqi_api(city):
    return requests.get('https://www.tianqiapi.com/api/?appid=14641533&appsecret=WY75ItCZ&version=v1&city='+city).json()

def weather(city, mode=0):
    assert isinstance(city, str), "`city` should be str."
    for i in city_dict.keys():
        if time.time()-city_dict[i]['time_now']>3600:
            city_dict.pop(i)
    
    if city in city_dict:
        a = city_dict[city]
    else:
        a = _get_tianqi_api(city)
        a['time_now'] = time.time()
        city_dict[city] = a
        
    if mode==0:
        name = ['week', 'wea', 'tem', 'tem1', 'tem2', 'humidity', 'visibility', 'pressure', 
                'win', 'win_speed', 'win_meter', 'sunrise', 'sunset', 'air', 'air_level', 'air_tips']
        name_zh = ['星期', '天气', '温度', '最高温度', '最低温度', '湿度', '能见度', '气压', '风向', '风力', '风速',
                   '日出', '日落', '空气质量指数', '空气质量等级', '出行建议']
        s = {j:[a['data'][0][j]] for j in name}
        s['city'] = [city]
        s = pd.DataFrame(s).reindex(columns=name)
        s.columns = name_zh
    elif mode==1:
        for i in a['data']:
            if i['date']==str(pd.to_datetime(datetime.datetime.now()))[:10]:
                break    
        s = (pd.DataFrame(i['hours'])
             .rename(columns={'hours':'分时', 'tem':'温度', 'wea':'天气', 'win':'风向', 'win_speed':'风力'})
             .drop(['wea_img'], axis=1))
    elif mode==2:
        for i in a['data']:
            if i['date']==str(pd.to_datetime(datetime.datetime.now()))[:10]:
                break
        s = {'指数':['空气质量指数', '紫外线指数', '运动指数', '血糖指数', '穿衣指数', '洗车指数', '空气污染扩散指数']}
        s['等级'] = [i['air_level']]+[j['level'] for j in i['index']]
        s['建议'] = [i['air_tips']]+[j['desc'] for j in i['index']]
        s = pd.DataFrame(s).reindex(columns=['指数', '等级', '建议'])
    elif mode==3:
        s = {}
        name = ['日期', '温度', '最高温度', '最低温度', '天气', '风向', '风力', '星期']
        name_en = ['date', 'tem', 'tem1', 'tem2', 'wea', 'win', 'win_speed', 'week']
        for i, j in zip(name, name_en):
            s[i] = [k[j] for k in a['data']]

        name = ['紫外线指数', '减肥指数', '血糖指数', '穿衣指数', '洗车指数', '空气污染扩散指数']
        for i, j in enumerate(name):
            s[j] = [k['index'][i]['level'] for k in a['data']]
        s = pd.DataFrame(s).drop(['减肥指数', '血糖指数'], axis=1)
    s['更新时间'] = a['update_time']
    return s