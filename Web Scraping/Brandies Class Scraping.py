# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 20:45:46 2020

@author: 14830
"""


import os
import re
from selenium import webdriver
import time
import pandas as pd
from bs4 import BeautifulSoup


driver_open = webdriver.Chrome("C:/Users/14830/Desktop/BUS 256A/chromedriver_85.exe")

url = 'http://registrar-prod.unet.brandeis.edu/registrar/schedule/classes/2020/Fall/100/all?independent_study=1'

driver_open.get(url)

html = driver_open.page_source

page = BeautifulSoup(html, 'lxml')
option_list = page.find('select', {'name': 'groupSeqNum'})\
                  .find_all('option')[1:]



def gen_url(dis_id):
    head = 'http://registrar-prod.unet.brandeis.edu/registrar/schedule/classes/2020/Fall/'
    tail = '/all?independent_study=1'
    return head + str(dis_id) + tail


mapping = []
o = option_list[1]
for o in option_list:
    mapping.append([o['value'], o.text.strip()])


all_course = []
#dis = mapping[0]
for dis in mapping:
    #print(dis)
    url = gen_url(dis[0])
    
    html = driver_open.page_source
    page = BeautifulSoup(html, 'lxml')
    #rows = page.find('table', {'id': 'classes-list'})
    #if rows is None:
    #    continue
    row_list = page.findAll('tr', {'class': ['rowfirst', 'row', 'rowodd']})
    r = row_list[0]

    for r in row_list:       
       #if r.text.strip() == '':           
       #    break      
        class_n = r.find_all('td')[0].text.strip()
        course_id = re.sub(' +', '      ', re.sub(r'\n[\s]*', ' ', r.find('a', {'class':'def'}).text.strip()).replace('\r', ''))
        name = r.find('strong').text.strip()
        tme = re.sub(r'\n[\s]*', ';   ', r.find_all('td')[3].text.strip().replace(u'\xa0', u''))
        enrll = r.find_all('td')[4].text.strip().replace('\n', '  ').replace('\r',' ')
        enrll_index = re.search('\d', enrll)
        enrll_status = enrll[0:enrll_index.start()]
        enrll_date = enrll[enrll_index.start():]
        consent_index = enrll_status.find('Consent')
        enrll_status_open = enrll_status[0:consent_index].strip()
        enrll_status_consent = enrll_status[consent_index:].strip()
        enrll_status = '    '.join([enrll_status_open, enrll_status_consent])
        enrll_date = re.sub(' +', ' ', enrll_date)         
        enrll = '    '.join([enrll_status, enrll_date])
        inst = ';  '.join(list(map(lambda x: x.text.strip(), r.findAll('td')[5].findAll('a', {'target': '_blank'}))))
        book = list(map(lambda x: x.text.strip(), r.findAll('a', {'target': '_blank'})))[-1]
        all_course.append([class_n, course_id, dis[1], name, tme, enrll, inst, book])


df = pd.DataFrame(all_course)
df.columns = ['Class Number', 'Course Number', 'Subjects', 'Course Title', 'Time and Location', 'Enrollment','Instructor(s)', 'View Books']


df.to_csv('JYZ_HW3 Schedule of Class Data Collection.csv', encoding = 'ansi')
#df.to_excel('Brandeis Schedule of Class Data Collection.xlsx', sheet_name = 'Schedule of Class')
  
