from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time 
import matplotlib.pyplot as plt

class FacebookCrawler:
    LOGIN_URL = 'https://mbasic.facebook.com/'

    def __init__(self, login, password):
        chrome_options = webdriver.ChromeOptions()
        prefs = {"profile.default_content_setting_values.notifications": 2}
        chrome_options.add_experimental_option("prefs", prefs)
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')

        self.driver = webdriver.Chrome(chrome_options=chrome_options)
        self.wait = WebDriverWait(self.driver, 10)

        self.login(login, password)

    def login(self, login, password):
        self.driver.get(self.LOGIN_URL)

        self.wait.until(EC.visibility_of_element_located((By.ID, "m_login_email")))

        self.driver.find_element_by_id('m_login_email').send_keys(login)
        self.driver.find_element_by_xpath('//*[@id="login_form"]/ul/li[2]/div/input').send_keys(password)
        self.driver.find_element_by_xpath('//*[@id="login_form"]/ul/li[3]/input').click()

        time.sleep(2)

    #go to friend profile/about page and check the relationship status
    def parse_friend_relation_data(self, link):
        self.driver.get(link.split('?')[0]+"/about")
        if 'Single' in self.driver.find_element_by_tag_name("body").text:
            if 'Male' in self.driver.find_element_by_tag_name("body").text:
                return 'Single,Male'
            elif 'Female' in self.driver.find_element_by_tag_name("body").text:
                return 'Single,Female'
        elif 'Married' in self.driver.find_element_by_tag_name("body").text:
            return 'Married'
        elif 'In a relationship' in self.driver.find_element_by_tag_name("body").text:
            return 'In a relationship'
        elif 'Engaged' in self.driver.find_element_by_tag_name("body").text:
            return 'Engaged'
        elif 'In a civil partnership' in self.driver.find_element_by_tag_name("body").text:
            return 'In a civil partnership'
        elif 'In a domestic partnership' in self.driver.find_element_by_tag_name("body").text:
            return 'In a domestic partnership'
        elif 'In an open relationship' in self.driver.find_element_by_tag_name("body").text:
            return 'In an open relationship'
        elif "It's complicated" in self.driver.find_element_by_tag_name("body").text:
            return "It's complicated"
        elif 'Separated' in self.driver.find_element_by_tag_name("body").text:
            return 'Separated'
        elif 'Divorced' in self.driver.find_element_by_tag_name("body").text:
            return 'Divorced'
        elif 'Widowed' in self.driver.find_element_by_tag_name("body").text:
            return 'Widowed'
        else:
            return 'NA'

    #form result dictionary from each friend in list
    def get_friends_data(self):
        f = open("friend-links.txt",'r')
        links = f.readlines()
        f.close()

        result = dict()
        for link in links:
            if link != '\n':
                result[link] = self.parse_friend_relation_data(link)

        f = open("friend-relation.txt",'w')
        for k in result.keys():
            f.write(k+':'+result[k])
            f.write('\n')
        f.close()
        return result

    #crawl profile and get all friends'profile links from friendlist
    def get_friends(self):
        self.driver.get('https://mbasic.facebook.com/anish.saha.18/friends?')

        f = open('friend-links.txt','w')

        while (self.driver.find_element_by_css_selector('#m_more_friends > a') is not None):

            for i in range(1,100):
                try:
                    e = self.driver.find_element_by_xpath('//*[@id="root"]/div[1]/div[2]/div['+str(i)+']/table/tbody/tr/td[2]/a')
                    f.write(e.get_attribute("href"))
                    f.write('\n')
                except :
                    break

            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            try:
                self.driver.find_element_by_css_selector('#m_more_friends > a').click()
            except :
                break 

        f.close()
        return 1

if __name__ == '__main__':
    crawler = FacebookCrawler(login='myemail@gmail.com', password='secretpassword')
    crawler.get_friends()
    crawler.get_friends_data()
    
    f = open('friend-relation.txt','r')
    rels = f.readlines()
    relation = set()
    for i in range(len(rels)):
        if i%2 != 0:
            relation.add(rels[i])

    rel_count = {rel:rels.count(rel) for rel in relation}
    plt.bar(list(rel_count.keys()), rel_count.values(), color='g')
    plt.show()

    rel_grp = {rel:list() for rel in relation}
    for i in range(0,len(rels),2):
        rel_grp[rels[i+1]].append(rels[i])

    print('Single - Females:')
    for ppl in rel_grp['Single,Female\n']:
        print(ppl.split('/')[3],":",ppl, end='')

    print('Single - Males:')
    for ppl in rel_grp['Single,Male\n']:
        print(ppl.split('/')[3],":",ppl, end='')