from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time 


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

        # wait for the login page to load
        self.wait.until(EC.visibility_of_element_located((By.ID, "m_login_email")))

        self.driver.find_element_by_id('m_login_email').send_keys(login)
        self.driver.find_element_by_xpath('//*[@id="login_form"]/ul/li[2]/div/input').send_keys(password)
        self.driver.find_element_by_xpath('//*[@id="login_form"]/ul/li[3]/input').click()

        # wait for the main page to load
        time.sleep(2)

    def get_friends(self):
        # navigate to "friends" page
        self.driver.get('https://mbasic.facebook.com/anish.saha.18/friends?')

        f = open('friend-links.txt','w')

        # continuous scroll until no more new friends loaded
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
                break  # no more friends loaded

        f.close()
        return 1


if __name__ == '__main__':
    crawler = FacebookCrawler(login='myemail@gmail.com', password='secretpassword')
    crawler.get_friends()
        
    