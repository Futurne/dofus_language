from selenium import webdriver
from selenium.webdriver.common.by import By

from src.scrap.encyclopedia_page import EncyclopediaScrap
from src.scrap.encyclopedia_item import EncyclopediaItem

if __name__ == '__main__':
    url = 'https://www.dofus.com/fr/mmorpg/encyclopedie/ressources'

    scrap = EncyclopediaScrap()
    scrap.start(url)

    scrap.scrap_category(url)

    scrap.driver.quit()
