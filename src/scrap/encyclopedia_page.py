#!/usr/bin/env python
# -*- coding: utf-8 -*-


import time

import pandas as pd
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

from src.scrap.encyclopedia_item import EncyclopediaItem


BASENAME_URLS = [
    'https://www.dofus.com/fr/mmorpg/encyclopedie/ressources',
    'https://www.dofus.com/fr/mmorpg/encyclopedie/armes',
    'https://www.dofus.com/fr/mmorpg/encyclopedie/equipements',
    'https://www.dofus.com/fr/mmorpg/encyclopedie/familiers',
    'https://www.dofus.com/fr/mmorpg/encyclopedie/montures',
    'https://www.dofus.com/fr/mmorpg/encyclopedie/consommables',
    'https://www.dofus.com/fr/mmorpg/encyclopedie/objets-d-apparat',
    'https://www.dofus.com/fr/mmorpg/encyclopedie/compagnons',
    'https://www.dofus.com/fr/mmorpg/encyclopedie/idoles',
]

TIMEOUT = 3


class EncyclopediaScrap:
    def __init__(self):
        self.driver = webdriver.Firefox()

    def check_for_cookies_panel(self) -> bool:
        """Accept the cookies if the panel is in the page.
        Return wether the cookies have been accepted or not.
        """
        time.sleep(TIMEOUT)  # Wait for the page to fully load

        try:
            element = self.driver.find_element(By.CLASS_NAME, 'ak-block-cookies-infobox')
            buttons = element.find_elements(By.TAG_NAME, 'button')
            for b in buttons:
                if 'ACCEPTER' in b.text:
                    b.click()  # Accept cookies
                    return True
        except NoSuchElementException:
            return False

        raise RuntimeError('Cookies panel has been found but no there is no accept button.')

    def start(self, url: str):
        """Load a page and check for cookies.
        """
        self.driver.get(url)
        self.check_for_cookies_panel()

    def list_items(self, url: str) -> list[str]:
        """Get all items links in the page.
        """
        self.driver.get(url)
        element = self.driver.find_element(By.TAG_NAME, 'tbody')
        trs = element.find_elements(By.TAG_NAME, 'tr')
        items = [
            tr.find_element(By.TAG_NAME, 'a').get_attribute('href')
            for tr in trs
        ]
        return items

    def total_pages(self, url: str) -> int:
        """Detect the number of pages for the specified category pointed the url.
        """
        self.driver.get(url)
        element = self.driver.find_element(By.CLASS_NAME, 'ak-panel-footer')
        lis = element.find_elements(By.TAG_NAME, 'li')
        pages = [
            int(l.text)
            for l in lis
            if l.text.isdecimal()
        ]
        return max(pages)

    def category_name(self, category_url: str) -> str:
        self.driver.get(category_url)
        element = self.driver.find_element(By.CLASS_NAME, 'ak-title-container')
        element = element.find_element(By.TAG_NAME, 'h1')
        raw_html = element.get_attribute('innerHTML')
        name = raw_html.split('\n')[-2]
        return name.strip()

    def check_404(self, url: str) -> bool:
        self.driver.get(url)
        try:
            self.driver.find_element(By.CLASS_NAME, 'ak-404')
            return True
        except NoSuchElementException:
            return False

        raise RuntimeError

    def scrap_category(self, category_url: str):
        max_page_number = self.total_pages(category_url)
        category_name = self.category_name(category_url).lower()

        df = pd.DataFrame(
            columns=[
                'url',
                'name',
                'description',
                'type',
                'level',
                'illustration_url',
            ]
        )

        for page_number in tqdm(range(1, max_page_number + 1)):
            url = EncyclopediaScrap.get_page_number(category_url, page_number)
            items = self.list_items(url)
            for item_url in tqdm(items):
                time.sleep(TIMEOUT)
                if self.check_404(item_url):
                    continue

                scrap_item = EncyclopediaItem(self.driver, item_url)
                scrap_item.scrap()

                df = scrap_item.add_to_df(df)
                df.to_csv(f'data/{category_name}.csv', index=False)

    @staticmethod
    def get_page_number(basename_url: str, number: int) -> str:
        return f'{basename_url}?page={number}'

