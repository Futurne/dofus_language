#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By


class EncyclopediaItem:
    def __init__(self, driver: webdriver.Firefox, item_url: str):
        self.driver = driver
        self.item_url = item_url

    def scrap_name(self):
        element = self.driver.find_element(By.CLASS_NAME, 'ak-title-container')
        element = element.find_element(By.TAG_NAME, 'h1')
        raw_html = element.get_attribute('innerHTML')
        name = raw_html.split('\n')[-2]
        self.item_name = name.strip()

    def scrap_illustration(self):
        element = self.driver.find_element(By.CLASS_NAME, 'ak-encyclo-detail-illu')
        element = element.find_element(By.TAG_NAME, 'img')
        self.illustration_url = element.get_attribute('src')

    def scrap_type(self):
        element = self.driver.find_element(By.CLASS_NAME, 'ak-encyclo-detail-type')
        element = element.find_element(By.TAG_NAME, 'span')
        self.item_type = element.text

    def scrap_level(self):
        element = self.driver.find_element(By.CLASS_NAME, 'ak-encyclo-detail-level')
        self.item_level = element.text

    def scrap_description(self):
        element = self.driver.find_element(By.CLASS_NAME, 'ak-panel-content')
        element = element.find_element(By.CLASS_NAME, 'ak-panel-content')
        self.item_description = element.text

    def scrap(self):
        self.driver.get(self.item_url)

        self.scrap_name()
        self.scrap_illustration()
        self.scrap_type()
        self.scrap_level()
        self.scrap_description()

    def add_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        item_df = pd.DataFrame({
            'url': [self.item_url],
            'name': [self.item_name],
            'description': [self.item_description],
            'type': [self.item_type],
            'level': [self.item_level],
            'illustration_url': [self.illustration_url],
        })

        df = pd.concat([df, item_df])
        return df

