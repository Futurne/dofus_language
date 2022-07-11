#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect data from an almanax page.
"""

import requests
from bs4 import BeautifulSoup


class AlmanaxPage:
    def __init__(self, url: str):
        self.url = url
        self.page = requests.get(url)
        self.soup = BeautifulSoup(self.page.text, 'html.parser')

    def boss_desc(self) -> str:
        desc = self.soup.find(id='almanax_boss_desc')  # div balise containing the description.
        desc = desc.text
        desc = desc.split('\n')[2:]  # Remove the title, get only the content description.
        desc = ' '.join(desc)
        desc = desc.strip()  # Remove starting and ending spaces.
        return desc

    def rubrikabrax(self) -> str:
        desc = self.soup.find(id='almanax_rubrikabrax')
        desc = desc.text
        desc = desc.split('\n')[2:]
        desc = ' '.join(desc)
        desc = desc.strip()
        return desc

    def meryde(self) -> str:
        desc = self.soup.find(id='almanax_meryde_effect')
        desc = desc.find('p')  # Find the first 'p' balise
        desc = desc.text
        return desc

