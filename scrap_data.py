#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

from src.scrap.almanax_scrap import ScrapAlmanax
from src.scrap.encyclopedia_scrap import BASENAME_URLS, EncyclopediaScrap


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Dofus datasets')

    parser.add_argument(
        '--almanax',
        action='store_true',
        help='Download from the almanax website'
    )

    for arg_name in BASENAME_URLS.keys():
        parser.add_argument(
            f'--{arg_name}',
            action='store_true',
            help=f'Download from the encyclopedia {arg_name} page'
        )

    args = parser.parse_args()

    # Create the 'data' directory if necessary
    if not os.path.isdir('data'):
        os.makedirs('data')

    if args.almanax:
        almanax = ScrapAlmanax()
        almanax.scrap()
        almanax.to_csv('data/almanax.csv')

    for category, url in BASENAME_URLS.items():
        if args.__dict__[category]:
            scrap = EncyclopediaScrap()
            scrap.start(url)
            scrap.scrap_category(url)
            scrap.driver.quit()

