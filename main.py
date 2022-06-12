#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from src.scrap.almanax_scrap import ScrapAlmanax


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in {'init'}:
        print(f'Usage: {sys.argv[0]} [init]')
        sys.exit(0)

    if sys.argv[1] == 'init':
        print(f'Downloading data...')
        almanax = ScrapAlmanax()
        almanax.scrap()
        almanax.to_csv('data/data.csv')
        print(f'Done !')
        sys.exit(0)
