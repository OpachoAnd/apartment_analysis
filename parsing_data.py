from time import sleep

import fake_useragent
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from db.requests_db import DBCommands

USER_AGENT = fake_useragent.UserAgent()


def url_pages(url: str, count_pages: int):
    pages = []
    for i in range(1, count_pages):
        headers = {
            'User-Agent': USER_AGENT.random
        }
        session_requests = requests.Session()
        requests_content = session_requests.get(f'http://citystar.ru/detal.htm?d=43&nm=%CE%E1%FA%FF%E2%EB%E5%ED%E8%FF+%2D+%CF%F0%EE%E4%E0%EC+%EA%E2%E0%F0%F2%E8%F0%F3+%E2+%E3%2E+%CC%E0%E3%ED%E8%F2%EE%E3%EE%F0%F1%EA%E5&pN={str(i)}', headers=headers)
        pages.append(BeautifulSoup(requests_content.content))
        sleep(3)
    return pages


def download_apartments(url: str, count_pages: int):
    list_url_pages = url_pages(url, count_pages)

    general_list = []
    heading_list = []

    for i in tqdm(list_url_pages, 'Extracting_Pages'):
        if len(heading_list) == 0:
            print('зашли')
            heading_list = [t.text for t in i.findAll('td', class_='tht')]
            area = heading_list[10: 13]
            heading_list[4:4] = area
            heading_list.pop(7)
            del heading_list[12:]
            heading_list.insert(0, 'blanc')
        for string in i.findAll('tr', class_='tbb'):
            # q = string.findAll('td', class_='ttx')
            param_list = [tag.text for tag in string.findAll('td', class_='ttx')]
            general_list.append(param_list)

    df = pd.DataFrame(general_list, columns=heading_list)
    df.to_excel('apartments_new.xlsx')

    return df


if __name__ == "__main__":
    parse_df = download_apartments('url', count_pages=6)
    # df = pd.read_excel(r'apartments_3.xlsx')
    db_commands = DBCommands()

    DBCommands.drop_tables()
    DBCommands.create_tables()

    df_byte = parse_df.to_json().encode()
    # df_byte = df.to_json().encode()
    db_commands.add_df_to_db(df_byte)
