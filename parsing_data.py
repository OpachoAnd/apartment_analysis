from time import sleep

import fake_useragent
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd

USER_AGENT = fake_useragent.UserAgent()


def url_pages(url: str, count_pages: int):
    pages = []
    for i in range(1, count_pages):
        headers = {
            'User-Agent': USER_AGENT.random
        }
        session_requests = requests.Session()
        # requests_content = session_requests.get(f'https://www.freepik.com/search?format=search&page={str(i)}&query=fabric%20care%20symbols&type=photo', headers=headers)
        requests_content = session_requests.get(f'http://citystar.ru/detal.htm?d=43&nm=%CE%E1%FA%FF%E2%EB%E5%ED%E8%FF+%2D+%CF%F0%EE%E4%E0%EC+%EA%E2%E0%F0%F2%E8%F0%F3+%E2+%E3%2E+%CC%E0%E3%ED%E8%F2%EE%E3%EE%F0%F1%EA%E5&pN={str(i)}', headers=headers)
        pages.append(BeautifulSoup(requests_content.content))
        # print(type(BeautifulSoup(requests_content.content)))
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
    # print(heading_list)
    #
    df.to_excel('apartments_3.xlsx')
    # for i in tqdm(list_url_pages, 'Extracting_Pages'):
    #     for string in i.findAll('tr', class_='tbb'):
    #         ttx = string.findAll('td', class_='ttx')


def download_images(url: str, count_pages: int):
    list_url_pages = url_pages(url, count_pages)
    j = 0
    for i in tqdm(list_url_pages, 'Extracting_Pages'):
        for img in i.findAll('img'):
            img_url = img.attrs.get("data-src")
            if img_url is not None:
                p = requests.get(img_url)
                out = open(f"C:/Users/opacho/Documents/dataset_CARE_LABEL/img_{str(j)}.jpg", "wb")
                out.write(p.content)
                out.close()
                j += 1


if __name__ == "__main__":
    download_apartments('url', count_pages=6)
    # download_images('url', count_pages=12)
