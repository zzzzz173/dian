import re
import time
import json
import requests
from bs4 import BeautifulSoup
# from pygments.styles.dracula import comment
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime
from itertools import islice
import pandas as pd
import numpy as np
import random
import os
import sys
import argparse


def get_bgm_url(page_num):
    try:
        url_list = []
        vote_num = []
        title_list = []
        base_url = "https://bangumi.tv/anime/browser?sort=rank&page="
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

        # 添加随机延迟
        time.sleep(random.uniform(1, 3))

        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        for i in range(1, page_num + 1):
            url = base_url + str(i)
            response = session.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # 提取动画链接和标题
            anime_links = soup.find_all('a', class_='l')
            for link in anime_links:
                href = link.get('href')
                if '/subject' in href:
                    url_list.append(href)
                    title_list.append(link.text.strip())

            # 提取评分人数
            rating_spans = soup.find_all('span', class_='tip_j')
            for span in rating_spans:
                number = re.findall(r'\d+', span.text)
                if number:
                    vote_num.append(int(number[0]))

        # 使用pandas来处理url、vote和title
        df = pd.DataFrame({
            'url': url_list,
            'vote_num': vote_num,
            'title': title_list
        })

        return df
    except Exception as e:
        print(f"获取URL列表时发生错误: {str(e)}")
        return pd.DataFrame()


def data_deal(stars, need_num, times):
    merged_times = np.concatenate(times)
    merged_stars = np.concatenate(stars).astype(int)

    # 将时间和星级转换为DataFrame并按时间排序
    df = pd.DataFrame({
        'time': merged_times,
        'stars': merged_stars
    })
    df['time'] = pd.to_datetime(df['time'], format="%Y-%m-%d %H:%M")
    df.sort_values(by='time', ascending=False, inplace=True)

    # 取前 need_num[0] 个星级值
    first_n_values = df['stars'].head(need_num[0])

    return first_n_values.mean()


def get_points(df):
    # 添加进度跟踪
    total = len(df)
    current = 0

    # 添加断点续传功能
    checkpoint_file = 'checkpoint.json'
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
    else:
        checkpoint = {'last_processed': 0}

    result = []

    kind = ['/collections?page=', '/doings?page=', '/on_hold?page=', '/dropped?page=']
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    for idx, row in df.iloc[checkpoint['last_processed']:].iterrows():
        try:
            current += 1
            print(f"处理进度: {current}/{total}")

            url = row['url']
            vote_num = row['vote_num']
            base_url = "https://bangumi.tv" + url + '/collections'
            response = session.get(base_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            num_temp = soup.find_all('small')
            need_num = []
            stars = [[] for _ in range(len(kind))]
            times = [[] for _ in range(len(kind))]

            for n in num_temp:
                number = re.findall(r'\d+', n.text)
                if number:
                    need_num.append(int(number[0]))

            need_num = need_num[1:5]
            need_num = np.array(need_num)
            need_num[0] = vote_num / 10
            comments_and_ratings = []

            for k, kind_content in enumerate(kind):
                page_num = 1
                success_num = 0
                max_page = get_max_page(url, kind_content, page_num, session, headers)
                print(f"max_page: {max_page}")

                while True:
                    base_url = "https://bangumi.tv" + url + kind_content + str(page_num)
                    response = session.get(base_url, headers=headers)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, 'html.parser')

                    stars_recent = soup.find_all('span', class_=re.compile(r'\bstarlight\b'))
                    # print(stars_recent)
                    times_recent = soup.find_all('p', class_=re.compile('info'))
                    user_container = soup.find_all('div', class_=re.compile('userContainer'))
                    text = []
                    position = []

                    for num, container in enumerate(user_container):
                        # 提取该 div 标签内的 HTML 字符串
                        container_html = str(container)

                        # 在 container_html 中查找目标文本
                        target_text = re.search(r'</p>(.*?)</div>', container_html, re.DOTALL)

                        if target_text.group(1).strip():
                            text.append(target_text.group(1).strip())
                            position.append(num)
                            stars_recent2 = container.find('span', class_=re.compile(r'\bstarlight\b'))
                            print(stars_recent2)
                            try:
                                number = re.findall(r'\d+', stars_recent2.get('class')[1])
                                comments_and_ratings.append((target_text.group(1).strip(), int(number[0])))

                            except:
                                pass
                            print(comments_and_ratings)

                    for j, s in enumerate(stars_recent):
                        number = re.findall(r'\d+', s.get('class')[1])  # 获取评分
                        if number:
                            print(j)
                            stars[k].append(number[0])
                            success_num += 1
                            times[k].append(times_recent[j].text)
                            print(f"已获取{success_num}个评分")
                            # if j in position:
                            #     comments_and_ratings.append((text[j], int(number[0])))
                            #     print(666)
                            #     print(comments_and_ratings)

                    print(comments_and_ratings)
                    print(f"第{page_num}页已获取{success_num}个评分")
                    page_num += 1
                    time.sleep(1)
                    if success_num >= need_num[k] or page_num > max_page:
                        need_num[k] = success_num
                        print(f"已获取{success_num}个评分")
                        break

            # 数据处理
            result.append(data_deal(stars, need_num, times))
            print(f"{row['title']}的评分人数为{vote_num}，评分为{result[-1]}")
            with open('comments_and_ratings.jsonl', 'a', encoding='utf-8') as f:
                # for item in comments_and_ratings:
                #     f.write(f"{item[0]}: {item[1]}\n")
                for item in comments_and_ratings:
                    record = {"text": item[0], "point": item[1]}
                    json.dump(record, f, ensure_ascii=False)
                    f.write('\n')

            # 保存检查点
            checkpoint['last_processed'] = idx
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)

        except Exception as e:
            print(f"处理 {row['title']} 时发生错误: {str(e)}")
            continue

    return result


def get_max_page(url, kind_content, page_num, session, headers):
    base_url = "https://bangumi.tv" + url + kind_content + str(page_num)
    response = session.get(base_url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')

    max_page_span = soup.find_all('span', class_='p_edge')
    max_page = 10
    if max_page_span:
        max_page_text = max_page_span[0].text
        max_page_number = re.findall(r'\d+', max_page_text)
        if max_page_number:
            max_page = int(max_page_number[-1])
    return max_page


if __name__ == '__main__':
    # 添加命令行参数支持
    parser = argparse.ArgumentParser(description='Bangumi动画数据爬虫')
    parser.add_argument('--pages', type=int, default=1, help='要爬取的页数')
    parser.add_argument('--output', type=str, default='anime_data.csv', help='输出文件名')
    args = parser.parse_args()

    try:
        # 获取数据
        df = get_bgm_url(args.pages)
        if df.empty:
            print("获取URL列表失败")
            sys.exit(1)

        print(f"成功获取 {len(df)} 条动画数据")

        # 保存原始数据
        df.to_csv('url_list.csv', index=False)

        # 获取评分
        result = get_points(df)

        # 合并结果并保存
        df['score'] = result
        df.to_csv(args.output, index=False)
        print(f"数据已保存到 {args.output}")

    except Exception as e:
        print(f"程序执行出错: {str(e)}")
