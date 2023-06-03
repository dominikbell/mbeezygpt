"""
A web scraper that collects lyrics from all the MoneyBoy songs and dumps them into one file
"""

from lxml import html
import requests
from tqdm import tqdm
from time import sleep


def main():
    file_name = "mb_input.txt"
    with open(file_name, 'w') as file:
        file.write('')

    mb_url = "https://www.azlyrics.com/m/moneyboy.html"

    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.5",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/113.0"
    }

    urls = get_all_song_urls(mb_url, headers)

    for url in tqdm(urls):
        write_single_song(url, file_name, headers)

        # sleep to not get blocked
        sleep(10)


def get_all_song_urls(artist_url, headers):
    """ Given the url to an artist get the urls to all of his linked songs

    Parameters
    ----------
    artist_url : str
        url to the artist on azlyrics.com
    """
    base_url = "https://www.azlyrics.com"
    url_list = []

    res = requests.get(artist_url, headers=headers)

    res.encoding = res.apparent_encoding

    tree = html.fromstring(res.text)

    songs = tree.xpath('//div[@class="listalbum-item"]/a')

    for song in songs:
        url_list.append(base_url + song.items()[0][1])

    return url_list


def write_single_song(song_url, file_name, headers):
    """ Writes the lyrics from the given url to the given file.

    Parameters
    ----------
    song_url : str
        url to the song as a string

    file_name : str
        name of the file that the song_text should be appended to
    """
    res = requests.get(song_url, headers=headers)

    res.encoding = res.apparent_encoding

    raw_text = res.text.split('<div>')[1].split('</div>')[0]
    raw_text = raw_text[135:]
    raw_text = raw_text.replace('<br>', '')

    with open(file_name, 'a') as file:
        file.write(raw_text)


if __name__ == '__main__':
    main()
