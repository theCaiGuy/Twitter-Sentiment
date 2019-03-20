import emoji
import numpy as np
import datetime

# if strings are unicode, just use give_emoji_free_text(text.encode('utf8'))
def give_emoji_free_text(text):
    allchars = [str for str in text.decode('utf-8')]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.decode('utf-8').split() if not any(i in str for i in emoji_list)])
    return clean_text

with open('./data/testdata/lee.txt', 'r') as file:
	for line in file:
		print(datetime.datetime.strptime(line[:21], '%Y-%m-%d %H:%M:%S'))