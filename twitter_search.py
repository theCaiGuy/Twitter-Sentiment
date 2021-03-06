#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import numpy as np
import searchtweets as st
import emoji

# if strings are unicode, just use give_emoji_free_text(text.encode('utf8'))
def give_emoji_free_text(text):
    allchars = [str for str in text.decode('utf-8')]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.decode('utf-8').split() if not any(i in str for i in emoji_list)])
    return clean_text

premium_search_args = st.load_credentials(filename='./twitter_keys.yaml', env_overwrite=False)

initial_queries = {'ansari': ['aziz ansari', '2018-01-11', '2018-01-14'],
					'franken': ['al franken', '2017-11-10', '2017-11-17'],
					'kelly': ['r kelly', '2019-01-01', '2019-01-04'],
					'lee': ['stan lee', '2018-01-07', '2018-01-10']}

second_queries = {'weinstein': ['harvey weinstein', '2017-09-25', '2017-10-06'],
					'spacey': ['kevin spacey', '2017-10-27', '2017-10-31'],
					'takei': ['george takei', '2017-11-08', '2017-11-10'],
					'franco': ['james franco', '2018-01-09', '2018-01-12']}


for (name, query) in second_queries.items():
	rule = st.gen_rule_payload(query[0], results_per_call=100)
	tweets = st.collect_results(rule, max_results=600, result_stream_args=premium_search_args)
	tweets = [give_emoji_free_text(tweet.all_text.encode('utf8')) for tweet in tweets]
	
	# save tweets to a file that can be read into a numpy array of JSON objects later
	np.savetxt("./data/testdata/%s_new.txt" % name, np.array(tweets), fmt="%s")
