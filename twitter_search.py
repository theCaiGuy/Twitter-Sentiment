#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3

import twitter_setup
import requests
from datetime import datetime

#api = twitter.Api(consumer_key=twitter_setup.api_key, consumer_secret=twitter_setup.api_secret) #, access_token_key=twitter_setup.access_token, access_token_secret=twitter_setup.access_token_secret)
auth_tuple = (twitter_setup.api_key, twitter_setup.api_secret)
base_url = 'https://api.twitter.com/1.1/tweets/search/fullarchive/dev.json'

initial_queries = {'weinstein': 'q="harvey weinstein" since:2017-10-01 until:2017-10-6&src=typd',
					'spacey': 'q="kevin spacey" since:2017-10-27 until:2017-10-31&src=typd',
					'takei': 'q="george takei" since:2017-11-08 until:2017-11-10&src=typd',
					'franco': 'q="james franco" since:2018-01-09 until:2018-01-12&src=typd',
					'ansari': 'q="aziz ansari" since:2018-01-11 until:2018-01-14&src=typd',
					'franken': 'q="al franken" since: 2017-11-10 until:2017-11-17&src=typd',
					'kelly': 'q="r kelly" since:2019-01-01 until:2019-01-04&src=typd',
					'lee': 'q="stan lee" since:2018-01-07 until:2018-01-10&src=typd'
}


# Dict containing the names of perpetrators and the most recent search results for them on Twitter
results = {k: [] for k in initial_queries.keys()}

for (name, query) in initial_queries.items():
	results[name] = requests.get(base_url, params={'query': query}, auth=auth_tuple)

	# The following line will run properly once we get our Premium Search API account approved, but for now we substitute the above
	#results[name] = api.GetSearch(raw_query=query)

print(results)

