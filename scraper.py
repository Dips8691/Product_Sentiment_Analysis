class sentiment: 
    def __init__(self) -> None:
        pass
def scrapper():
        """
        run scrapper()
        scrapes amazon review pages copies comments, deposits information in data.csv
        on prompt inputs= url , num_reviews desired
        note: 
        url input: input url like
        "https://www.amazon.com/product-reviews/B08MQZXN1X/ref=cm_cr_arp_d_viewopt_rvwer? ie=UTF8&filterByStar=all_stars&reviewerType=avp_only_reviews&pageNumber=1"
        has to finish in page number, and number of reviews""" 
        
        from selectorlib import Extractor
        import requests 
        import json 
        from time import sleep
        import csv
        from dateutil import parser as dateparser
        import pandas as pd 

        # Create an Extractor by reading from the YAML file
        e = Extractor.from_yaml_file('selectors.yml')

        url=input("input url: ")
        num_reviews= int(input("how many reviews: "))

        print (num_reviews)

        urls=[]
        urls.append(url)
        for i in range(1,((num_reviews // 9)-1 )):
            urls.append(url.replace(url[-1],str(i)))

        with open("urls.txt",'w') as f:
            for line in urls:
                f.write(line)
                f.write('\n')

        def scrape(url):    
            headers = {
                'authority': 'www.amazon.com',
                'pragma': 'no-cache',
                'cache-control': 'no-cache',
                'dnt': '1',
                'upgrade-insecure-requests': '1',
                'user-agent': 'Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36',
                'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                'sec-fetch-site': 'none',
                'sec-fetch-mode': 'navigate',
                'sec-fetch-dest': 'document',
                'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
            }

            # Download the page using requests
            print("Downloading %s"%url)
            r = requests.get(url, headers=headers)
            # Simple check to check if page was blocked (Usually 503)
            if r.status_code > 500:
                if "To discuss automated access to Amazon data please contact" in r.text:
                    print("Page %s was blocked by Amazon. Please try using better proxies\n"%url)
                else:
                    print("Page %s must have been blocked by Amazon as the status code was %d"%(url,r.status_code))
                return None
            # Pass the HTML of the page and create 
            return e.extract(r.text)

        # product_data = []
        with open("urls.txt",'r') as urllist, open('data/data.csv','w') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=["title","content","date","variant","images","verified","author","rating","product","url"],quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for url in urllist.readlines():
                data = scrape(url) 
                if data:
                    for r in data['reviews']:
                        r["product"] = data["product_title"]
                        r['url'] = url
                        #if 'verified' in r:
                         #   if 'Verified Purchase' in r['verified']:
                         #       r['verified'] = 'Yes'
                          #  else:
                           #     r['verified'] = 'Yes'
                        r['rating'] = r['rating'].split(' out of')[0]
                        date_posted = r['date'].split('on ')[-1]
                        if r['images']:
                            r['images'] = "\n".join(r['images'])
                        r['date'] = dateparser.parse(date_posted).strftime('%d %b %Y')
                        writer.writerow(r)
                    # sleep(5)
