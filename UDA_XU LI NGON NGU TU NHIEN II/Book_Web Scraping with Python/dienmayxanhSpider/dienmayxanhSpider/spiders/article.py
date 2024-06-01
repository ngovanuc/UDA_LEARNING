'''To create a crawler, you will add a new file inside the spiders directory 
at wikiSpider/wikiSpider/spiders/article.py. In your newly created article.py 
file, write the following:'''

from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule


class ArticleSpider(CrawlSpider):
    name = 'article'
    allowed_domains = ['wikipedia.org']
    start_urls = ['https://en.wikipedia.org/wiki/Benevolent_dictator_for_life']
    rules = [Rule(LinkExtractor(allow=r'.*'), callback='parse_items', follow=True)]

    def parse_items(self, response):
        url = response.url
        title = response.css('h1::text').extract_first()
        text = response.xpath('//div[@id="mw-content-text"]//text()').extract_first()
        lastUpdated = lastUpdated.replace(
            'This page was last edited on ', ''
        )
        print('URL is: {}'.format(url))
        print('title is: {} '.format(title))
        print('text is: {}'.format(text))
        print('Last updated: {}'.format(lastUpdated))


# run this:
# scrapy runspider article.py
