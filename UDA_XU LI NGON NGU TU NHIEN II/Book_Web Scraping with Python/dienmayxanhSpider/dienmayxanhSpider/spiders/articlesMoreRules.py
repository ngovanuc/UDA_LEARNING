'''Bằng cách sử dụng hai lớp Rule và LinkExtractor riêng biệt với một chức 
năng phân tích cú pháp duy nhất, bạn có thể tạo một trình thu thập thông 
tin thu thập dữ liệu Wikipedia, xác định tất cả các trang bài viết và các 
trang không phải bài viết đang gắn cờ (articlesMoreRules.py)'''

from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule


class ArticleSpider(CrawlSpider):
    name='Chapter03_WritingWebCrawler.ipynb'
    allowed_domain = ['wikipedia.org']
    start_urls = ['https://en.wikipedia.org/wiki/Benevolent_dictator_for_life']
    rules=[
        Rule(LinkExtractor(allow='^(/wiki/)((?!:).)*$'), callback='parse_items', follow=True, cb_kwargs={'is_article':True}),
        Rule(LinkExtractor(allow='.*'),callback='parse_items',cb_kwargs={'is_article':False})
    ]

    def parse_items(self, response, is_article):
        print(response.url)
        title = response.css('h1::text').extract_first()
        if is_article:
            url=response.url
            text=response.xpath('//div[@id="mw-content-text"]//text()').extract()
            lastUpdated=response.css('li#footer-info-lastmod''::text').extract_first()
            lastUpdated=lastUpdated.replace('This page was ''last edited on ','')
            print('Title is: {} '.format(title))
            print('title is: {} '.format(title))
            print('text is: {}'.format(text))
        else:
            print('This is not an article: {}'.format(title))