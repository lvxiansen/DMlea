import re
line = "<url>http://yule.sohu.com/yanqiang/p/12108527.html</url>"
content = "<content>哈哈 这是</content>"
line = re.sub('<url>|</url>','',line)
print(line)
content = re.sub('<content>|</content>', '', content)
print(content)
url_split = line.replace('http://', '').split('.')
print(url_split)
sohu_index = url_split.index('sohu')
print(sohu_index)