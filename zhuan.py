import os
import codecs
import shutil
# from chardet import detect
# f = open("D:\\SogouCS\\news.sohunews.370803.txt",'rb+')
# content = f.read()
# codeType = detect(content)['encoding']
# print(codeType)
# #GB2312


rootdir = 'D:\DMLea\SogouCS'
dirs = os.listdir(rootdir) #列出文件夹下所有的目录与文件
def convert(file, in_enc="GB2312", out_enc="UTF-8"):
    """
    该程序用于将目录下的文件从指定格式转换到指定格式，默认的是GBK转到utf-8
    :param file:    文件路径
    :param in_enc:  输入文件格式
    :param out_enc: 输出文件格式
    :return:
    """
    in_enc = in_enc.upper()
    out_enc = out_enc.upper()
    try:
        print("convert [ " + file.split('\\')[-1] + " ].....From " + in_enc + " --> " + out_enc )
        f = codecs.open(file, 'r', in_enc)
        new_content = f.read()
        codecs.open(file, 'w', out_enc).write(new_content)
    # print (f.read())
    except IOError as err:
        print("I/O error: {0}".format(err))
# 转换为utf-8格式

for i in range(0,len(dirs)):
    convert(os.path.join(rootdir,dirs[i]))