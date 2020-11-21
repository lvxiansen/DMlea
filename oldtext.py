import os
from chardet import detect
#2008年数据文件格式修改以及txt文件合并

#获取原始语料文件夹下文件列表
def listdir_get(path, list_name):
  for file in os.listdir(path):
    file_path = os.path.join(path, file)
    if os.path.isdir(file_path):
      listdir_get(file_path, list_name)
    else:
      list_name.append(file_path)

#修改文件编码为utf-8
def code_transfer(list_name):
  for fn in list_name:
    with open(fn, 'rb+') as fp:
      content = fp.read()
      print(fn, "：现在修改")
      codeType = detect(content)['encoding']
      content = content.decode(codeType, "ignore").encode("utf8")
      fp.seek(0)
      fp.write(content)
      print(fn, "：已修改为utf8编码")
    fp.close()

#合并各txt文件
def combine_txt(data_original_path, list_name, out_path):
  cnt = 0 #未正确读取的文件数

  for name in list_name:
    try:
      file = open(name, 'rb')
      fp = file.read().decode("utf8")
    except UnicodeDecodeError:
      cnt += 1
      print("Error:", name)
      file.close()
      continue
    print(name)
    corpus_old = open(out_path, "a+", encoding="utf8")
    corpus_old.write(fp)
    corpus_old.close()
    file.close()
  print("共：", cnt, "文件未正确读取")

#2008年数据文件格式修改以及txt文件合并

#原始语料路径
data_original_path = "./zuoye/"
out_path = "./corpus_old.txt"

#获取文件路径
list_name = []
listdir_get(data_original_path, list_name)

#修改编码
#code_transfer(list_name)

#合并各txt文件
combine_txt(data_original_path, list_name, out_path)

#类别统计
#cate_statistics("corpus_old.txt")