import os   #导入模块
path = './train_images/'    #设置创建后文件夹存放的位置
for i in range(10):    #这里创建10个文件夹
    # *定义一个变量判断文件是否存在,path指代路径,str(i)指代文件夹的名字
    isExists = os.path.exists(path+str(i))
    if not isExists:    #判断如果文件不存在,则创建
        os.makedirs(path+str(i))
        print("%s 目录创建成功"%i)
    else:
        print("%s 目录已经存在"%i)
        continue    #如果文件不存在,则继续上述操作,直到循环结束
