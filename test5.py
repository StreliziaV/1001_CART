#由于我们做的是一个随机森林，有22棵树，所以递归次数大约在（2**7）*22次，所以需要增加递归次数限制
import sys
sys.setrecursionlimit(10000)

#以下是随机森林需要用到的数据集
photo = []#
commu = []#
shop = []#
family = []#
busi = []#
tool = []#
medical = []#
travel = []#
health = []#
game = []#
book = []#
video = []#
social = []#
pro = []#
fin = []#
sports = []#
style = []#
library = []#
maps = []#
date = []#
comic = []#
event = []#
house = []#
#用一下两个函数打开文件，并对文件里的内容按照category一栏进行划分，将每一行的数据变成list的格式，里面是string或者float,
#并将每一行放入对应的category里面，（格式为list of lists）
def trans_data(sett = family):#此函数功能为进一步规范数据格式，将“NA”等类型转化为平均数，并将格式为string的数字转化为float
    sum_r = 0
    lit = [2,3,4,6,9,10,11] #需要转化的数据所在的column
    for row in sett:
        if row[1] != "NA":
            sum_r = sum_r + eval(row[1])
    if len(sett) != 0:
        ave = sum_r/len(sett)
    else:
        ave = 0
    for row in sett:
        if row[1] == "NA":
            row[1] = ave
        else:
            x = eval(row[1])
            row[1] = x
    for i in lit:
        for row in sett:
            b = 1
            try:
                if not row[i].startswith("V"):
                    x = eval(row[i])
                    row[i] = x
            except:
                b += 1
    for row in sett:  #对于转化不成功的数据组，进行抛弃，每个分组大概会有五到六个不成功数据，具体原因不知道
        if isinstance(row[11],str):
            if not row[11].startswith("V"):
                
                sett.remove(row)
    for row in sett:
        if row[1] > 6:
            sett.remove(row)

def datasplit(fhand = open('train.csv','r'),amount = 7589):  #对数据进行分组，以category为依据，并对数据组中数据进行初步规范
    for i in range(amount):
        try:
            m = fhand.readline()
            l = m.split(',')
            
            #对数据进行格式上的修正
            for i in range(len(l)):
                l[i] = l[i].strip('"')
                if i == len(l) - 1:
                    l[i] = l[i].rstrip('"\n')
            for i in range(4):
                if (l[6] == "000") or (l[6] == "000+") or (l[6] == "00+") or (l[6] == "0+"):
                    l[5] = l[5] + l[6]
                    l.remove(l[6])
            l[5] = l[5].strip("+")
            l[4] = l[4].strip("M")
            l[4] = l[4].strip("k")
            l[7] = l[7].strip("$")
            if len(l) == 14:
                l.remove(l[10])
            ver1 = l[11].split('.') #对版本的表示进行调整，1.0.1 --> 101
            ver2 = ""
            for i in ver1:
                ver2 = ver2 + i
            l[11] = ver2
            if not l[12].startswith("V"):
                l[12] = l[12].strip("and up")
                ver1 = l[12].split('.') #对安卓版本的表示进行调整，1.0.1 --> 101
                ver2 = ""
                for i in ver1:
                    ver2 = ver2 + i
                l[12] = ver2

            if i != 0:
                x = l[1]
                l.remove(x)
                if x == "COMICS":                
                    comic.append(l)
                elif x == "EVENTS":               
                    event.append(l)
                elif x == "HOUSE_AND_HOME":                
                    house.append(l)
                elif x == "MAPS_AND_NAVIGATION":                
                    maps.append(l)
                elif x == "NEWS_AND_MAGAZINES":                
                    news.append(l)
                elif x == "WEATHER":                
                    weather.append(l)
                elif x == "ENTERTAINMENT":
                    enter.append(l)
                elif x == "LIFESTYLE":
                    style.append(l)
                elif x == "PERSONALIZATION":
                    person.append(l)
                elif x == "EDUCATION":
                    education.append(l)
                elif x == "FINANCE":
                    finance.append(l)
                elif x == "VIDEO_PLAYERS":
                    video.append(l)
                elif x == "BOOKS_AND_REFERENCE":
                    book.append(l)
                elif x == "HEALTH_AND_FITNESS":
                    health.append(l)
                elif x == "TRAVEL_AND_LOCAL":
                    travel.append(l)
                elif x == "MEDICAL":
                    medical.append(l)
                elif x == "TOOLS":
                    tool.append(l)
                elif x == "BUSINESS":
                    busi.append(l)
                elif x == "PARENTING":
                    parent.append(l)
                elif x == "SPORTS":
                    sports.append(l)
                elif x == "PRODUCTIVITY":
                    productivity.append(l)
                elif x == "GAME":
                    game.append(l)
                elif x == "DATING":
                    date.append(l)
                elif x == "SOCIAL":
                    social.append(l)
                elif x == "FAMILY":
                    family.append(l)
                elif x == "SHOPPING":
                    shop.append(l)
                elif x == "COMMUNICATION":
                    commu.append(l)
                elif x == "PHOTOGRAPHY":
                    photo.append(l)
                elif x == "LIBRARIES_AND_DEMO":
                    library.append(l)
                m = None
        except:
            m = None


#在将数据整理好之后，开始找最佳gini值
#以下函数是对非连续性变量关于rating的gini值的计算与选择,
# 非连续变量是指变量的值较少，一般在两三个，三四个左右，对每个值进行两两分组，并计算每个分组方式关于rating的gini值
#取最小gini值，以及该gini值对应的变量等各类特征
def setsplit(rows,col_index,splitpoint_value):#拆分子节点 
    try:
        split = None
        judge_flag_sptype = None
        if isinstance(splitpoint_value,int):
            split = lambda row:int(row[col_index])>=splitpoint_value
            judge_flag_sptype = 'set1 >= split point'
        elif isinstance(splitpoint_value,float):#判断标准为数值类型时
            split = lambda row:float(row[col_index])>=splitpoint_value
            judge_flag_sptype = 'set1 >= split point'
        elif isinstance(splitpoint_value,list):#判断标准为列表类型时
            split = lambda row:row[col_index] in splitpoint_value
            judge_flag_sptype = 'set1 in the list'
        else:#判断标准为字符串等其他类型时
            split = lambda row:row[col_index]==splitpoint_value
            judge_flag_sptype = 'set1 == split point'

        chiset1 = [row for row in rows if split(row)]
        chiset2 = [row for row in rows if not split(row)]
    except:
        chiset1,chiset2,judge_flag_sptype = [],[],0
    return chiset1,chiset2,judge_flag_sptype

def cal_Gini(dataset):#计算非连续变量的基尼指数
    total = len(dataset)#总数
    samcount = {}#各样本数
    for row in dataset:
        rating = row[2]
        if float(rating) >= 4.5:
            rating = True
        else:
            rating = False
        if rating not in samcount:
            samcount[rating] = 0#创建
        else:
            samcount[rating] += 1#计数
    sum_pp = 0
    for k in samcount:
        pk = int(samcount[k])/total
        sum_pp += pk**2
    gini = 1- sum_pp
    return gini

def choosebestsplitpoint(dataset,col):#选择最佳的切分点 输入为训练数据集 以及 作为切分标准的列（列表形式）

    data_under_same_label = list(col)
    try:
        bestsplitpoint = data_under_same_label[1]
        col_index = dataset[1].index(col[1])
    except:
        bestsplitpoint = 0
        col_index = 0
    best_gini = 100.0
    datalist = []#用于记录出现的数据种类
    opt_set1, opt_set2 = [], []#最优切分时产生的子集
    judge_flag = []#判断信息
    for data in data_under_same_label:
        if data not in datalist:
            datalist.append(data)
        set1,set2,judge_flag_sptype = setsplit(dataset,col_index,bestsplitpoint)
        gini_set1 = cal_Gini(set1)
        gini_set2 = cal_Gini(set2)
        gini_index = len(set1)/len(data_under_same_label)*gini_set1+len(set2)/len(data_under_same_label)*gini_set2
        if gini_index < best_gini:
            bestsplitpoint = data
            best_gini = gini_index
            opt_set1, opt_set2 = set1, set2
            optset_flag = judge_flag_sptype
    if len(datalist) > 2:#当出现过的切分点的值的种类大于2，则将切分点值尝试进行组合并再次进行gini计算和比较
        i = 0
        j = 0
        k = 0
        n = 0
        combined_split_point = []#组合切分点
        while k < len(datalist):#k以控制组合切分点中切分点值的数量
            while i < len(datalist):#i，j用以选取切分点值加入组合切分点值
                j = 0
                while j < len(datalist):
                    n = 1
                    combined_split_point = [datalist[i], datalist[j]]
                    while n <= k and j+n < len(datalist)-1:
                        combined_split_point.append(datalist[j+n])
                        n += 1
                    set1,set2,judge_flag_sptype = setsplit(dataset,col_index,combined_split_point)
                    gini_set1 = cal_Gini(set1)
                    gini_set2 = cal_Gini(set2)
                    gini_index = len(set1)/len(data_under_same_label)*gini_set1+len(set2)/len(data_under_same_label)*gini_set2
                    if gini_index < best_gini:
                        bestsplitpoint = data
                        best_gini = gini_index
                        opt_set1, opt_set2 = set1, set2
                        optset_flag = judge_flag_sptype
                    j += 1
                i += 1
            k += 1
    #计算opt_set1 and opt_set2 的rating的平均值
    sumofrating_set1 = 0#评价值总和
    sumofrating_set2 = 0
    avrofrating_set1 = 0#平均评价值
    avrofrating_set2 = 0
    truecount_set1 = 0#4.5评价值计数
    truecount_set2 = 0
    totalcount_set1 = 0#总计数
    totalcount_set2 = 0
    
    for row in opt_set1:
        sumofrating_set1 += row[1]
        totalcount_set1 += 1
        if row[1] >= 4.5:
            truecount_set1 += 1
    if totalcount_set1 != 0:
        avrofrating_set1 = sumofrating_set1/totalcount_set1
        trueprop_set1 = truecount_set1/totalcount_set1
    else:
        avrofrating_set1 = 0
        trueprop_set1 = 0

    for row in opt_set2:
        sumofrating_set2 += row[1]
        totalcount_set2 += 1
        if row[1] >= 4.5:
            truecount_set2 += 1
    if totalcount_set2 != 0:
        avrofrating_set2 = sumofrating_set2/totalcount_set2
        trueprop_set2 = truecount_set2/totalcount_set2
    else:
        avrofrating_set2 = 0
        trueprop_set2 = 0

    judge_flag.append(optset_flag)#以列表形式存储以下信息
    judge_flag.append(opt_set1)#[切分点形式 子集1 子集2 平均值1 占比1 平均值2 占比2]
    judge_flag.append(opt_set2)
    judge_flag.append(avrofrating_set1)
    judge_flag.append(trueprop_set1)
    judge_flag.append(avrofrating_set2)
    judge_flag.append(trueprop_set2)

    return bestsplitpoint, best_gini, judge_flag

      #选择最优的划分标准
def choosebestfeature(dataset, indexlist):#关于dataset的格式：输入的dataset应该是list of lists（如：[[1,1,1,1],[2,2,2,2]]
    try:
        rows = list(dataset)
        
        bestgini = 100.0
        bestpoint = None
        featureslist = []
        
        fea_type = []
        fea_content_rating =[]
        fea_lastupdate =[]

        for row in rows:
            if 5 in indexlist:
                fea_type.append(row[5])
            if 7 in indexlist:
                fea_content_rating.append(row[7])
            if 9 in indexlist:
                fea_lastupdate.append(row[9])
        
        #非连续变量部分
        feasp, feagini, flag = choosebestsplitpoint(dataset, fea_type)
        fea_type = [feasp,feagini,flag,5]#feature的格式为[最优切分点，该feature的最小gini指数, 判断信息]
        feasp, feagini,flag = choosebestsplitpoint(dataset,fea_content_rating)
        fea_content_rating = [feasp,feagini, flag,7]
        feasp, feagini,flag = choosebestsplitpoint(dataset, fea_lastupdate)
        fea_lastupdate = [feasp,feagini,flag,9]
        #比较feature的gini部分
        featureslist.append(fea_type)
        featureslist.append(fea_content_rating)
        featureslist.append(fea_lastupdate)
        #开始比较gini
        for feature in featureslist:
            if feature[1] < bestgini:
                bestgini = feature[1]
                bestpoint = feature[0]
                bestfeature_index = feature[3]
                bestflag = feature[2]#[]
    except:
        bestfeature_index,bestpoint,bestgini,bestflag = 0,0,10000,[0,0,0,0,0,0,0]
    return bestfeature_index,bestpoint,bestgini,bestflag
    
#以下四个函数用于计算连续性变量关于rating的gini值
#连续性变量指数量较多，一般为integer或float,如文件train中的Install,size，reviews等
#取每一个变量值，所有变量值按照大于或小于该变量值分类，每一种分类方法都需要算一次gini值
#取最佳gini，以及对应变量，该变量的最佳切分点等特征
def continuous_varie(sett,col = 2): #在对连续变量计算gini时，先以"Varies with device"为依据，进行是与否的分类，以此来算gini值
    
    gini =  0                       #并与由连续性变量算出来的最小gini值进行比较
    var,num = [],[] 
    for row_1 in sett:
        if isinstance(row_1[col],str):
            var.append(row_1[1])
        else:
            num.append(row_1[1])
    big1,big2,sma1,sma2 = [],[],[],[]
    for i in var:
        if i <= 4.5:
            sma1.append(i)
        else:
            big1.append(i)
    for i in num:
        if i <= 4.5:
            sma2.append(i)
        else:
            big2.append(i)
    b1,b2,s1,s2 = len(big1),len(big2),len(sma1),len(sma2)
    
    p1,p2 = b1/(b1+s1),b2/(b2+s2)
    gini = (var/(var+num))*(2*p1*(1-p1)) + (num/(num+var))*(2*p2*(1-p2))
    return gini

def trans_vari(sett,col = 2): #此函数的作用为，在对连续性变量计算gini值时，将"Varies with device"转换为该组数据的平均值
    sum1 = 0
    for row in sett:
        if not isinstance(row[col],str):
            sum1 += row[col]
    ave = sum1/(len(sett))
    for row in sett:
        if isinstance(row[col],str):
            row[col] = ave

def continuous_gini(sett,col = 2): #输入的sett为之前分类好的某一个数据集，col是连续性变量所在的列
    ginis = [] #gini指数的列表，对每一个连续变量算出其gini值后，加入ginis列表，并以此判断第几个最小，应用最小的gini值
    gini_judge = None
    propl = [None]
    try:
        gini_v = continuous_varie(sett,col) #算出通过Varies with devices分类的gini值，并与其他gini值进行比较
    except:
        gini_v = 100
    trans_vari(sett,col) 
    for row_1 in sett:
        nod = row_1[col]
        large = []
        small = []
        for row_2 in sett:
            if row_2[col] < nod :
                small.append(row_2[1])
            else:
                large.append(row_2[1])
        lar,sma = len(large),len(small)
        r_large1,r_large2= [],[]
        r_small1,r_small2 = [],[]
        for rate in large:
            if rate <= 4.5:
                r_small1.append(rate)
            else:
                r_large1.append(rate)
        for rate in small:
            if rate <= 4.5:
                r_small2.append(rate)
            else:
                r_large2.append(rate)
        r_lar1,r_lar2 = len(r_large1),len(r_large2)
        r_sma1,r_sma2 = len(r_small1),len(r_small2)
        if r_lar1+r_sma1 != 0:
            p1 = r_lar1/(r_lar1+r_sma1)
        else:
            p1 = 0
        if r_lar2+r_sma2 != 0:
            p2 = r_lar2/(r_lar2+r_sma2)
        else:
            p2 = 0
        gini = (lar/(lar+sma))*(2*p1*(1-p1)) + (sma/(lar+sma))*(2*p2*(1-p2))
        ginis.append(gini)
        
    ginis_1 = sorted(ginis) #这里开始，是找出最小的gini值在数据集中对应的变量值：result
    result = 0
    for i in range(len(ginis)): 
        if ginis_1[0] == ginis[i]:
            result = i
    if gini_v < ginis_1[0]:
        gini_judge = gini_v
        for i in range(len(ginis)): 
            if gini_v == ginis[i]:
                result = i
        posi,nega = [],[]
        posi1,nega1 = [],[]
        sum1,sum2 = 0,0
        sum3,sum4 = 0,0
        p1,p2,n1,n2 = [],[],[],[]
        for row in sett:
            if row[col] == "Varies with device":
                posi1.append(row)
                posi.append(row[1])
            else:
                nega1.append(row)
                nega.append(row[1])
        for i in posi:
            if i > 4.5:
                sum1 = sum1 + i
                p1.append(i)
            else:
                sum2 = sum2 + i
                p2.append(i)
        for i in nega:
            if i > 4.5:
                sum3 = sum3 + i
                n1.append(i)
            else:
                n2.append(i)
                sum4 += i
        avep,aven = (sum1+sum2)/(len(posi)),(sum3+sum4)/(len(nega))   #两边变量rating的平均值
        prop,pron = (len(p1))/(len(p1)+len(p2)),(len(n1))/(len(n1)+len(n2)) #两边变量rating大于4.5的占比
        propl.append(posi1)
        propl.append(nega1)
        propl.append(avep)
        propl.append(prop)
        propl.append(aven)
        propl.append(pron)

        result = "Varies with device"

    else:
        gini_judge = ginis_1[0]
        big,lit = [],[]
        big1,lit1 = [],[]
        sum1,sum2 = 0,0
        sum3,sum4 = 0,0
        b1,b2,l1,l2 = [],[],[],[]
        for row in sett:
            if row[col] >= sett[result][col] :
                big.append(row[1])
                big1.append(row)
            else:
                lit1.append(row)
                lit.append(row[1])
        for i in big:
            if i > 4.5:
                b1.append(i)
                sum1 = sum1 + i
            else:
                b2.append(i)
                sum2 += i
        for i in lit:
            if i > 4.5:
                l1.append(i)
                sum3 += i
            else:
                l2.append(i)
                sum4 += i
        aveb,avel = (sum1+sum2)/(len(big)),(sum3+sum4)/(len(lit))
        prob,prol = (len(b1))/(len(b1)+len(b2)),(len(l1))/(len(l1)+len(l2))
        propl.append(big1)
        propl.append(lit1)
        propl.append(aveb)
        propl.append(prob)
        propl.append(avel)
        propl.append(prol)
        
        
        result = sett[result][col]
        
    return gini_judge,result,propl#col的index 切分值 gini 列表[平均值 占比 平均值 占比]

def continuous_feature(sett,conti = [2,3,4,11]):
    if len(conti) != 0:
        g,r,p = [],[],[]
        gini_judge, result, propl, col = 0, 0, None, 0, 

        for i in conti:
            
            x,y,z = continuous_gini(sett,i)
            g.append(x)
            r.append(y)
            p.append(z)


        ginis1 = sorted(g)
        for i in range(len(conti)):
            if g[i] == ginis1[0]:
                gini_judge, result, propl, col = g[i], r[i], p[i], conti[i]
    else:
        gini_judge, result, propl, col = 1000,0,0,0

    return col, result, gini_judge, propl   

#通过multigini_select在连续性与非连续性中，选出gini值最小的，以及该gini值的对应特征
def multigini_select(sett,conti,dis):
    if len(dis) == 0:
        col, result, gini_judge, propl = continuous_feature(sett,conti)
        return col, result, gini_judge, propl
    elif len(conti) == 0:
        bestfeature_index,bestpoint,bestgini,bestflag = choosebestfeature(sett,dis)
        return bestfeature_index,bestpoint,bestgini,bestflag
    else:
        col, result, gini_judge, propl = continuous_feature(sett,conti)
        bestfeature_index,bestpoint,bestgini,bestflag = choosebestfeature(sett,dis)
        if gini_judge < bestgini:
            return col, result, gini_judge, propl
        else:
            return bestfeature_index,bestpoint,bestgini,bestflag

#以下开始实现随机森林

#这个class用于定义决策树上的node
class decisionnode:
    def __init__(self, col_index = None, splitpoint_value = None, flag = None, leftchiset = None, rightchiset = None, leftchinode = None, rightchinode = None):
        self.col_index = col_index#待检测的label的列指数
        self.splitpoint_value = splitpoint_value#切分点值
        self.flag = flag #节点的信息（列表形式）——[子节点意义（大于切分点值与否等），平均值，true的比例]
        self.leftchiset = leftchiset#左子集
        self.rightchiset = rightchiset#右子集
        self.leftchinode = leftchinode#左子节点
        self.rightchinode = rightchinode#右子节点
#此函数是统合了两个对数据进行加工处理的函数，在种树的最开始应用
def dataprocess(fhand = open('train.csv','r'),amount = 7589):
    dataset_cate = [photo, commu, shop, family, busi, tool, medical, travel, health, game, book, video, social, pro, fin, sports, style, library, maps, date, comic, event, house]
    datasplit(fhand,amount)
    for dataset in dataset_cate:
        trans_data(dataset)
    return dataset_cate

#这是递归地生成决策树的函数，主要为seed，node_initial用于生成最初的决策树的root
#输入分好category的一个列表 数据为该分类下的所有app信息；输出决策树的root node
def node_initial(datas_under_category, conti = [2,3,4,11], dis = [5,7,9]):
    feature_index, splitpoint, gini, flag = multigini_select(datas_under_category,conti,dis)
    adjusted_flag = [flag[0],flag[3],flag[4],flag[5],flag[6]]#切分点形式 平均值1 占比1 平均值2 占比2
    rootnode = decisionnode(feature_index,splitpoint,adjusted_flag,flag[1],flag[2])
    return rootnode

def seed(rootnode, datas_under_category = family,conti = [2,3,4,11],dis = [5,7,9]):
    if rootnode.col_index in conti:
        conti.remove(rootnode.col_index)
    elif rootnode.col_index in dis:
        dis.remove(rootnode.col_index)
    
    if len(conti) + len(dis) == 0:
        print(0)
        if rootnode.flag[1] > rootnode.flag[3]:
            rootnode.leftchinode = True    #在最终分出来的子节点中，True表示左边（判断结果为“是”）的分支为大于4.5
            
        else:
            rootnode.leftchinode = False
    elif rootnode.flag[2] > 0.75:
        rootnode.leftchinode = True
    else:
        rootnode.leftchinode = node_initial(rootnode.leftchiset, conti, dis)
        rootnode.rightchinode = node_initial(rootnode.rightchiset, conti, dis)
        seed(rootnode.leftchinode, rootnode.leftchiset, conti, dis)
        seed(rootnode.rightchinode, rootnode.rightchiset, conti, dis)
    return rootnode

#森林函数（按category调用种树函数seed）(输入数据集 输出一个字典)（字典key为category名称，内容为category对应的决策树的root node)
#返回随机森林字典：forest
def ct_forest(fhand = open('train.csv','r'),amount = 7589):
    
    datasetsorted = dataprocess(fhand = open('train.csv','r'),amount = 7589)
    forest = {'photo':None, 'commu':None, 'shop':None, 'family':None, 'busi':None, 'tool':None, 'medical':None, 'travel':None, 'health':None, 'game':None, 'book':None, 'video':None, 'social':None, 'pro':None, 'fin':None, 'sports':None, 'style':None, 'library':None, 'maps':None, 'date':None, 'comic':None, 'event':None, 'house':None}
    
    root1 = node_initial(photo,[2,3,4,11],[5,7,9])
    forest['photo'] = seed(root1,photo)
    dis = [5,7,9]
    conti = [2,3,4,11]

    root2 = node_initial(commu,[2,3,4,11],[5,7,9])
    forest['commu'] = seed(root2,commu)
    dis = [5,7,9]
    conti = [2,3,4,11]

    root3 = node_initial(shop,[2,3,4,11],[5,7,9])
    forest['shop'] = seed(root3,shop)
    dis = [5,7,9]
    conti = [2,3,4,11]

    root4 = node_initial(photo,[2,3,4,11],[5,7,9])
    forest['family'] = seed(root4,family)
    dis = [5,7,9]
    conti = [2,3,4,11]

    root5 = node_initial(busi,[2,3,4,11],[5,7,9])
    forest['busi'] = seed(root5,busi)
    dis = [5,7,9]
    conti = [2,3,4,11]

    root6 = node_initial(tool,[2,3,4,11],[5,7,9])
    forest['tool'] = seed(root6,tool)
    dis = [5,7,9]
    conti = [2,3,4,11]

    root7 = node_initial(medical,[2,3,4,11],[5,7,9])
    forest['medical'] = seed(root7,medical)
    dis = [5,7,9]
    conti = [2,3,4,11]

    root8 = node_initial(travel,[2,3,4,11],[5,7,9])
    forest['travel'] = seed(root8,travel)
    dis = [5,7,9]
    conti = [2,3,4,11]

    root9 = node_initial(health,[2,3,4,11],[5,7,9])
    forest['health'] = seed(root9,health)
    dis = [5,7,9]
    conti = [2,3,4,11]

    root10 = node_initial(game,[2,3,4,11],[5,7,9])
    forest['game'] = seed(root10,game)
    dis = [5,7,9]
    conti = [2,3,4,11]

    root11 = node_initial(book,[2,3,4,11],[5,7,9])
    forest['book'] = seed(root11,book)
    dis = [5,7,9]
    conti = [2,3,4,11]

    root12 = node_initial(video,[2,3,4,11],[5,7,9])
    forest['video'] = seed(root12,video)
    dis = [5,7,9]
    conti = [2,3,4,11]

    root13 = node_initial(social,[2,3,4,11],[5,7,9])
    forest['social'] = seed(root13,social)
    dis = [5,7,9]
    conti = [2,3,4,11]

    root14 = node_initial(pro,[2,3,4,11],[5,7,9])
    forest['pro'] = seed(root14,pro)
    dis = [5,7,9]
    conti = [2,3,4,11]

    root15 = node_initial(fin,[2,3,4,11],[5,7,9])
    forest['fin'] = seed(root15,fin)
    dis = [5,7,9]
    conti = [2,3,4,11]

    root16 = node_initial(sports,[2,3,4,11],[5,7,9])
    forest['sports'] = seed(root16,sports)
    dis = [5,7,9]
    conti = [2,3,4,11]

    root17 = node_initial(style,[2,3,4,11],[5,7,9])
    forest['style'] = seed(root17,style)
    dis = [5,7,9]
    conti = [2,3,4,11]

    root18 = node_initial(library,[2,3,4,11],[5,7,9])
    forest['library'] = seed(root18,library)
    dis = [5,7,9]
    conti = [2,3,4,11]

    root19 = node_initial(maps,[2,3,4,11],[5,7,9])
    forest['maps'] = seed(root19,maps)
    dis = [5,7,9]
    conti = [2,3,4,11]

    root20 = node_initial(date,[2,3,4,11],[5,7,9])
    forest['date'] = seed(root20,date)
    dis = [5,7,9]
    conti = [2,3,4,11]

    root21 = node_initial(comic,[2,3,4,11],[5,7,9])
    forest['comic'] = seed(root21,comic)
    dis = [5,7,9]
    conti = [2,3,4,11]

    root22 = node_initial(event,[2,3,4,11],[5,7,9])
    forest['event'] = seed(root22,event)
    dis = [5,7,9]
    conti = [2,3,4,11]

    root23 = node_initial(house,[2,3,4,11],[5,7,9])
    forest['house'] = seed(root23,house)
    dis = [5,7,9]
    conti = [2,3,4,11]

    return forest

#最后是通过训练好的决策树去进行测试

#test_main为测试主函数，通过调用递归函数test_process，返回正确率accuracy   

def test_main(fore,correct = [],error = []):
    busi = []#
    tool = []#
    medical = []#
    travel = []#
    health = []#
    game = []#
    book = []#
    video = []#
    social = []#
    pro = []#
    fin = []#
    sports = []#
    style = []#
    library = []#
    maps = []#
    date = []#
    comic = []#
    event = []#
    house = []#
    dataset_cate = [photo, commu, shop, family, busi, tool, medical, travel, health, game, book, video, social, pro, fin, sports, style, library, maps, date, comic, event, house]
    cate = ["photo", "commu", "shop", "family", "busi", "tool", "medical", "travel", "health", "game", "book", "video", "social", "pro", "fin", "sports", "style", "library", "maps", "date", "comic", "event", "house"]
    for i in dataset_cate:
        i = []
    dataprocess(fhand = open('test.csv','r'),amount = 3254)
    for ca in range(len(cate)):
        tree = fore[cate[ca]]
        test_process(tree,dataset_cate[ca],correct,error)
    
    accu = len(correct)/(len(correct) + len(error))
    return accu

def test_process(tree, sett, correct = [], error = []):
    left,right = [],[]
    col = tree.col_index
    spl = tree.splitpoint_value
    if isinstance(spl,str):
        for row in sett:
            if row[col] == spl:
                left.append(row)
            else:
                right.append(row)
    if isinstance(spl,float) or isinstance(spl,int):
        for row in sett:
            if row[col] > spl:
                left.append(row)
            else:
                right.append(row)
    if tree.leftchinode == True:
        for row in left:
            if row[1] > 4.5:
                correct.append(row[1])
            else:
                error.append(row[1])
        for row in right:
            if row[1] < 4.5:
                correct.append(row[1])
            else:
                error.append(row[1])
    elif tree.leftchinode == False:
        for row in left:
            if row[1] <= 4.5:
                correct.append(row[1])
            else:
                error.append(row[1])
        for row in right:
            if row[1] >= 4.5:
                correct.append(row[1])
            else:
                error.append(row[1])
    else:
        if tree.leftchinode != None:
            test_process(tree.leftchinode,left,correct,error)
        if tree.rightchinode != None:
            test_process(tree.rightchinode,right,correct,error)


x = ct_forest()
accuracy = test_main(x,[],[])
print(accuracy)