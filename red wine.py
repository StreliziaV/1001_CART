#以防超出递归数量限制（默认为1000次），设置更大的数量限制
import sys
sys.setrecursionlimit(10000)

#用以下函数打开文件将每一行的数据变成list的格式，里面是float,
#并将每一行放入对应的settt里面，（settt的格式为list of lists）    
def datasplit(fhand = open('train.csv','r'),amount = 1120):  #对数据进行分组，以category为依据，并对数据组中数据进行初步规范
    settt = []
    for i in range(amount):
        
        m = fhand.readline()
        l = m.split(',')

        if i != 0:
            for i in range(len(l)):
                l[i] = eval(l[i])
            settt.append(l)
    return settt
#以下两个函数用于计算连续性变量关于rating的gini值
#连续性变量指数量较多，一般为integer或float,如文件train中的Install,size，reviews等
#取每一个变量值，所有变量值按照大于或小于该变量值分类，每一种分类方法都需要算一次gini值
#取最佳gini，以及对应变量，该变量的最佳切分点等特征
def continuous_gini(sett,col = 0): #输入的sett为之前分类好的某一个数据集，col是连续性变量所在的列
    ginis = [] #gini指数的列表，对每一个连续变量算出其gini值后，加入ginis列表，并以此判断第几个最小，应用最小的gini值
    gini_judge = None
    propl = [None]
    gini_v = 100
    for row_1 in sett:
        nod = row_1[col]
        large = []
        small = []
        for row_2 in sett:
            if row_2[col] < nod :
                small.append(row_2[11])
            else:
                large.append(row_2[11])
        lar,sma = len(large),len(small)
        r_large1,r_large2= [],[]
        r_small1,r_small2 = [],[]
        for rate in large:
            if rate <= 6:
                r_small1.append(rate)
            else:
                r_large1.append(rate)
        for rate in small:
            if rate <= 6:
                r_small2.append(rate)
            else:
                r_large2.append(rate)
        r_lar1,r_lar2 = len(r_large1),len(r_large2)
        r_sma1,r_sma2 = len(r_small1),len(r_small2)
        if r_lar1+r_sma1 != 0:
            p1 = r_lar1/(r_lar1+r_sma1)
        else:
            p1 = 10
        if r_lar2+r_sma2 != 0:
            p2 = r_lar2/(r_lar2+r_sma2)
        else:
            p2 = 10
        gini = (lar/(lar+sma))*(2*p1*(1-p1)) + (sma/(lar+sma))*(2*p2*(1-p2))
        ginis.append(gini)
    ginis_1 = sorted(ginis) #这里开始，是找出最小的gini值在数据集中对应的变量值：result
    result = 0
    for i in range(len(ginis)): 
        if ginis_1[0] == ginis[i]:
            result = i
    if gini_v < ginis_1[0]:
        gini_v = 10000

    else:
        gini_judge = ginis_1[0]
        big,lit = [],[]
        big1,lit1 = [],[]
        sum1,sum2 = 0,0
        sum3,sum4 = 0,0
        b1,b2,l1,l2 = [],[],[],[]
        for row in sett:
            if row[col] >= sett[result][col] :
                big.append(row[11])
                big1.append(row)
            else:
                lit1.append(row)
                lit.append(row[11])
        for i in big:
            if i > 6:
                b1.append(i)
                sum1 = sum1 + i
            else:
                b2.append(i)
                sum2 += i
        for i in lit:
            if i > 6:
                l1.append(i)
                sum3 += i
            else:
                l2.append(i)
                sum4 += i
        if len(big) != 0:
            aveb = (sum1+sum2)/(len(big))
        else:
            aveb = 0
        if len(lit) != 0:
            avel = (sum3+sum4)/(len(lit))
        else:
            avel = 0
        if len(b1) + len(b2) != 0:
            prob = (len(b1))/(len(b1)+len(b2))
        else:
            prob = 0
        if len(l1)+len(l2) != 0:
            prol = (len(l1))/(len(l1)+len(l2))
        else:
            prol = 0
        propl.append(big1)
        propl.append(lit1)
        propl.append(aveb)
        propl.append(prob)
        propl.append(avel)
        propl.append(prol)
        
        
        result = sett[result][col]
        
    return gini_judge,result,propl#col的index 切分值 gini 列表[平均值 占比 平均值 占比]

def continuous_feature(sett,conti = [0,1,2,3,4,5,6,7,8,9,10]):
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

#以下开始实现树的建设
#这个class用于定义决策树上的node
class decisionnode:
    def __init__(self, col_index = None, splitpoint_value = None, flag = None, leftchiset = None, rightchiset = None, leftchinode = None, rightchinode = None, parent = None):
        self.col_index = col_index#待检测的label的列指数
        self.splitpoint_value = splitpoint_value#切分点值
        self.flag = flag #节点的信息（列表形式）——[子节点意义（大于切分点值与否等），平均值，true的比例]
        self.leftchiset = leftchiset#左子集
        self.rightchiset = rightchiset#右子集
        self.leftchinode = leftchinode#左子节点
        self.rightchinode = rightchinode#右子节点
        self.parent = parent     #parent为一个分支节点的母节点，除了根节点，其他所有节点都有一个母节点


#此函数用于生成最开始的树根节点
def node_initial(datas_under_category, conti = [0,1,2,3,4,5,6,7,8,9,10], parent = None):
    feature_index, splitpoint, gini, flag = continuous_feature(datas_under_category,conti)
    try:
        adjusted_flag = [flag[0],flag[3],flag[4],flag[5],flag[6]]#切分点形式 平均值1 占比1 平均值2 占比2
        rootnode = decisionnode(feature_index, splitpoint, adjusted_flag, flag[1], flag[2], parent=parent)
    except:
        rootnode = decisionnode(None,None,None,None,None,parent = parent)
    return rootnode

#此函数为树的生长过程，通过对rootnode,datas_under_category,和conti进行替换，实现递归生长
#其中有比较详细的判断是否大于6的过程
def seed(rootnode, datas_under_category, conti = [0,1,2,3,4,5,6,7,8,9,10]):
    if rootnode.col_index in conti:
        conti.remove(rootnode.col_index)
        
    if len(conti) == 0:
        if rootnode.flag != None:
            if rootnode.flag[2] > 0.5:     
                rootnode.leftchinode = True    #！！！在最终分出来的子节点中，True表示左边（判断结果为“是”）的分支为大于6
            else:
                rootnode.leftchinode = False 
        else:
            pa = rootnode.parent     #在递归分支过程中，会出现一个节点的特征全为None的情况，该情况下，需要通过self.parent链接回到母节点
            if pa.flag != None:      #并通过母节点的flag特征直接对母节点的两个子节点进行输出判断
                if pa.flag[2] > 0.5:         #！！！采取多数表决模式
                    pa.leftchinode = True    #！！！！在最终分出来的子节点中，True表示左边（判断结果为“是”）的分支为大于6，其对应的右分支为小于6
                else:                        #！！！False则表示相反的情况
                    pa.leftchinode = False 

    elif rootnode.flag[2] > 0.5:        #预剪枝，若一个分支中，质量数大于6占比超过百分之五十，则判断该节点为大于6的分支
        rootnode.leftchinode = True     #而其对应的同源子分支为小于6的分支
    elif rootnode.flag[4] > 0.5:
        rootnode.leftchinode = False
    elif conti != []:
        
        f = rootnode.flag
        if (len(rootnode.leftchiset) == 0) or (len(rootnode.rightchiset) == 0):
            if f[1] > f[3]:
                rootnode.leftchinode = True
            else:
                rootnode.leftchinode = False
        else:   #递归生长
            rootnode.leftchinode = node_initial(rootnode.leftchiset, conti, parent = rootnode)
            rootnode.leftchinode = seed(rootnode.leftchinode, rootnode.leftchiset, conti)
            rootnode.rightchinode = node_initial(rootnode.rightchiset, conti, parent = rootnode)
            rootnode.rightchinode = seed(rootnode.rightchinode, rootnode.rightchiset, conti)
    
    return rootnode

#此函数为测试用的主函数
#通过对变量的更改，实现递归，并返回正确率accu
def test_process(tree, sett, correct = [], error = []):
    left,right = [],[]         #这两个列表为测试数组依照决策树中的判断条件，分出的大于与小于的部分，两个都是通过sett生成的新的list of list
    col = tree.col_index
    spl = tree.splitpoint_value   #决策树中节点的变量分裂点(用做判断依据，如该数据为0.5，那么大于0.5的数据就会进入left，小于的进入right)
    if col != None:
        for row in sett:
            if row[col] > spl:
                left.append(row)
            else:
                right.append(row)
    if tree.leftchinode == True:
        for row in left:
            if row[11] > 6:
                correct.append(row[11])
            else:
                error.append(row[11])
        for row in right:
            if row[1] < 6:
                correct.append(row[11])
            else:
                error.append(row[11])
    elif tree.leftchinode == False:
        for row in left:
            if row[11] <= 6:
                correct.append(row[11])
            else:
                error.append(row[11])
        for row in right:
            if row[11] >= 6:
                correct.append(row[11])
            else:
                error.append(row[11])
    else:
        if tree.leftchinode != None:
            test_process(tree.leftchinode,left,correct,error)
        
        if tree.rightchinode != None:
            test_process(tree.rightchinode,right,correct,error)
    accu = (len(correct))/(len(correct)+len(error))
    return accu
#先通过datasplit对原始csv文件中的数据进行整理，并添加到sett中，sett的格式为list of list，每一行数据，在sett中为一个列表
sett = datasplit()
#生成最开始的树根节点
root = node_initial(sett)
#将树根节点输入seed函数，实现在根节点的基础上，通过左右分支完善一整棵树
#返回的tree是长成后整棵树的树根节点
tree = seed(root,sett)

#以下是对test文件测试的过程
#先用datasplit对test中的数据加工处理，同sett一样，输出一个list of list
test = datasplit(fhand = open('test.csv','r'),amount = 481)
#将树的根节点输入测试主函数，通过递归完成对所有数据的判断，返回值为准确率
accuracy = test_process(tree,test)
print("the accuracy is ", accuracy)
print(tree)