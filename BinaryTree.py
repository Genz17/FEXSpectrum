import torch
import torch.nn as nn
from Operations import unary_functions, binary_functions, unary_functions_str, binary_functions_str
from Operations import UnaryOperation, BinaryOperation

class BinaryTreeNode(object):
    def __init__(self, is_unary, isLeave=True, count=0):
        self.operation      = lambda x:0
        self.is_unary       = is_unary
        self.count          = count
        self.isLeave        = isLeave
        self.leftchild      = None
        self.rightchild     = None

    def insertLeft(self, is_unary):
        self.isLeave = False
        if self.leftchild == None:
            self.leftchild = BinaryTreeNode(is_unary)
        else:
            pass

    def insertRight(self, is_unary):
        self.isLeave = False
        if self.rightchild == None:
            self.rightchild = BinaryTreeNode(is_unary)
        else:
            pass


def OperationPlace(tree, operationInxList):
    for i in range(1, tree.count+1):
        ShowTree(tree, i).operation = str((i, operationInxList[i-1].item()))
        if ShowTree(tree,i).is_unary:
            print(ShowTree(tree,i).is_unary)
            print(str(i)+'th node\'s operation is ',unary_functions_str[operationInxList[i-1].item()],'\n')
        else:
            print(ShowTree(tree,i).is_unary)
            print(str(i)+'th node\'s operation is ',binary_functions_str[operationInxList[i-1].item()], '\n')
    # if tree.leftchild == None and tree.rightchild == None:
    #     tree.operation = operationDict[str((tree.count, operationInxList[tree.count-1].item()))]

    # if tree.leftchild != None and tree.rightchild == None:
    #     tree.operation = operationDict[str((tree.count, operationInxList[tree.count-1].item()))]
    #     OperationPlace(tree.leftchild, operationInxList, operationDict)

    # if tree.leftchild == None and tree.rightchild != None:
    #     tree.operation = operationDict[str((tree.count, operationInxList[tree.count-1].item()))]
    #     OperationPlace(tree.rightchild, operationInxList, operationDict)

    # if tree.leftchild != None and tree.rightchild != None:
    #     tree.operation = operationDict[str((tree.count, operationInxList[tree.count-1].item()))]
    #     OperationPlace(tree.leftchild, operationInxList, operationDict)
    #     OperationPlace(tree.rightchild, operationInxList, operationDict)


def BasicTreeGen():
    tree = BinaryTreeNode(False)

    tree.insertLeft(False)
    tree.insertRight(False)

    tree.leftchild.insertLeft(False)
    tree.leftchild.insertRight(False)
    tree.rightchild.insertLeft(False)
    tree.rightchild.insertRight(False)

    tree.leftchild.leftchild.insertLeft(False)
    tree.leftchild.leftchild.insertRight(False)
    tree.leftchild.rightchild.insertLeft(False)
    tree.leftchild.rightchild.insertRight(False)
    tree.rightchild.leftchild.insertLeft(False)
    tree.rightchild.leftchild.insertRight(False)
    tree.rightchild.rightchild.insertLeft(False)
    tree.rightchild.rightchild.insertRight(False)

    tree.leftchild.leftchild.leftchild.insertLeft(True)
    tree.leftchild.leftchild.leftchild.insertRight(True)
    tree.leftchild.leftchild.rightchild.insertLeft(True)
    tree.leftchild.leftchild.rightchild.insertRight(True)
    tree.leftchild.rightchild.leftchild.insertLeft(True)
    tree.leftchild.rightchild.leftchild.insertRight(True)
    tree.leftchild.rightchild.rightchild.insertLeft(True)
    tree.leftchild.rightchild.rightchild.insertRight(True)
    tree.rightchild.leftchild.leftchild.insertLeft(True)
    tree.rightchild.leftchild.leftchild.insertRight(True)
    tree.rightchild.leftchild.rightchild.insertLeft(True)
    tree.rightchild.leftchild.rightchild.insertRight(True)
    tree.rightchild.rightchild.leftchild.insertLeft(True)
    tree.rightchild.rightchild.leftchild.insertRight(True)
    tree.rightchild.rightchild.rightchild.insertLeft(True)
    tree.rightchild.rightchild.rightchild.insertRight(True)
    NodeNumCompute(tree)
    return tree

def ShowTree(tree, cnt, ans=None):
    if tree.count == cnt:
        return tree
    else:
        childList = [tree.leftchild, tree.rightchild]
        for tree in childList:
            if tree != None:
               ans = ShowTree(tree, cnt)
               if ans != None:
                   break
    if ans != None:
        return ans

def ComputeThroughTree(treeNode, inputData, operators):
    if treeNode.leftchild == None and treeNode.rightchild == None:
        ans = operators[treeNode.operation](inputData)

    if treeNode.leftchild != None and treeNode.rightchild == None:
        ans = operators[treeNode.operation](ComputeThroughTree(treeNode.leftchild,inputData,operators))

    if treeNode.leftchild == None and treeNode.rightchild != None:
        ans = operators[treeNode.operation](ComputeThroughTree(treeNode.rightchild,inputData,operators))

    if treeNode.leftchild != None and treeNode.rightchild != None:
        ans = operators[treeNode.operation](ComputeThroughTree(treeNode.leftchild,inputData,operators),ComputeThroughTree(treeNode.rightchild,inputData,operators))
    return ans

def NodeNumCompute(tree, num=0):
    if tree.leftchild == None and tree.rightchild == None:
        num = num + 1
        tree.count = num

    if tree.leftchild != None and tree.rightchild == None:
        num = NodeNumCompute(tree.leftchild, num) + 1
        tree.count = num

    if tree.leftchild == None and tree.rightchild != None:
        num = NodeNumCompute(tree.rightchild, num) + 1
        tree.count = num

    if tree.leftchild != None and tree.rightchild != None:
        num = NodeNumCompute(tree.leftchild, num)
        num = NodeNumCompute(tree.rightchild, num)
        num += 1
        tree.count = num
    return num

def TotalopNumCompute(tree):
    opList = [0 for i in range(tree.count)]
    for i in range(1,tree.count+1):
        treeNode = ShowTree(tree, i)
        if treeNode.is_unary:
            opList[treeNode.count-1] = len(unary_functions)
        else:
            opList[treeNode.count-1] = len(binary_functions)
    return opList


def LeaveNumCompute(tree):
    leaveList = []
    for i in range(1,tree.count+1):
        leave = ShowTree(tree, i)
        if leave.leftchild == None and leave.rightchild == None:
            leaveList.append(leave.count)
    return leaveList

class TrainableTree(nn.Module):
    def __init__(self, dim, outNum, medNum):
        super(TrainableTree, self).__init__()
        self.dim                = dim
        self.tree               = BasicTreeGen()
        self.medNum             = medNum
        self.outNum             = outNum
        # self.linearTransform    = {str(i): nn.Linear(dim, outNum) for i in LeaveNumCompute(self.tree)}
        # self.linearTransform    = nn.ModuleDict(self.linearTransform)
        self.linearTransform    = nn.Linear(dim, self.medNum)
        # self.OperatorsGen(self.tree)

    def forward(self, inputData):
        a = torch.prod(inputData**2-torch.ones_like(inputData), 1).view(-1,1)
        a = a/(torch.sqrt(1000+torch.sum(a**2,1)).view(-1,1))
        dataTransformed = self.linearTransform(inputData)
        res = ComputeThroughTree(self.tree, dataTransformed, self.operators)
        return a*res

    def PlaceOP(self, operationList):
        OperationPlace(self.tree, operationList)

    def LinearGen(self):
        # for key in self.linearTransform:
        #     if ShowTree(self.tree, int(key)).isLeave:
        #         for layer in self.linearTransform[key].modules():
        #             if type(layer) == type(nn.Linear(1,1)):
        #                 nn.init.kaiming_uniform_(layer.weight)
        #                 nn.init.zeros_(layer.bias)
        #                 layer.weight = nn.Parameter(layer.weight.to(torch.float64)/torch.norm(layer.weight))
        #                 layer.bias = nn.Parameter(layer.bias.to(torch.float64))
        #     else:
        #         self.linearTransform.pop(key)
        for layer in self.linearTransform.modules():
            nn.init.kaiming_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layer.weight = nn.Parameter(layer.weight.to(torch.float64)/torch.norm(layer.weight))
            layer.bias = nn.Parameter(layer.bias.to(torch.float64))

    def OperationsRefresh(self):
        for key in self.operators:
            if type(self.operators[key]) == type(UnaryOperation(unary_functions[0], 1,1,1)):
                nn.init.kaiming_uniform_(self.operators[key].li.weight)
                nn.init.zeros_(self.operators[key].li.bias)
                self.operators[key].li.weight = nn.Parameter(self.operators[key].li.weight.to(torch.float64)/torch.norm(self.operators[key].li.weight))
                self.operators[key].li.bias = nn.Parameter(self.operators[key].li.bias.to(torch.float64))
            else:
                nn.init.kaiming_uniform_(self.operators[key].li1.weight)
                nn.init.zeros_(self.operators[key].li1.bias)
                self.operators[key].li1.weight = nn.Parameter(self.operators[key].li1.weight.to(torch.float64)/torch.norm(self.operators[key].li1.weight))
                self.operators[key].li1.bias = nn.Parameter(self.operators[key].li1.bias.to(torch.float64))
                # nn.init.kaiming_uniform_(self.operators[key].li2.weight)
                # nn.init.zeros_(self.operators[key].li2.bias)
                # self.operators[key].li2.weight = nn.Parameter(self.operators[key].li2.weight.to(torch.float64)/100)
                # self.operators[key].li2.bias = nn.Parameter(self.operators[key].li2.bias.to(torch.float64)/100)
                # nn.init.kaiming_uniform_(self.operators[key].li3.weight)
                # nn.init.zeros_(self.operators[key].li3.bias)
                # self.operators[key].li3.weight = nn.Parameter(self.operators[key].li3.weight.to(torch.float64)/100)
                # self.operators[key].li3.bias = nn.Parameter(self.operators[key].li3.bias.to(torch.float64)/100)


    def OperatorsGen(self, tree, opList=None):
        self.operators = nn.ModuleDict({})
        if opList == None:
            opList = [0 for i in range(tree.count)]
        for ii in range(1, tree.count+1):
            i = opList[ii-1].item()
            if ShowTree(tree, ii).is_unary:
                if ii == tree.count:
                    self.operators.update({str((ShowTree(tree,ii).count, i)):UnaryOperation(unary_functions[i], self.medNum, self.outNum, ii)})
                else:
                    self.operators.update({str((ShowTree(tree,ii).count, i)):UnaryOperation(unary_functions[i], self.medNum, self.medNum, ii)})
            else:
                if ii == tree.count:
                    self.operators.update({str((ShowTree(tree,ii).count, i)):BinaryOperation(binary_functions[i], self.medNum, self.outNum, ii)})
                else:
                    self.operators.update({str((ShowTree(tree,ii).count, i)):BinaryOperation(binary_functions[i], self.medNum, self.medNum, ii)})
        self.operators = nn.ModuleDict(self.operators)