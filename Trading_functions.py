# Functions & Classes for Trading
import pickle
import numpy as np
from pymongo import MongoClient
import pandas as pd

# Real price data, so cannot upload on github repo
# with open('./close_df.pkl', 'rb') as f:
#     close_df = pickle.load(f)
# with open('./volume_df.pkl', 'rb') as f:
#     volume_df = pickle.load(f)
# with open('./acml_df.pkl', 'rb') as f:
#     acml_df = pickle.load(f)
# with open('./low_df.pkl', 'rb') as f:
#     low_df = pickle.load(f)
# with open('./high_df.pkl', 'rb') as f:
#     high_df = pickle.load(f)
# with open('./open_df.pkl', 'rb') as f:
#     open_df = pickle.load(f)
# with open('./return_df.pkl', 'rb') as f:
#     return_df = pickle.load(f)
    
mask = (volume_df==0).replace(True, np.nan).replace(False, 1)
close_df *= mask
volume_df *= mask
acml_df *= mask
low_df *= mask
high_df *= mask
open_df *= mask

    
def get_item(data_id):
    if data_id == 'stck_clpr':
        return close_df
    elif data_id == 'stck_oprc':
        return open_df
    elif data_id == 'stck_hgpr':
        return high_df
    elif data_id == 'stck_lwpr':
        return low_df
    elif data_id == 'acml_vol':
        return acml_df
    else:
        return volume_df

def get_item2(data_id):
    if data_id == 'stck_clpr':
        return close_df2
    elif data_id == 'stck_oprc':
        return open_df2
    elif data_id == 'stck_hgpr':
        return high_df2
    elif data_id == 'stck_lwpr':
        return low_df2
    elif data_id == 'acml_vol':
        return acml_df2
    else:
        return volume_df2
    
# FactorGraph
class FactorGraph(object):
    def __init__(self, id_set):
        self.id_set = id_set
        self.data = None
        self.expression = None
        self.depth = 0
    
    def initialize(self, concept_id, full, training):
        self.expression = concept_id
        if full:
            self.data = get_item(concept_id)
        else:
            self.data = get_item2(concept_id)
        if training == 'training':
            self.data = self.data.loc['20060101':'20170101']
        elif training == 'validation':
            self.data = self.data.loc['20170101':'20210101']
        elif training == 'test':
            self.data = self.data.loc['20210101':]
        else:
            pass
        
# Get Data operator
def get(f_graph):
    f_graph.expression = 'GET' +'(' + f_graph.expression + ')'
    return f_graph

# Modification operator
def shift_N(f_graph, shift_num):
    f_graph.expression = 'SHIFT ' + str(shift_num) + '(' + f_graph.expression + ')'
    f_graph.data = f_graph.data.shift(shift_num)
    return f_graph

def rollingMean_N(f_graph, roll_num):
    f_graph.expression = 'ROLLING MEAN ' + str(roll_num) + '(' + f_graph.expression + ')'
    f_graph.data = f_graph.data.rolling(roll_num).mean()
    return f_graph

def rollingStdv_N(f_graph, roll_num):
    f_graph.expression = 'ROLLING STDV ' + str(roll_num) + '(' + f_graph.expression + ')'
    f_graph.data = f_graph.data.rolling(roll_num).std()
    return f_graph

def delta(f_graph, change_num):
    f_graph.expression = 'DELTA '+ str(change_num)+'(' + f_graph.expression + ')'
    f_graph.data = f_graph.data.diff(periods=change_num)
    return f_graph

def pct_delta(f_graph, change_num):
    f_graph.expression = 'PCT DELTA '+ str(change_num)+'(' + f_graph.expression + ')'
    f_graph.data = f_graph.data.pct_change(periods=change_num)
    return f_graph

# Combination Operator

def plus(f_graph, f_graph2):
    f_graph.expression = 'PLUS(' + f_graph.expression + ',' + f_graph2.expression + ')'
    f_graph.data = f_graph.data + f_graph2.data
    return f_graph

def minus(f_graph, f_graph2):
    f_graph.expression = 'MINUS(' + f_graph.expression + ',' + f_graph2.expression + ')'
    f_graph.data = f_graph.data - f_graph2.data
    return f_graph

def multiply(f_graph, f_graph2):
    f_graph.expression = 'MULTIPLY(' + f_graph.expression + ',' + f_graph2.expression + ')'
    f_graph.data = f_graph.data * f_graph2.data
    return f_graph

def divide(f_graph, f_graph2):
    f_graph.expression = 'DIVIDE(' + f_graph.expression + ',' + f_graph2.expression + ')'
    f_graph.data = f_graph.data / f_graph2.data
    return f_graph

def max_operator(f_graph, f_graph2):
    f_graph.expression = 'MAX(' + f_graph.expression + ',' + f_graph2.expression + ')'
    f_graph.data = (f_graph.data >= f_graph2.data) * f_graph.data + (f_graph2.data > f_graph.data) * f_graph2.data
    return f_graph

def min_operator(f_graph, f_graph2):
    f_graph.expression = 'MIN(' + f_graph.expression + ',' + f_graph2.expression + ')'
    f_graph.data = (f_graph.data <= f_graph2.data) * f_graph.data + (f_graph2.data < f_graph.data) * f_graph2.data
    return f_graph

def weight_avg(f_graph, f_graph2, w):
    f_graph.expression = 'WEIGHT AVG '+ str(w) +'(' + f_graph.expression + ',' + f_graph2.expression + ')'
    f_graph.data = f_graph.data*w * f_graph2.data*(1-w)
    return f_graph

def geometric_avg(f_graph, f_graph2, w):
    f_graph.expression = 'GEO AVG '+ str(w) +'(' + f_graph.expression + ',' + f_graph2.expression + ')'
    f_graph.data = f_graph.data**w * f_graph2.data**(1-w)
    return f_graph

# Filter operator - 1
def bigger(f_graph, num):
    f_graph.expression = 'BIGGER ' + str(num) + '(' + f_graph.expression + ')'
    f_graph.data = (f_graph.data > num).astype(float)
    return f_graph

# Filter operator - 2
def bigger_double(f_graph, f_graph2):
    f_graph.expression = 'BIGGER DOUBLE'+'(' + f_graph.expression + ',' + f_graph2.expression + ')'
    f_graph.data = (f_graph.data > f_graph2.data).astype(float)
    return f_graph

dic_imputer = {'GET': get}
p_imputer = [1]

dic_modification = {'SHIFT 1': lambda x: shift_N(x, 1),
              'SHIFT 2': lambda x: shift_N(x, 2),
              'SHIFT 3': lambda x: shift_N(x, 3),
              'SHIFT 4': lambda x: shift_N(x, 4),
              'SHIFT 5': lambda x: shift_N(x, 5),
              'ROLLING MEAN 5': lambda x: rollingMean_N(x, 5),
              'ROLLING MEAN 10': lambda x: rollingMean_N(x, 10),
              'ROLLING MEAN 20': lambda x: rollingMean_N(x, 20),
              'ROLLING MEAN 60': lambda x: rollingMean_N(x, 60),
              'ROLLING STDV 5': lambda x: rollingStdv_N(x, 5),
              'ROLLING STDV 10': lambda x: rollingStdv_N(x, 10),
              'ROLLING STDV 20': lambda x: rollingStdv_N(x, 20),
              'ROLLING STDV 60': lambda x: rollingStdv_N(x, 60),
              'DELTA 1': lambda x: delta(x, 1),
              'DELTA 2': lambda x: delta(x, 2),
              'DELTA 3': lambda x: delta(x, 3),
              'DELTA 5': lambda x: delta(x, 5),
              'PCT DELTA 1': lambda x: pct_delta(x, 1),
              'PCT DELTA 2': lambda x: pct_delta(x, 2),
              'PCT DELTA 3': lambda x: pct_delta(x, 3),
              'PCT DELTA 5': lambda x: pct_delta(x, 5),
                   }
p_modification = [1/21] * 21 # can apply optimization here

dic_combination = {'PLUS': plus,
                  'MINUS': minus,
                  'MULTIPLY': multiply,
                  'DIVIDE': divide,
                   'WEIGHT AVG 0.3': lambda x, y: weight_avg(x,y, 0.3),
                   'WEIGHT AVG 0.5': lambda x, y: weight_avg(x,y, 0.5),
                   'WEIGHT AVG 0.7': lambda x, y: weight_avg(x,y, 0.7),
                   'GEO AVG 0.3': lambda x, y: geometric_avg(x,y, 0.3),
                   'GEO AVG 0.5': lambda x, y: geometric_avg(x,y, 0.5),
                   'GEO AVG 0.7': lambda x, y: geometric_avg(x,y, 0.7),
                  }
p_combination = [1/6] * 4 + [1/18] * 6 # can apply optimization here

dic_filter = {
    'BIGGER 0': lambda x: bigger(x, 0),
    'BIGGER 1': lambda x: bigger(x, 1),
    'BIGGER DOUBLE': lambda x, y: bigger_double(x,y),
}
p_filter = [1/3] * 3

dic_operator = [dic_imputer, p_imputer, dic_modification, p_modification, dic_combination, p_combination, dic_filter, p_filter]

class GraphGenerator(object):
    def __init__(self, id_set, dic_operator):
        self.id_set = id_set
        self.dic_imputer = dic_operator[0]
        self.p_imputer = dic_operator[1]
        self.dic_modification = dic_operator[2]
        self.p_modification = dic_operator[3]
        self.dic_combination = dic_operator[4]
        self.p_combination = dic_operator[5]
        self.dic_filter = dic_operator[6]
        self.p_filter = dic_operator[7]
    
    def expression_to_graph(self, expression, full, training):
        operation = expression.split('(')[0]
        if operation in self.dic_imputer.keys():
            f_graph = FactorGraph(self.id_set)
            f_graph.initialize(expression.split('(')[1].split(')')[0], full, training)
            return self.dic_imputer[operation](f_graph)
        elif operation in self.dic_modification.keys():
            bracket_1 = expression.find('(')
            bracket_2 = expression[::-1].find(')')
            return self.dic_modification[operation](self.expression_to_graph(expression[bracket_1+1:len(expression)-bracket_2-1], full, training))
        elif operation in self.dic_combination.keys():
            counter = 0
            bracket_1 = expression.find('(')
            bracket_2 = expression[::-1].find(')')
            inner = expression[bracket_1+1:len(expression)-bracket_2-1]
            for i in range(len(inner)):
                if (counter == 0) and inner[i] == ',':
                    first = inner[:i]
                    second = inner[i+1:]
                elif inner[i] == '(':
                    counter +=1
                elif inner[i] == ')':
                    counter -=1
            return self.dic_combination[operation](self.expression_to_graph(first, full, training), self.expression_to_graph(second, full, training))
        elif operation == 'BIGGER DOUBLE':
            counter = 0
            bracket_1 = expression.find('(')
            bracket_2 = expression[::-1].find(')')
            inner = expression[bracket_1+1:len(expression)-bracket_2-1]
            for i in range(len(inner)):
                if (counter == 0) and inner[i] == ',':
                    first = inner[:i]
                    second = inner[i+1:]
                elif inner[i] == '(':
                    counter +=1
                elif inner[i] == ')':
                    counter -=1
            return self.dic_filter[operation](self.expression_to_graph(first, full, training), self.expression_to_graph(second, full, training))
        else:
            bracket_1 = expression.find('(')
            bracket_2 = expression[::-1].find(')')
            return self.dic_filter[operation](self.expression_to_graph(expression[bracket_1+1:len(expression)-bracket_2-1], full, training))
        
# RandomGenerator
class RandomGenerator(GraphGenerator):
    def __init__(self, id_set, dic_operator):
        super().__init__(id_set, dic_operator)
    
    def random_expression(self, depth, max_depth):
        rand_value = random.random()
        if depth ==0:
            if random.random() < 1/3:
                depth +=1
                return 'BIGGER DOUBLE' + '(' + self.random_expression(depth, max_depth) + ',' + self.random_expression(depth, max_depth) + ')'
            else:
                depth +=1
                return np.random.choice(['BIGGER 0', 'BIGGER 1'], p=[0.5, 0.5]) + '(' + self.random_expression(depth, max_depth) + ')'
        if rand_value < depth/max_depth:
            rand_ind = int(random.random()*len(self.id_set))
            return 'GET' + '('+ self.id_set[rand_ind]+ ')'
        elif rand_value < 1 - depth/(max_depth):
            rand_ind = int(random.random()*len(self.p_modification))
            depth +=1
            return np.random.choice(list(self.dic_modification.keys()), p=self.p_modification) + '(' + self.random_expression(depth, max_depth) + ')'
        else:
            rand_ind = int(random.random()*len(self.p_combination))
            depth +=1
            return np.random.choice(list(self.dic_combination.keys()), p=self.p_combination) + '(' + self.random_expression(depth, max_depth) + ',' + self.random_expression(depth, max_depth) + ')'
        
def winsorize(x):
    if (x>0.69) and (x<1.31):
        return x
    else:
        return 1