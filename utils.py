import os
import torch
import torch.nn as nn
import random
import numpy as np
import scipy.sparse as sp


class Dataset(object):
    def __init__(self, path):
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)
        
        self.num_users, self.num_items = self.trainMatrix.shape
        
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()    
        return mat


# calculate NPMI values
def PMI_score(mat, k=2, is_normalised=False, eps=1e-20):
    '''mat: user-item interactoins'''
    N, M = mat.shape
    PMI_u, PMI_i = np.zeros((N, N)), np.zeros((M, M))
    NPMI_u, NPMI_i = np.zeros((N, N)), np.zeros((M, M))
    items, users = np.sum(mat,0), np.sum(mat,1)
    C_u, C_i = 0, 0   # the number of the co-occurrence user/item
    
    # obtain co-occurrence weights for users
    for i in range(N):
        for j in range(i+1, N):
            num_i, num_j = users[i], users[j]  # the number of the user i/j makes rates
            num_co = np.sum(mat[i,:]*mat[j,:]) # the number of users (i,j) make corate
            C_u += num_co
            PMI_u[i,j] = (num_co if num_co > 0 else eps) / (num_i * num_j if num_i * num_j > 0 else eps)
            PMI_u[j,i] = PMI_u[i,j]
    for i in range(N): PMI_u[i,i] = 1/C_u
    PMI_u = np.log2(PMI_u*C_u)
    
    # obtain co-occurrence weights for items
    for i in range(M):
        for j in range(i+1, M):
            num_i, num_j = items[i], items[j]  # the number of the item i/j rated by all users
            num_co = np.sum(mat[:,i]*mat[:,j]) # the number of items (i,j) corated by all users
            C_i += num_co
            PMI_i[i,j] = (num_co if num_co > 0 else eps) / (num_i * num_j if num_i * num_j > 0 else eps)
            PMI_i[j,i] = PMI_i[i,j]
    for i in range(M): PMI_i[i,i] = 1/C_i
    PMI_i = np.log2(PMI_i*C_i)
    
    PMI_u, PMI_i = PMI_u*(PMI_u > np.log2(k)), PMI_i*(PMI_i > np.log2(k))
    
    # NPMI
    if is_normalised:
        for i in range(N):
            for j in range(i+1, N):
                num_co = np.sum(mat[i,:]*mat[j,:]) # the number of users (i,j) make corate
                NPMI_u[i,j] = PMI_u[i,j]/(-np.log2(num_co/C_u if num_co>0 else eps))
                NPMI_u[j,i] = NPMI_u[i,j]

        for i in range(M):
            for j in range(i+1, M):
                num_co = np.sum(mat[:,i]*mat[:,j]) # the number of items (i,j) corated by all users
                NPMI_i[i,j] = PMI_i[i,j]/(-np.log2(num_co/C_i if num_co>0 else eps))
                NPMI_i[j,i] = NPMI_i[i,j]
                
        return NPMI_u, NPMI_i
    else:
        return PMI_u, PMI_i


# get the edges for graph building
def get_edge_index(mat):
    N, M = mat.shape
    node0, node1 = [],[]
    for i in range(N):
        for j in range(M):
            if (mat[i,j] > 0) and (i != j):
                node0.append(i)
                node1.append(j)
    edge_index  = torch.tensor([node0, node1], dtype=torch.long)
    return edge_index


# Set a seed for training
def setup_seed(seed):
    np.random.seed(seed)                         # Numpy module.
    random.seed(seed)                            # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)                      # CPU seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)             # GPU
        torch.cuda.manual_seed_all(seed)         # if you are using multi-GPU
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# generate pair instances: [user, item, rating]
# generate triple instances: [user, item_0, item_1, rating]
def generate_instances(train_mat, positive_size=1, negative_time=4, is_sparse=False, is_pair=True):
    data = []
    users_num, items_num = train_mat.shape
    
    if is_sparse:
        indptr = train_mat.indptr
        indices = train_mat.indices
    
    if is_pair:    
        for u in range(users_num):
            if is_sparse:
                rated_items = indices[indptr[u]:indptr[u+1]] #用户u中有评分项的id
            else:
                rated_items = np.where(train_mat[u,:]>0)[0]
        
            for i in rated_items:
                data.append([u,i,1.])
                for _ in range(negative_time):
                    j = np.random.randint(items_num)
                    while j in rated_items:
                        j = np.random.randint(items_num)
                    data.append([u,j,0.])
    else:            
        for u in range(users_num):
            if is_sparse:
                rated_items = indices[indptr[u]:indptr[u+1]] #用户u中有评分项的id
            else:
                rated_items = np.where(train_mat[u,:]>0)[0]
        
            for item0 in rated_items:
                for item1 in np.random.choice(rated_items, size=positive_size):
                    data.append([u,item0,item1,1.])
                for _ in range(positive_size*negative_time):
                    item1 = np.random.randint(items_num) # no matter item1 is positive or negtive
                    item2 = np.random.randint(items_num)
                    while item2 in rated_items:
                        item2 = np.random.randint(items_num)
                    data.append([u,item2,item1,0.])
    return data
