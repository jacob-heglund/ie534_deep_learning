import numpy as np
import os
import nltk
import itertools
import io

nltk.download('punkt')

##########################################
# program controls
blue_waters = 0
##########################################
# set important directories
if blue_waters:
    src_dir = '/mnt/c/scratch/training/tra392/hw7/src'
    train_directory = '/projects/training/bauh/NLP/aclImdb/train/'
    test_directory = '/projects/training/bauh/NLP/aclImdb/test/'
    glove_filename = '/projects/training/bauh/NLP/glove.840B.300d.txt'

else:
    src_dir = 'C:/home/classes/IE534_DL/hw7/src'
    train_directory = os.path.join(src_dir, 'data/aclImdb/train/')
    test_directory = os.path.join(src_dir, 'data/aclImdb/test/')
    glove_filename = os.path.join(src_dir, 'data/glove.840B.300d.txt')

preprocessed_data_dir = os.path.join(src_dir, 'preprocessed_data')
## create directory to store preprocessed data
if not os.path.exists(preprocessed_data_dir):
    os.mkdir(preprocessed_data_dir)

## get all of the training reviews (including unlabeled reviews)

pos_filenames = os.listdir(train_directory + 'pos/')
neg_filenames = os.listdir(train_directory + 'neg/')
unsup_filenames = os.listdir(train_directory + 'unsup/')

pos_filenames = [train_directory+'pos/'+filename for filename in pos_filenames]
neg_filenames = [train_directory+'neg/'+filename for filename in neg_filenames]
unsup_filenames = [train_directory+'unsup/'+filename for filename in unsup_filenames]

filenames = pos_filenames + neg_filenames + unsup_filenames

count = 0
x_train = []
for filename in filenames:
    with io.open(filename,'r',encoding='utf-8') as f:
        line = f.readlines()[0]
    line = line.replace('<br />',' ')
    line = line.replace('\x96',' ')
    line = nltk.word_tokenize(line)
    line = [w.lower() for w in line]

    x_train.append(line)
    count += 1
    if count % 500 == 0:
        print(count)

## get all of the test reviews
pos_filenames = os.listdir(test_directory + 'pos/')
neg_filenames = os.listdir(test_directory + 'neg/')

pos_filenames = [test_directory+'pos/'+filename for filename in pos_filenames]
neg_filenames = [test_directory+'neg/'+filename for filename in neg_filenames]

filenames = pos_filenames+neg_filenames

count = 0
x_test = []
for filename in filenames:
    with io.open(filename,'r',encoding='utf-8') as f:
        line = f.readlines()[0]
    line = line.replace('<br />',' ')
    line = line.replace('\x96',' ')
    line = nltk.word_tokenize(line)
    line = [w.lower() for w in line]

    x_test.append(line)
    count += 1
    if count % 500 == 0:
        print(count)


## number of tokens per review
no_of_tokens = []
for tokens in x_train:
    no_of_tokens.append(len(tokens))
no_of_tokens = np.asarray(no_of_tokens)
print('Total: ', np.sum(no_of_tokens), ' Min: ', np.min(no_of_tokens), ' Max: ', np.max(no_of_tokens), ' Mean: ', np.mean(no_of_tokens), ' Std: ', np.std(no_of_tokens))

### word_to_id and id_to_word. associate an id to every unique token in the training data
all_tokens = itertools.chain.from_iterable(x_train)
word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}

all_tokens = itertools.chain.from_iterable(x_train)
id_to_word = [token for idx, token in enumerate(set(all_tokens))]
id_to_word = np.asarray(id_to_word)

## let's sort the indices by word frequency instead of random
x_train_token_ids = [[word_to_id[token] for token in x] for x in x_train]
count = np.zeros(id_to_word.shape)
for x in x_train_token_ids:
    for token in x:
        count[token] += 1
indices = np.argsort(-count)
id_to_word = id_to_word[indices]
count = count[indices]

hist = np.histogram(count,bins=[1,10,100,1000,10000])
print(hist)
for i in range(10):
    print(id_to_word[i],count[i])

## recreate word_to_id based on sorted list
word_to_id = {token: idx for idx, token in enumerate(id_to_word)}
## assign -1 if token doesn't appear in our dictionary
## add +1 to all token ids, we went to reserve id=0 for an unknown token
x_train_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_train]
x_test_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_test]

## save dictionary
dict_path = os.path.join(preprocessed_data_dir, 'imdb_dictionary.npy')
np.save(dict_path, np.asarray(id_to_word))

## save training data to single text file
train_path = os.path.join(preprocessed_data_dir, 'imdb_train.txt')
with io.open(train_path,'w',encoding='utf-8') as f:
    for tokens in x_train_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")

test_path = os.path.join(preprocessed_data_dir, 'imdb_test.txt')
## save test data to single text file
with io.open(test_path, 'w',encoding='utf-8') as f:
    for tokens in x_test_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")

with io.open(glove_filename,'r',encoding='utf-8') as f:
    lines = f.readlines()

glove_dictionary = []
glove_embeddings = []
count = 0
for line in lines:
    line = line.strip()
    line = line.split(' ')
    glove_dictionary.append(line[0])
    embedding = np.asarray(line[1:],dtype=np.float)
    glove_embeddings.append(embedding)
    count+=1
    if(count>=100000):
        break

glove_dictionary = np.asarray(glove_dictionary)
glove_embeddings = np.asarray(glove_embeddings)
# added a vector of zeros for the unknown tokens
glove_embeddings = np.concatenate((np.zeros((1,300)),glove_embeddings))

word_to_id = {token: idx for idx, token in enumerate(glove_dictionary)}

x_train_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_train]
x_test_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_test]

path1 = os.path.join(preprocessed_data_dir, 'glove_dictionary.npy')
np.save(path1, glove_dictionary)
path2 = os.path.join(preprocessed_data_dir, 'glove_embeddings.npy')
np.save(path2, glove_embeddings)

path3 = os.path.join(preprocessed_data_dir, 'imdb_train_glove.txt')
with io.open(path3, 'w',encoding='utf-8') as f:
    for tokens in x_train_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")

path4 = os.path.join(preprocessed_data_dir, 'imdb_test_glove.txt')
with io.open(path4, 'w',encoding='utf-8') as f:
    for tokens in x_test_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")
