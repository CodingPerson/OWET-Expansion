# 原始数据集路径
from transformers import BertModel, BertTokenizer

raw_dataPath = {
    'BBN': 'dataset/BBN/test.json',
    'OntoNotes': ['dataset/ontonotes/g_test.json', 'dataset/ontonotes/g_dev.json'],
    'FewNerd': ['dataset/FewNerd/supervised/train.txt',
                'dataset/FewNerd/supervised/dev.txt',
                'dataset/FewNerd/supervised/test.txt']
}

# 统一格式数据集路径
std_dataPath = {
    'BBN': 'dataset/stdData/BBN/BBN.json',
    'OntoNotes': 'dataset/stdData/OntoNotes/OntoNotes.json',
    "FewNerd": 'dataset/stdData/FewNerd/FewNerd.json'
}

# unknown中的类型
unknownSet = {
    'BBN': {'/WORK_OF_ART', '/WORK_OF_ART/BOOK', '/WORK_OF_ART/SONG', '/WORK_OF_ART/PLAY', '/SUBSTANCE/FOOD',
            '/ANIMAL', '/LAW', '/ORGANIZATION/EDUCATIONAL', '/LOCATION/RIVER', '/GPE/COUNTRY', '/EVENT/WAR'},
    'OntoNotes': {'/person/title', '/other/art', '/other/art/writing', '/other/art/broadcast',
                  '/person/artist', '/person/artist/author', '/other/legal', '/other/food',
                  '/other/product/car', '/organization/company', '/organization/company/news',
                  '/organization/company/broadcast', '/location/geography/body_of_water',
                  '/other/event/violent_conflict'},
    'FewNerd': {'/location/body_of_water', '/location/park', '/person/soldier', '/organization/education',
                '/building/hospital', '/building/restaurant', '/art', '/art/film', '/art/written_art', '/art/broadcast',
                '/art/other', '/art/music', '/art/painting', '/product/food', '/product/weapon',
                '/event/natural_disaster', '/misc/disease', '/misc/law'}
}

# 划分后数据集路径
split_dataPath = {
    'BBN': (
        'dataset/stdData/BBN/split/labeled.json',
        'dataset/stdData/BBN/split/val.json',
        'dataset/stdData/BBN/split/unlabeled.json',
        'dataset/stdData/BBN/split/unknown.json',
        'dataset/stdData/BBN/split/extra_known.json'
    ),
    'OntoNotes': (
        'dataset/stdData/OntoNotes/split/labeled.json',
        'dataset/stdData/OntoNotes/split/val.json',
        'dataset/stdData/OntoNotes/split/unlabeled.json',
        'dataset/stdData/OntoNotes/split/unknown.json',
        'dataset/stdData/OntoNotes/split/extra_known.json'
    ),
    'FewNerd': (
        'dataset/stdData/FewNerd/split/labeled.json',
        'dataset/stdData/FewNerd/split/unlabeled.json',
        'dataset/stdData/FewNerd/split/unknown.json'
    )
}
split_infoPath = {
    'BBN': 'dataset/stdData/BBN/split/info.csv',
    'OntoNotes': 'dataset/stdData/OntoNotes/split/info.csv',
    "FewNerd": 'dataset/stdData/FewNerd/split/info.csv'
}

# 类型映射文件路径
type2id_path = {
    'BBN': ('dataset/stdData/BBN/split/type2id.txt', 'dataset/stdData/BBN/split/type2id_pad.txt'),
    'OntoNotes': ('dataset/stdData/OntoNotes/split/type2id.txt', 'dataset/stdData/OntoNotes/split/type2id_pad.txt'),
    "FewNerd": ('dataset/stdData/FewNerd/split/type2id.txt', 'dataset/stdData/FewNerd/split/type2id_pad.txt')
}

# known和unknown类型数量
type_nums = {
    'BBN': {'base': {'known': 34, 'unknown': 11}, 'pad': {'known': 47, 'unknown': 14}},
    'OntoNotes': {'base': {'known': 43, 'unknown': 14}, 'pad': {'known': 74, 'unknown': 20}},
    "FewNerd": {'base': {'known': 56, 'unknown': 18}, 'pad': {'known': 56, 'unknown': 18}}
}

# 数据集的正确聚类数量
dataset_gold_k = {
    'BBN': [45,16],
    'OntoNotes': [56,30,4],
    'FewNerd': 66,
}

type_description = {
    'BBN': 'dataset/stdData/type_desc/BBN/gpt-5.4-type.txt',
    'OntoNotes': 'dataset/stdData/type_desc/OntoNotes/gpt-5.4-type.txt',
    'FewNerd': 'dataset/stdData/type_desc/FewNerd/gpt-5.4-type.txt',
}

# 每个数据集的level级数
level_num = {
    'BBN': 2,
    'OntoNotes': 3,
    'FewNerd': 2,
}

# 基础模型及路径
base_model = {
    'bert': {'model': BertModel, 'tokenizer': BertTokenizer},
}
model_type_path = {
    'bert-base-uncased': '.modelfile/bert-base-uncased',
}
