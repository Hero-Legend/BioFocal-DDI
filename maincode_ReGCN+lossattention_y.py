##############这个代码的主要思想是BioBERT+BiLSTM+REGCN+lossattention
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torch_geometric.nn import GCNConv
from torch_geometric.nn.dense import Linear as DenseLinear
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AdamW
from torchvision.transforms import transforms
from tqdm import tqdm
import torch.cuda
import xml.etree.ElementTree as ET
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import dgl
import numpy as np
import torch as th
from dgl.nn import RelGraphConv
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN
from torch.nn import MultiheadAttention
from scipy.stats import uniform, randint
import torch.optim as optim
import matplotlib.pyplot as plt



#############################################定义 BERT 模型和 tokenizer##############################################

#导入Biobert
model_path = './model_path/biobert'                     #这个要用相对路径，不要用绝对路径
biobert_tokenizer = AutoTokenizer.from_pretrained(model_path)
biobert_model = AutoModel.from_pretrained(model_path)


# #导入bert
# model_path_1 = './model_path/bert_pretrain'                     #这个要用相对路径，不要用绝对路径
# bert_tokenizer = AutoTokenizer.from_pretrained(model_path_1)
# bert_model = AutoModel.from_pretrained(model_path_1)



####################################################################################################################

#############################################读取数据################################################################

df_train = pd.read_csv('./data/ddi2013ms/augmented_train_data.tsv', sep='\t')
df_dev = pd.read_csv('./data/ddi2013ms/dev.tsv', sep='\t')
df_test = pd.read_csv('./data/ddi2013ms/test.tsv', sep='\t')
print("read")

# print("训练集数据量：", df_train.shape)
# print("验证集数据量：", df_dev.shape)
# print("测试集数据量：", df_test.shape)

####################################################################################################################

#######################################################定义模型参数##################################################
#定义训练设备，默认为GPU，若没有GPU则在CPU上训练
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备

num_label=5

# 定义模型参数
max_length = 300
batch_size = 32


# #############################################定义数据集和数据加载器###################################################
# # 定义数据集类
# 定义标签到整数的映射字典
label_map = {
    'DDI-false': 0,
    'DDI-effect': 1,
    'DDI-mechanism': 2,
    'DDI-advise': 3,
    'DDI-int': 4
    # 可以根据你的实际标签情况添加更多映射关系
}

# 定义数据集类
class DDIDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def construct_txt_intra_matrix(self, word_num):
        """构建文本模态内的矩阵"""
        mat = np.zeros((max_length, max_length), dtype=np.float32)
        mat[:word_num, :word_num] = 1.0
        return mat

    def __getitem__(self, idx):
        sentence = str(self.data['sentence'][idx])
        label_str = self.data['label'][idx]
        label = label_map[label_str]

        encoding = self.tokenizer(sentence, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
 
        # 使用 attention_mask 来确定有效的 token 数量
        word_num = encoding['attention_mask'].sum().item()
        txt_intra_matrix = self.construct_txt_intra_matrix(word_num)

        # # 输出检查语句
        # print(f"txt_intra_matrix shape: {txt_intra_matrix.shape}")

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'txt_intra_matrix': torch.tensor(txt_intra_matrix, dtype=torch.long)
        }

# 定义数据加载器
def create_data_loader(df, tokenizer, max_length, batch_size):
    dataset = DDIDataset(
        dataframe=df,
        tokenizer=tokenizer,

        max_length=max_length
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True  # 设置 drop_last=True 来丢弃最后一个不满足批次大小的批次,因为我们在LSTM和GCN维度转换时，出现了维度不匹配问题，找了很久原因，发现是在最后batch时，数据只有4条，导致维度出错
    )


# 过采样少数类
# def oversample_dataframe(df):
#     # 获取特征和标签
#     X = df['sentence'].values.reshape(-1, 1)
#     y = df['label'].map(label_map).values

#     # 过采样
#     oversampler = RandomOverSampler(sampling_strategy='not majority')
#     X_resampled, y_resampled = oversampler.fit_resample(X, y)

#     # 构建新的DataFrame
#     df_resampled = pd.DataFrame({
#         'sentence': X_resampled.flatten(),
#         'label': [list(label_map.keys())[i] for i in y_resampled]
#     })
    
#     return df_resampled

# # 过采样训练数据集
# df_train_resampled = oversample_dataframe(df_train)
# train_data_loader = create_data_loader(df_train_resampled, biobert_tokenizer, max_length, batch_size)


# # 定义一个函数来执行混合采样,用的是SMOTEENN
# def hybrid_sample_dataframe(df, label_map):
#     # 获取特征和标签
#     sentences = df['sentence'].values
#     labels = df['label'].map(label_map).values

#     # 向量化文本特征
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(sentences)

#     # 混合采样
#     sampler = SMOTEENN(sampling_strategy='not majority')
#     X_resampled, y_resampled = sampler.fit_resample(X, labels)

#     # 构建新的DataFrame
#     resampled_sentences = vectorizer.inverse_transform(X_resampled)
#     resampled_sentences = [" ".join(sent) for sent in resampled_sentences]

#     df_resampled = pd.DataFrame({
#         'sentence': resampled_sentences,
#         'label': [list(label_map.keys())[i] for i in y_resampled]
#     })
    
#     return df_resampled

# df_train_resampled = hybrid_sample_dataframe(df_train, label_map)
# train_data_loader = create_data_loader(df_train_resampled, biobert_tokenizer, max_length, batch_size)


# # # 定义一个函数来执行混合采样,用的是ADASYN
# def adasyn_sample_dataframe(df, label_map):
#     sentences = df['sentence'].values
#     labels = df['label'].map(label_map).values

#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(sentences)

#     adasyn = ADASYN(sampling_strategy='minority')
#     X_resampled, y_resampled = adasyn.fit_resample(X, labels)

#     resampled_sentences = vectorizer.inverse_transform(X_resampled)
#     resampled_sentences = [" ".join(sent) for sent in resampled_sentences]

#     df_resampled = pd.DataFrame({
#         'sentence': resampled_sentences,
#         'label': [list(label_map.keys())[i] for i in y_resampled]
#     })
    
#     return df_resampled

# df_train_resampled = adasyn_sample_dataframe(df_train, label_map)
# train_data_loader = create_data_loader(df_train_resampled, biobert_tokenizer, max_length, batch_size)



# # 加载数据集和数据加载器
train_data_loader = create_data_loader(df_train, biobert_tokenizer, max_length, batch_size)
dev_data_loader = create_data_loader(df_dev, biobert_tokenizer, max_length, batch_size)
test_data_loader = create_data_loader(df_test, biobert_tokenizer, max_length, batch_size)

# for batch in test_data_loader:
#     print(batch)
#     break  # 这将打印第一批数据并中断循环。



#图通道
class BertBiLSTM(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=1, num_classes=256, freeze_bert=True):
        super(BertBiLSTM, self).__init__()
        self.bert = biobert_model
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, 256)  # 添加线性层
        self.dropout = nn.Dropout(0.8)  # 添加Dropout层

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        lstm_output, _ = self.lstm(sequence_output)
        output_features = self.linear(lstm_output)  # 应用线性层

        return output_features


# 将节点特征和关系矩阵转换为 DGL 图对象
def create_dgl_graph(fea, mat, device):

    num_nodes = fea.shape[0]

    # 获取非零的源和目标节点索引
    nonzero_indices = np.nonzero(mat)
    src, dst = nonzero_indices[0], nonzero_indices[1]

    # 创建一个 DGL 图对象
    g = dgl.graph((src, dst), num_nodes=num_nodes)

    # 添加边权重
    edge_weights = mat[src, dst]

    g.edata['weight'] = edge_weights.clone().detach().to(device)

    # 随机生成边类型作为示例，边类型的范围是 [0, num_rels-1]
    num_rels = 5  # 关系的数量，根据实际情况调整
    etypes = th.randint(low=0, high=num_rels, size=(len(src),)).to(device)

    # 添加边类型
    g.edata['etype'] = etypes

    # 将图和节点特征移动到相同的设备上
    g = g.to(device)
    fea = fea.to(device)

    assert fea.shape[0] == num_nodes, "Feature shape does not match number of nodes"

    # 在图中添加自环
    g = dgl.add_self_loop(g)
    
    # 将节点特征添加到图中
    g.ndata['feat'] = fea 

    return g


class GCNRelationModel(nn.Module):
    def __init__(self, d_model=512, d_hidden=256, dropout=0.8):
        super().__init__()
        
        self.dp = dropout
        self.d_model = d_model
        self.hid = d_hidden

        #####################BERT_Bi-LSTM作为嵌入层，它的输出作为特征
        self.BertBiLSTM = BertBiLSTM(hidden_dim=256, num_layers=1, num_classes=256, freeze_bert=True)

        # REGCN layer
        self.RelGraphConv = RelGraphConv(in_feat=256, out_feat=256, num_rels=5, regularizer="bdd", num_bases=256, bias=True)

        self.dropout = nn.Dropout(dropout)  # 添加dropout层

        # 添加多头注意力层
        self.attention = MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.8)

        # output mlp layers
        self.MLP = nn.Linear(256*300, 256)

    def forward(self, input_ids, attention_mask, labels, mat):
	    
        # 获取当前模型所在的设备
        device = input_ids.device

	    # 节点特征
        fea = self.BertBiLSTM(input_ids, attention_mask)
    
        gcn_features_list = []
        attention_scores_list = []
        for i in range(fea.shape[0]):
            single_fea = fea[i]
            single_mat = mat[i]
            g = create_dgl_graph(single_fea, single_mat, device)

            # 提取边类型
            etypes = g.edata['etype']

            # 确保传递边类型给 RelGraphConv
            outputs = self.RelGraphConv(g, g.ndata['feat'], etypes)
            outputs = self.dropout(outputs)

            # 在这个形状下直接应用多头注意力
            attn_output, attn_scores = self.attention(outputs.unsqueeze(1), outputs.unsqueeze(1), outputs.unsqueeze(1), need_weights=True)

            # 将注意力得分保存在列表中
            attention_scores_list.append(attn_scores.squeeze(1).mean(dim=0))

            attn_output = attn_output.permute(1, 0, 2)  # 从 [num_nodes, 1, hidden_dim] 到 [1, num_nodes, hidden_dim]

            gcn_features_list.append(attn_output)

        # 将所有批次的特征拼接在一起
        gcn_features = torch.cat(gcn_features_list, dim=0)  # 形状变为 [batch_size, num_nodes, hidden_dim]

        batch_size, seq_len, hidden_size = gcn_features.shape

        # 展平特征以适应 MLP 层的输入
        gcn_features = gcn_features.view(batch_size, seq_len * hidden_size)

        gcn_features = self.MLP(gcn_features)

        # 将所有的注意力得分拼接
        attn_scores = torch.cat(attention_scores_list, dim=0)

        return gcn_features, attn_scores


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, outputs, labels):
        ce_loss = F.cross_entropy(outputs, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            at = torch.tensor([self.alpha[i] for i in labels], device=labels.device)
            focal_loss = at * focal_loss
        
        return focal_loss.mean()

# 焦点损失结合注意力
def focal_loss_with_attention(outputs, labels, attention_scores, alpha=0.8, gamma=2):
    # 焦点损失
    focal_loss_fn = FocalLoss(gamma=gamma, alpha=None)  # 你可以为每个类别传递alpha值
    focal_loss = focal_loss_fn(outputs, labels)
    attention_loss = torch.mean(attention_scores)
    return focal_loss + alpha * attention_loss



#分类器
class Classifier(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=5):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, features):
        logits = self.fc(features)  
        return logits


#整体模型
class BioMedRelationExtractor(nn.Module):
    def __init__(self):
        super(BioMedRelationExtractor, self).__init__()
        #self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        self.GCNRelationModel = GCNRelationModel()
        self.classifier = Classifier()

    def forward(self, input_ids, attention_mask, labels, mat):

        # 通过BiLSTM-GCN通道获取特征和注意力得分
        gcn_features, attn_scores = self.GCNRelationModel(input_ids, attention_mask, labels, mat)

        # 分类器进行分类
        logits = self.classifier(gcn_features)
        return logits, attn_scores


# 在训练和测试之前定义 true_labels 和 predicted_probs
true_labels = []
predicted_probs = []


# 训练代码
def train_model(model, train_data_loader, optimizer, criterion, device):

    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    epoch_true_labels = []
    epoch_pred_labels = []

    for batch in train_data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        mat = batch['txt_intra_matrix'].to(device)

        optimizer.zero_grad()
        logits, attn_scores = model(input_ids, attention_mask, labels, mat)    # 从模型中获取 outputs 和 attn_scores

        loss = focal_loss_with_attention(logits, labels, attn_scores)  # 使用焦点损失函数
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()
        
        epoch_true_labels.extend(labels.cpu().numpy())
        epoch_pred_labels.extend(predicted.cpu().numpy())

        # 记录每个 batch 的真实标签和预测概率
        true_labels.extend(labels.cpu().numpy())
        predicted_probs.extend(F.softmax(logits, dim=1).detach().cpu().numpy())  # Use detach() here

    train_loss = running_loss / len(train_data_loader)
    train_acc = correct_preds / total_preds
    
    # 计算混淆矩阵和 F1 值
    conf_matrix = confusion_matrix(epoch_true_labels, epoch_pred_labels)  # Use epoch_pred_labels here
    accuracy = accuracy_score(epoch_true_labels, epoch_pred_labels)
    precision = precision_score(epoch_true_labels, epoch_pred_labels, average='weighted', zero_division=1)
    recall = recall_score(epoch_true_labels, epoch_pred_labels, average='weighted', zero_division=1)
    f1 = 2*precision*recall/(precision+recall)
    
    return train_loss, train_acc, conf_matrix, f1


# 测试代码
def test_model(model, test_data_loader, criterion, device):

    model.eval()
    epoch_true_labels = []
    epoch_pred_labels = []

    with torch.no_grad():
        for batch in test_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            mat = batch['txt_intra_matrix'].to(device)

            logits, attn_scores = model(input_ids, attention_mask, labels, mat)
            _, predicted = torch.max(logits, 1)

            epoch_true_labels.extend(labels.cpu().numpy())
            epoch_pred_labels.extend(predicted.cpu().numpy())

    # 计算混淆矩阵、准确率、精确率、召回率和 F1 值
    conf_matrix = confusion_matrix(epoch_true_labels, epoch_pred_labels)
    accuracy = accuracy_score(epoch_true_labels, epoch_pred_labels)
    precision = precision_score(epoch_true_labels, epoch_pred_labels, average='weighted', zero_division=1)
    recall = recall_score(epoch_true_labels, epoch_pred_labels, average='weighted', zero_division=1)
    f1 = 2*precision*recall/(precision+recall)

    # 计算每个类别的F1值
    class_report = classification_report(epoch_true_labels, epoch_pred_labels, output_dict=True, zero_division=1)
    f1_per_class = {label: metrics['f1-score'] for label, metrics in class_report.items() if label.isdigit()}
    
    return conf_matrix, accuracy, precision, recall, f1, f1_per_class


#模型实例化
model = BioMedRelationExtractor().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()


# 训练模型
num_epochs = 30

# 存储训练过程中每个 epoch 的结果
epoch_train_losses = []
epoch_train_accuracies = []
epoch_train_f1_scores = []
epoch_train_conf_matrices = []

# 打开文件，以追加模式（'a'）写入
with open('./figure/training_results.txt', 'a') as f:
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):

        train_loss, train_acc, conf_matrix, f1 = train_model(model, train_data_loader, optimizer, criterion, device)

        #保存结果文件
        f.write(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, F1 Score: {f1:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix) + '\n')

        #打印输出
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        
        # 保存每个 epoch 的结果用于后续可视化
        epoch_train_losses.append(train_loss)
        epoch_train_accuracies.append(train_acc)
        epoch_train_f1_scores.append(f1)
        epoch_train_conf_matrices.append(conf_matrix)



with open('./figure/test_results.txt', 'w') as f:
    test_conf_matrix, test_accuracy, test_precision, test_recall, test_f1, test_f1_per_class = test_model(model, test_data_loader, criterion, device)
    f.write("Test Results:\n")
    f.write("Confusion Matrix:\n")
    f.write(str(test_conf_matrix) + '\n')
    f.write("Accuracy: " + str(test_accuracy) + '\n')
    f.write("Precision: " + str(test_precision) + '\n')
    f.write("Recall: " + str(test_recall) + '\n')
    f.write("F1 Score: " + str(test_f1) + '\n')
    f.write("F1 Score per Class:\n")
    for label, f1 in test_f1_per_class.items():
        f.write(f"Class {label}: {f1:.4f}\n")
    print("Test Results:")
    print("Confusion Matrix:")
    print(test_conf_matrix)
    print("Accuracy:", test_accuracy)
    print("Precision:", test_precision)
    print("Recall:", test_recall)
    print("F1 Score:", test_f1)
    print("F1 Score per Class:")
    for label, f1 in test_f1_per_class.items():
        print(f"Class {label}: {f1:.4f}")



##############################################画图####################################################
# 计算每个类别的 AUC
# 假设你有 `true_labels` 和 `predicted_probs` 以及 `label_map`
num_classes = 5  # 根据你的情况调整
fpr = dict()
tpr = dict()
roc_auc = dict()

# 对于每个类别，计算fpr, tpr和AUC
for i in range(num_classes):
    # 获取每个类的真值和预测概率
    class_true_labels = [1 if true_label == i else 0 for true_label in true_labels]
    class_predicted_probs = [probs[i] for probs in predicted_probs]

    # 计算 fpr 和 tpr
    fpr[i], tpr[i], _ = roc_curve(class_true_labels, class_predicted_probs)
    roc_auc[i] = auc(fpr[i], tpr[i])


##################################训练集结果画图####################################################
# 画训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_train_losses, marker='o', label='Train Loss')
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.title('Training Loss Over Epochs', fontsize=20)
plt.legend(fontsize=18)
plt.grid(True, linestyle="--", alpha=0.8)
plt.tight_layout()
plt.savefig('./figure/training_loss_over_epochs.png', dpi=300)
plt.show()

# 画训练准确率曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_train_accuracies, marker='o', label='Train Accuracy')
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.title('Training Accuracy Over Epochs', fontsize=20)
plt.legend(fontsize=18)
plt.grid(True, linestyle="--", alpha=0.8)
plt.tight_layout()
plt.savefig('./figure/training_accuracy_over_epochs.png', dpi=300)
plt.show()

# 画训练 F1 分数曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_train_f1_scores, marker='o', label='Train F1 Score')
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('F1 Score', fontsize=20)
plt.title('Training F1 Score Over Epochs', fontsize=20)
plt.legend(fontsize=18)
plt.grid(True, linestyle="--", alpha=0.8)
plt.tight_layout()
plt.savefig('./figure/training_f1_score_over_epochs.png', dpi=300)
plt.show()




################################################测试集结果画图##############################
# 画混淆矩阵热力图
plt.figure(figsize=(10, 8))
sns.heatmap(test_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_map.keys(), yticklabels=label_map.keys(), annot_kws={"size": 20})
plt.xlabel('Predicted Label', fontsize=20)
plt.ylabel('True Label', fontsize=20)
plt.title('Confusion Matrix', fontsize=20)
plt.xticks(rotation=45, ha='right')   # 设置x轴标签，旋转一定角度以避免重叠（如果需要）
plt.yticks(rotation=0)           # 设置y轴标签水平显示
plt.tight_layout()  # 调整子图布局以适应标签
plt.legend(fontsize=20)
plt.savefig('./figure/confusion_matrix_heatmap.png', dpi=300)  # 保存混淆矩阵热力图
plt.show()

# 画准确率
plt.figure(figsize=(10, 8))
bar_width = 0.3  # # 设置柱子的宽度,可以根据需要调整这个值
plt.bar(range(len(label_map)), [test_accuracy]*len(label_map), color='skyblue', width=bar_width, align='center')
plt.xlabel('Class', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('', fontsize=16)
plt.xticks(range(len(label_map)), label_map.keys(), rotation=45, ha='right')
plt.tight_layout()  # 调整子图布局以适应标签
plt.savefig('./figure/accuracy_by_class.png', dpi=300)  # 保存准确率图
plt.show()

#画AUC曲线图
plt.figure(figsize=(10, 6))

for i, label in enumerate(label_map):
    plt.plot(fpr[i], tpr[i], label=f'{label} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 随机分类器的线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC)', fontsize=16)
plt.legend(loc='lower right')
plt.tight_layout()  # 调整子图布局以适应标签
plt.savefig('./figure/auc_by_class.png', dpi=300)
plt.show()
