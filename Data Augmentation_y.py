from transformers import BioGptForCausalLM, BioGptTokenizer
import pandas as pd
import sacremoses


# 加载BioGPT模型和分词器
tokenizer = BioGptTokenizer.from_pretrained('/root/yuanzhu/BIO/Biomedical_GPT/biogpt')
model = BioGptForCausalLM.from_pretrained('/root/yuanzhu/BIO/Biomedical_GPT/biogpt')

# 加载数据集
train_data = pd.read_csv('./data/ddi2013ms/train.tsv', sep='\t')

# 检查类别分布，找出不平衡问题
category_counts = train_data['label'].value_counts()
print("Category Distribution:\n", category_counts)

# BioGPT生成函数
def generate_biogpt_text(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        temperature=0.7,
        top_k=50,
        top_p=0.85,
        do_sample=True  # 启用采样，生成多样性文本
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 重点增强少数类样本
def augment_minority_class(data, target_class, num_samples=500):
    minority_samples = data[data['label'] == target_class]
    augmented_samples = []

    # 针对少数类样本生成增强数据
    for _, row in minority_samples.iterrows():
        prompt = row['sentence']
        generated_text = generate_biogpt_text(prompt)
        
        # 保持生成数据的格式与原始数据一致
        augmented_samples.append({
            'index': row['index'],  # 保持相同的index结构
            'sentence': generated_text,
            'label': target_class
        })

        # 如果达到目标样本数，停止生成
        if len(augmented_samples) >= num_samples:
            break

    return pd.DataFrame(augmented_samples)

# 根据类别的数量分布，决定要增强的类别（类别加权增强）
target_classes = category_counts[category_counts < category_counts.max()].index
augmented_data = []

# 针对每一个少数类类别进行数据增强
for target_class in target_classes:
    print(f"Augmenting data for class: {target_class}")
    # 增强到最大类别样本数
    augmented_class_data = augment_minority_class(train_data, target_class, num_samples=category_counts.max())
    augmented_data.append(augmented_class_data)

# 合并增强的数据
augmented_df = pd.concat(augmented_data)
augmented_train_data = pd.concat([train_data, augmented_df], ignore_index=True)

# 保持与原数据集的列名和格式一致，保存增强后的数据
augmented_train_data.to_csv('augmented_train_data_balanced.tsv', sep='\t', index=False)

