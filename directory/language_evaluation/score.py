import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import warnings

# 忽略BERTScore的警告
warnings.filterwarnings("ignore")

# 读取Excel文件
# df = pd.read_excel("nhs_test_no_rag.xlsx", sheet_name="Sheet1")
df = pd.read_excel("nhs_test_with_rag.xlsx", sheet_name="Sheet1")

# 初始化评估工具
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# 定义计算指标的函数
def calculate_metrics(row):
    # 提取候选文本和参考文本（假设参考文本为"Treatments"列）
    candidate_before = str(row["Response before RAG"])
    candidate_after = str(row["Response after RAG"])
    reference = str(row["Treatments"])

    # 预处理（简单分词）
    ref_tokens = reference.split()
    cand_before_tokens = candidate_before.split()
    cand_after_tokens = candidate_after.split()

    metrics = {}

    # 计算BLEU（使用n-gram权重）
    try:
        metrics["BLEU_before"] = sentence_bleu([ref_tokens], cand_before_tokens)
        metrics["BLEU_after"] = sentence_bleu([ref_tokens], cand_after_tokens)
    except:
        metrics["BLEU_before"] = metrics["BLEU_after"] = 0.0

    # 计算ROUGE
    rouge_before = rouge_scorer.score(reference, candidate_before)
    rouge_after = rouge_scorer.score(reference, candidate_after)
    
    metrics.update({
        "ROUGE1_before": rouge_before["rouge1"].fmeasure,
        "ROUGE2_before": rouge_before["rouge2"].fmeasure,
        "ROUGEL_before": rouge_before["rougeL"].fmeasure,
        "ROUGE1_after": rouge_after["rouge1"].fmeasure,
        "ROUGE2_after": rouge_after["rouge2"].fmeasure,
        "ROUGEL_after": rouge_after["rougeL"].fmeasure,
    })

    # 计算BERTScore（需要GPU加速）
    try:
        P_before, R_before, F_before = bert_score([candidate_before], [reference], lang="en")
        P_after, R_after, F_after = bert_score([candidate_after], [reference], lang="en")
        metrics["BERTScore_before"] = F_before.mean().item()
        metrics["BERTScore_after"] = F_after.mean().item()
    except:
        metrics["BERTScore_before"] = metrics["BERTScore_after"] = 0.0

    return pd.Series(metrics)

# 应用计算函数
result_df = df.apply(calculate_metrics, axis=1)

# 合并结果到原始数据
final_df = pd.concat([df, result_df], axis=1)

# 保存结果到新文件
final_df.to_excel("with_rag_metrics.xlsx", index=False)

print("指标计算完成，结果已保存到 nhs_test_with_metrics.xlsx")