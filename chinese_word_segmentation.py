#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
中文分词工具实现与评价
===================
本程序实现了三种中文分词算法：
1. 基于词典的最大匹配法（正向和逆向）
2. 基于统计的HMM分词法
3. 结合jieba库的高级分词方法
并对其分词结果进行评价。
"""

import re
import time
import math
import jieba
import numpy as np
from collections import Counter

class ChineseWordSegmentation:
    def __init__(self):
        """初始化分词器"""
        # 加载词典
        self.dict_path = "dict.txt"  # 假设有一个词典文件
        self.user_dict_path = "user_dict.txt"  # 用户自定义词典
        self.stopwords_path = "stopwords.txt"  # 停用词表
        
        # 初始化词典和模型
        self.word_dict = self.load_dict(self.dict_path)
        self.max_word_length = max([len(word) for word in self.word_dict]) if self.word_dict else 10
        
        # HMM模型参数
        self.states = ['B', 'M', 'E', 'S']  # B:开始, M:中间, E:结束, S:单字成词
        self.load_hmm_params()
        
        # jieba分词初始化
        jieba.initialize()
        try:
            jieba.load_userdict(self.user_dict_path)
        except:
            print("未找到用户词典，使用默认词典")
            
        # 停用词表
        self.stopwords = self.load_stopwords(self.stopwords_path)

    def load_dict(self, dict_path):
        """
        加载词典
        如果找不到词典文件，返回一个简单的默认词典
        """
        word_dict = {}
        default_dict = {
            "中国": 10000, "北京": 8000, "人民": 7000, "国家": 6000,
            "城市": 5000, "发展": 4500, "经济": 4000, "社会": 3800,
            "技术": 3500, "计算机": 3200, "软件": 3000, "互联网": 2800,
            "分词": 2500, "自然语言处理": 2200, "机器学习": 2000,
            "人工智能": 1800, "算法": 1600, "数据": 1400, "科学": 1200
        }
        
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word, freq = line.strip().split()
                    word_dict[word] = int(freq)
            return word_dict
        except:
            print(f"无法加载词典文件 {dict_path}，使用默认词典")
            return default_dict

    def load_stopwords(self, stopwords_path):
        """加载停用词表"""
        try:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                return set([line.strip() for line in f])
        except:
            print(f"无法加载停用词表 {stopwords_path}，使用空停用词表")
            return set()
            
    def load_hmm_params(self):
        """
        加载HMM模型参数，如果没有预训练的参数，使用默认值
        """
        # 初始状态概率
        self.start_p = {'B': 0.7, 'M': 0.0, 'E': 0.0, 'S': 0.3}
        
        # 转移概率
        self.trans_p = {
            'B': {'B': 0.0, 'M': 0.9, 'E': 0.1, 'S': 0.0},
            'M': {'B': 0.0, 'M': 0.8, 'E': 0.2, 'S': 0.0},
            'E': {'B': 0.7, 'M': 0.0, 'E': 0.0, 'S': 0.3},
            'S': {'B': 0.7, 'M': 0.0, 'E': 0.0, 'S': 0.3}
        }
        
        # 发射概率，这里用简单的默认值
        self.emit_p = {}
        for state in self.states:
            self.emit_p[state] = {}
            # 默认情况下每个字符属于每个状态的概率相等
            for char in "一二三四五六七八九十百千万亿中国人民北京上海":
                self.emit_p[state][char] = 0.05
                
        # 实际应用中应该根据语料库统计得到更准确的发射概率

    def forward_max_match(self, text):
        """
        正向最大匹配算法
        从左到右按最大长度匹配词典中的词
        """
        words = []
        i = 0
        while i < len(text):
            matched = False
            for j in range(min(self.max_word_length, len(text) - i), 0, -1):
                word = text[i:i+j]
                if word in self.word_dict:
                    words.append(word)
                    matched = True
                    i += j
                    break
            if not matched:  # 未匹配到词典中的词，将单字成词
                words.append(text[i])
                i += 1
        return words

    def backward_max_match(self, text):
        """
        逆向最大匹配算法
        从右到左按最大长度匹配词典中的词
        """
        words = []
        i = len(text)
        while i > 0:
            matched = False
            for j in range(min(self.max_word_length, i), 0, -1):
                word = text[i-j:i]
                if word in self.word_dict:
                    words.insert(0, word)
                    matched = True
                    i -= j
                    break
            if not matched:  # 未匹配到词典中的词，将单字成词
                words.insert(0, text[i-1])
                i -= 1
        return words
        
    def bi_direction_max_match(self, text):
        """
        双向最大匹配算法
        同时使用正向和逆向最大匹配，并选择词数更少或单字更少的结果
        """
        forward_words = self.forward_max_match(text)
        backward_words = self.backward_max_match(text)
        
        if len(forward_words) != len(backward_words):
            return forward_words if len(forward_words) < len(backward_words) else backward_words
        else:
            # 如果词数相同，返回单字成词较少的那个
            forward_single = sum(1 for word in forward_words if len(word) == 1)
            backward_single = sum(1 for word in backward_words if len(word) == 1)
            return forward_words if forward_single < backward_single else backward_words

    def hmm_segment(self, text):
        """
        基于HMM的分词方法
        使用Viterbi算法计算最可能的标注序列
        """
        if not text:
            return []
            
        # 处理长文本，分段进行
        if len(text) > 1000:
            # 按标点符号或空格分段
            segments = re.split(r'([，。！？；：、\s]+)', text)
            result = []
            for i in range(0, len(segments), 2):
                seg = segments[i]
                if seg:
                    result.extend(self.hmm_segment(seg))
                if i + 1 < len(segments):
                    result.append(segments[i+1])
            return result
        
        # Viterbi算法
        V = [{}]  # 存储路径概率
        path = {}  # 存储路径
        
        # 初始化
        for state in self.states:
            V[0][state] = self.start_p[state] * self.emit_p.get(state, {}).get(text[0], 0.01)
            path[state] = [state]
            
        # 递推
        for t in range(1, len(text)):
            V.append({})
            new_path = {}
            
            for curr_state in self.states:
                # 计算从各个前一状态转移过来的概率
                probs = [
                    (V[t-1][prev_state] * 
                     self.trans_p.get(prev_state, {}).get(curr_state, 0.01) * 
                     self.emit_p.get(curr_state, {}).get(text[t], 0.01), 
                     prev_state)
                    for prev_state in self.states
                ]
                max_prob, max_state = max(probs)
                
                V[t][curr_state] = max_prob
                new_path[curr_state] = path[max_state] + [curr_state]
                
            path = new_path
            
        # 找出最可能的状态序列
        (max_prob, max_state) = max([(V[len(text) - 1][state], state) for state in self.states])
        best_path = path[max_state]
        
        # 根据状态序列进行分词
        words = []
        word = ""
        for i, char in enumerate(text):
            if best_path[i] == 'B':  # 词开始
                word = char
            elif best_path[i] == 'M':  # 词中间
                word += char
            elif best_path[i] == 'E':  # 词结束
                word += char
                words.append(word)
                word = ""
            else:  # 'S'，单字成词
                words.append(char)
        
        # 处理最后可能未添加的词
        if word:
            words.append(word)
            
        return words

    def jieba_segment(self, text, mode="default"):
        """
        使用jieba分词
        mode: default, cut_all, search, HMM
        """
        if mode == "default":
            return list(jieba.cut(text, cut_all=False))
        elif mode == "cut_all":
            return list(jieba.cut(text, cut_all=True))
        elif mode == "search":
            return list(jieba.cut_for_search(text))
        elif mode == "HMM":
            return list(jieba.cut(text, cut_all=False, HMM=True))
        else:
            return list(jieba.cut(text))

    def remove_stopwords(self, words):
        """移除停用词"""
        return [word for word in words if word not in self.stopwords]

    def segment(self, text, method="all", remove_stopwords=False):
        """
        分词主函数
        method: forward, backward, bidirection, hmm, jieba, all
        """
        # 预处理文本
        text = re.sub(r'\s+', ' ', text)  # 规范化空白符
        
        results = {}
        
        if method == "forward" or method == "all":
            results["forward_max_match"] = self.forward_max_match(text)
            
        if method == "backward" or method == "all":
            results["backward_max_match"] = self.backward_max_match(text)
            
        if method == "bidirection" or method == "all":
            results["bidirection_max_match"] = self.bi_direction_max_match(text)
            
        if method == "hmm" or method == "all":
            results["hmm"] = self.hmm_segment(text)
            
        if method == "jieba" or method == "all":
            results["jieba_default"] = self.jieba_segment(text, "default")
            results["jieba_search"] = self.jieba_segment(text, "search")
            
        # 去除停用词
        if remove_stopwords:
            for key in results:
                results[key] = self.remove_stopwords(results[key])
                
        return results

    def evaluate(self, segmented, gold_standard):
        """
        评估分词结果
        segmented: 分词结果
        gold_standard: 标准分词结果（人工标注）
        返回精确率、召回率和F1值
        """
        # 将分词结果转换为字符位置集合
        def get_positions(words):
            positions = set()
            offset = 0
            for word in words:
                positions.add((offset, offset + len(word)))
                offset += len(word)
            return positions
            
        segmented_positions = get_positions(segmented)
        gold_positions = get_positions(gold_standard)
        
        # 计算指标
        true_positives = len(segmented_positions & gold_positions)
        if not segmented_positions:
            precision = 0
        else:
            precision = true_positives / len(segmented_positions)
            
        if not gold_positions:
            recall = 0
        else:
            recall = true_positives / len(gold_positions)
            
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
            
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def evaluate_all_methods(self, text, gold_standard):
        """评估所有分词方法的结果"""
        results = self.segment(text, method="all")
        evaluation = {}
        
        for method, result in results.items():
            evaluation[method] = self.evaluate(result, gold_standard)
            
        return evaluation

    def compare_methods(self, texts, gold_standards):
        """比较不同方法在多个文本上的性能"""
        all_evaluations = []
        
        for text, gold in zip(texts, gold_standards):
            evaluation = self.evaluate_all_methods(text, gold)
            all_evaluations.append(evaluation)
            
        # 计算平均性能
        avg_performance = {}
        for method in all_evaluations[0].keys():
            avg_performance[method] = {
                "precision": np.mean([eval[method]["precision"] for eval in all_evaluations]),
                "recall": np.mean([eval[method]["recall"] for eval in all_evaluations]),
                "f1": np.mean([eval[method]["f1"] for eval in all_evaluations])
            }
            
        return avg_performance

    def segment_file(self, input_file, output_file, method="bidirection"):
        """对文件进行分词"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f_in:
                text = f_in.read()
                
            results = self.segment(text, method=method)
            
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for method_name, words in results.items():
                    f_out.write(f"Method: {method_name}\n")
                    f_out.write(' '.join(words))
                    f_out.write('\n\n')
                    
            return True
        except Exception as e:
            print(f"分词文件时出错: {e}")
            return False

    def benchmark(self, text, repeat=5):
        """性能基准测试"""
        methods = ["forward", "backward", "bidirection", "hmm", "jieba"]
        times = {}
        
        for method in methods:
            start_time = time.time()
            for _ in range(repeat):
                self.segment(text, method=method)
            elapsed = (time.time() - start_time) / repeat
            times[method] = elapsed
            
        return times


def main():
    """主函数，演示分词工具的使用和评价"""
    segmenter = ChineseWordSegmentation()
    
    # 示例文本
    texts = [
        "我爱北京天安门",
        "中华人民共和国成立于一九四九年",
        "自然语言处理是人工智能的一个重要分支",
        "高科技计算机网络信息技术的飞速发展促进了社会经济的腾飞"
    ]
    
    # 标准分词结果（假设的人工标注）
    gold_standards = [
        ["我", "爱", "北京", "天安门"],
        ["中华人民共和国", "成立", "于", "一九四九年"],
        ["自然语言处理", "是", "人工智能", "的", "一个", "重要", "分支"],
        ["高科技", "计算机", "网络", "信息", "技术", "的", "飞速", "发展", "促进", "了", "社会经济", "的", "腾飞"]
    ]
    
    # 1. 单个文本的分词结果展示
    print("=== 单个文本的分词结果 ===")
    text = texts[2]
    results = segmenter.segment(text, method="all")
    
    for method, words in results.items():
        print(f"{method}: {' '.join(words)}")
    print()
    
    # 2. 评估分词结果
    print("=== 分词结果评估 ===")
    evaluation = segmenter.evaluate_all_methods(text, gold_standards[2])
    
    for method, metrics in evaluation.items():
        print(f"{method}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
    print()
    
    # 3. 比较不同方法在多个文本上的性能
    print("=== 多文本分词方法比较 ===")
    avg_performance = segmenter.compare_methods(texts, gold_standards)
    
    for method, metrics in avg_performance.items():
        print(f"{method}:")
        print(f"  Avg. Precision: {metrics['precision']:.4f}")
        print(f"  Avg. Recall: {metrics['recall']:.4f}")
        print(f"  Avg. F1: {metrics['f1']:.4f}")
    print()
    
    # 4. 性能基准测试
    print("=== 性能基准测试 ===")
    benchmark_text = "".join(texts)
    times = segmenter.benchmark(benchmark_text)
    
    for method, elapsed in times.items():
        print(f"{method}: {elapsed:.6f} 秒")


if __name__ == "__main__":
    main()