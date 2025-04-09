#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用于中文分词的HMM模型训练
===================
本程序实现了基于隐马尔可夫模型（HMM）的中文分词器的训练过程，
包括从语料库计算初始状态概率、转移概率和发射概率。
"""

import os
import re
import json
import math
import numpy as np
from collections import defaultdict, Counter

class HMMSegmentationTrainer:
    def __init__(self):
        """初始化HMM分词模型训练器"""
        # 状态定义
        self.states = ['B', 'M', 'E', 'S']  # B:词的开始, M:词的中间, E:词的结束, S:单字成词
        
        # 模型参数
        self.start_p = {state: 0.0 for state in self.states}  # 初始状态概率
        self.trans_p = {state: {s: 0.0 for s in self.states} for state in self.states}  # 转移概率
        self.emit_p = {state: defaultdict(float) for state in self.states}  # 发射概率
        
        # 统计计数
        self.state_count = {state: 0 for state in self.states}  # 状态出现次数
        self.transition_count = {state: {s: 0 for s in self.states} for state in self.states}  # 状态转移次数
        self.emission_count = {state: defaultdict(int) for state in self.states}  # 状态发射次数
        
        self.total_words = 0
        self.total_sentences = 0
        
    def char_tagging(self, word):
        """
        对单词进行字符标注
        返回字符及其对应的BMES标签
        """
        length = len(word)
        if length == 1:
            return [(word, 'S')]
        else:
            return [(word[0], 'B')] + [(word[i], 'M') for i in range(1, length-1)] + [(word[length-1], 'E')]
            
    def process_corpus_file(self, file_path):
        """
        处理语料库文件，统计状态转移和发射概率
        file_path: 语料库文件路径
        """
        if not os.path.exists(file_path):
            print(f"语料库文件 {file_path} 不存在")
            return False
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    self.total_sentences += 1
                    
                    # 假设语料库中的分词用空格分隔
                    words = line.split()
                    self.total_words += len(words)
                    
                    # 对句子中的每个词进行标注
                    chars_tags = []
                    for word in words:
                        chars_tags.extend(self.char_tagging(word))
                    
                    # 统计初始状态概率
                    if chars_tags:
                        self.state_count[chars_tags[0][1]] += 1
                    
                    # 统计状态转移概率和发射概率
                    for i in range(len(chars_tags)):
                        char, tag = chars_tags[i]
                        
                        # 统计状态计数
                        self.state_count[tag] += 1
                        
                        # 统计发射计数
                        self.emission_count[tag][char] += 1
                        
                        # 统计转移计数
                        if i > 0:
                            prev_tag = chars_tags[i-1][1]
                            self.transition_count[prev_tag][tag] += 1
                    
            print(f"成功处理语料库文件 {file_path}")
            print(f"总句子数: {self.total_sentences}")
            print(f"总词数: {self.total_words}")
            return True
        except Exception as e:
            print(f"处理语料库文件时出错: {e}")
            return False
            
    def calculate_probabilities(self):
        """计算HMM模型的概率参数"""
        # 计算初始状态概率
        init_total = sum(self.state_count.values())
        for state in self.states:
            if init_total > 0:
                self.start_p[state] = self.state_count[state] / init_total
            else:
                # 如果没有数据，使用均匀分布
                self.start_p[state] = 1.0 / len(self.states)
                
        # 计算转移概率
        for from_state in self.states:
            trans_sum = sum(self.transition_count[from_state].values())
            for to_state in self.states:
                if trans_sum > 0:
                    self.trans_p[from_state][to_state] = self.transition_count[from_state][to_state] / trans_sum
                else:
                    # 如果没有数据，使用均匀分布
                    self.trans_p[from_state][to_state] = 1.0 / len(self.states)
                    
        # 计算发射概率
        for state in self.states:
            emit_sum = sum(self.emission_count[state].values())
            for char, count in self.emission_count[state].items():
                if emit_sum > 0:
                    self.emit_p[state][char] = count / emit_sum
                else:
                    # 如果没有数据，则为0
                    self.emit_p[state][char] = 0.0
                    
        print("HMM模型概率参数计算完成")
        
    def add_smoothing(self, alpha=0.01):
        """
        使用加法平滑处理零概率问题
        alpha: 平滑参数
        """
        # 平滑转移概率
        for from_state in self.states:
            trans_sum = sum(self.transition_count[from_state].values())
            for to_state in self.states:
                # 添加平滑值
                self.trans_p[from_state][to_state] = (self.transition_count[from_state][to_state] + alpha) / (trans_sum + alpha * len(self.states))
                
        # 平滑发射概率
        for state in self.states:
            emit_sum = sum(self.emission_count[state].values())
            unique_chars = set()
            for s in self.states:
                unique_chars.update(self.emission_count[s].keys())
                
            for char in unique_chars:
                # 添加平滑值
                self.emit_p[state][char] = (self.emission_count[state][char] + alpha) / (emit_sum + alpha * len(unique_chars))
                
        print(f"使用加法平滑 (alpha={alpha}) 处理零概率问题")
        
    def save_model(self, model_path):
        """
        保存训练好的HMM模型参数
        model_path: 模型保存路径
        """
        model = {
            'start_p': self.start_p,
            'trans_p': self.trans_p,
            'emit_p': {state: dict(emissions) for state, emissions in self.emit_p.items()},
            'states': self.states
        }
        
        try:
            with open(model_path, 'w', encoding='utf-8') as f:
                json.dump(model, f, ensure_ascii=False, indent=2)
            print(f"HMM模型已保存到 {model_path}")
            return True
        except Exception as e:
            print(f"保存模型时出错: {e}")
            return False
            
    def load_model(self, model_path):
        """
        加载训练好的HMM模型参数
        model_path: 模型路径
        """
        try:
            with open(model_path, 'r', encoding='utf-8') as f:
                model = json.load(f)
                
            self.start_p = model['start_p']
            self.trans_p = model['trans_p']
            self.emit_p = {state: defaultdict(float, emissions) for state, emissions in model['emit_p'].items()}
            self.states = model['states']
            
            print(f"成功加载HMM模型 {model_path}")
            return True
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return False
            
    def train(self, corpus_files, model_path=None, smoothing=True, alpha=0.01):
        """
        训练HMM模型
        corpus_files: 语料库文件列表
        model_path: 模型保存路径
        smoothing: 是否使用平滑
        alpha: 平滑参数
        """
        for file_path in corpus_files:
            self.process_corpus_file(file_path)
            
        self.calculate_probabilities()
        
        if smoothing:
            self.add_smoothing(alpha)
            
        if model_path:
            self.save_model(model_path)
            
        return {
            'start_p': self.start_p,
            'trans_p': self.trans_p,
            'emit_p': {state: dict(emissions) for state, emissions in self.emit_p.items()},
            'states': self.states
        }
        
    def generate_default_model(self, model_path):
        """
        生成默认的HMM模型
        model_path: 模型保存路径
        """
        # 设置初始状态概率
        self.start_p = {'B': 0.7, 'M': 0.0, 'E': 0.0, 'S': 0.3}
        
        # 设置转移概率
        self.trans_p = {
            'B': {'B': 0.0, 'M': 0.9, 'E': 0.1, 'S': 0.0},
            'M': {'B': 0.0, 'M': 0.8, 'E': 0.2, 'S': 0.0},
            'E': {'B': 0.7, 'M': 0.0, 'E': 0.0, 'S': 0.3},
            'S': {'B': 0.7, 'M': 0.0, 'E': 0.0, 'S': 0.3}
        }
        
        # 设置一些常见汉字的发射概率
        common_chars = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名科生义听那特色记厂日来把量品众决百产南应万则动领年前目信心革通使进去证土计性林所电组路";
        for state in self.states:
            for char in common_chars:
                if state == 'B':
                    self.emit_p[state][char] = 0.02
                elif state == 'M':
                    self.emit_p[state][char] = 0.02
                elif state == 'E':
                    self.emit_p[state][char] = 0.02
                else:  # 'S'
                    self.emit_p[state][char] = 0.02
                    
        # 保存模型
        return self.save_model(model_path)
        
    def segment(self, text):
        """
        使用训练好的HMM模型进行分词
        text: 待分词文本
        """
        if not text:
            return []
            
        # Viterbi算法
        V = [{}]  # 存储路径概率
        path = {}  # 存储路径
        
        # 初始化
        for state in self.states:
            V[0][state] = self.start_p[state] * self.emit_p[state].get(text[0], 0.00001)
            path[state] = [state]
            
        # 递推
        for t in range(1, len(text)):
            V.append({})
            new_path = {}
            
            for curr_state in self.states:
                # 计算从各个前一状态转移过来的概率
                probs = [
                    (V[t-1][prev_state] * 
                     self.trans_p[prev_state][curr_state] * 
                     self.emit_p[curr_state].get(text[t], 0.00001), 
                     prev_state)
                    for prev_state in self.states
                ]
                max_prob, max_state = max(probs)
                
                V[t][curr_state] = max_prob
                new_path[curr_state] = path[max_state] + [curr_state]
                
            path = new_path
            
        # 找出最可能的状态序列
        prob_path = {}
        for state in self.states:
            prob_path[state] = V[len(text) - 1][state]
            
        (max_prob, max_state) = max([(prob_path[state], state) for state in self.states])
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
        
    def evaluate(self, test_file, output_file=None):
        """
        评估HMM分词器在测试集上的性能
        test_file: 测试文件路径
        output_file: 分词结果输出文件路径
        """
        if not os.path.exists(test_file):
            print(f"测试文件 {test_file} 不存在")
            return None
            
        try:
            correct_count = 0
            total_gold = 0
            total_pred = 0
            
            results = []
            
            with open(test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # 提取原文本和标准分词
                    gold_words = line.split()
                    text = ''.join(gold_words)
                    
                    # HMM分词
                    pred_words = self.segment(text)
                    
                    # 计算性能指标
                    gold_boundaries = set(self.get_boundaries(gold_words))
                    pred_boundaries = set(self.get_boundaries(pred_words))
                    
                    correct = len(gold_boundaries & pred_boundaries)
                    correct_count += correct
                    total_gold += len(gold_boundaries)
                    total_pred += len(pred_boundaries)
                    
                    # 记录结果
                    results.append({
                        'text': text,
                        'gold': gold_words,
                        'pred': pred_words,
                        'correct': correct,
                        'total_gold': len(gold_boundaries),
                        'total_pred': len(pred_boundaries)
                    })
            
            # 计算总体性能
            if total_pred == 0 or total_gold == 0:
                precision = recall = f1 = 0
            else:
                precision = correct_count / total_pred
                recall = correct_count / total_gold
                f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                
                # 输出性能指标
                print(f"HMM分词器在测试集上的性能:")
                print(f"精确率: {precision:.4f}")
                print(f"召回率: {recall:.4f}")
                print(f"F1值: {f1:.4f}")
                
                # 保存分词结果
                if output_file:
                    with open(output_file, 'w', encoding='utf-8') as f_out:
                        for result in results:
                            f_out.write(f"原文: {result['text']}\n")
                            f_out.write(f"标准分词: {' '.join(result['gold'])}\n")
                            f_out.write(f"HMM分词: {' '.join(result['pred'])}\n")
                            f_out.write(f"正确数: {result['correct']}, 标准词数: {result['total_gold']}, 预测词数: {result['total_pred']}\n")
                            f_out.write("\n")
                            
                    print(f"分词结果已保存到 {output_file}")
                
                return {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'correct_count': correct_count,
                    'total_gold': total_gold,
                    'total_pred': total_pred,
                    'results': results
                }
                
        except Exception as e:
            print(f"评估分词器时出错: {e}")
            return None
            
    def get_boundaries(self, words):
        """获取分词边界位置"""
        boundaries = []
        offset = 0
        for word in words:
            offset += len(word)
            boundaries.append(offset)
        return boundaries[:-1]  # 去掉最后一个边界（文本结尾）
        
    def analyze_errors(self, test_file, output_file=None):
        """
        分析分词错误类型
        test_file: 测试文件路径
        output_file: 错误分析输出文件路径
        """
        eval_result = self.evaluate(test_file)
        if not eval_result:
            return None
            
        error_types = {
            'split_error': 0,  # 将一个词错误地分成多个词
            'merge_error': 0,  # 将多个词错误地合并成一个词
            'both_error': 0,   # 既有分割错误又有合并错误
            'other_error': 0    # 其他错误
        }
        
        error_examples = {
            'split_error': [],
            'merge_error': [],
            'both_error': [],
            'other_error': []
        }
        
        for result in eval_result['results']:
            gold_words = result['gold']
            pred_words = result['pred']
            
            if result['correct'] < result['total_gold']:  # 有错误
                # 分析错误类型
                gold_boundaries = set(self.get_boundaries(gold_words))
                pred_boundaries = set(self.get_boundaries(pred_words))
                
                missed_boundaries = gold_boundaries - pred_boundaries
                extra_boundaries = pred_boundaries - gold_boundaries
                
                if missed_boundaries and not extra_boundaries:
                    error_types['merge_error'] += 1
                    if len(error_examples['merge_error']) < 10:
                        error_examples['merge_error'].append({
                            'text': result['text'],
                            'gold': gold_words,
                            'pred': pred_words
                        })
                elif extra_boundaries and not missed_boundaries:
                    error_types['split_error'] += 1
                    if len(error_examples['split_error']) < 10:
                        error_examples['split_error'].append({
                            'text': result['text'],
                            'gold': gold_words,
                            'pred': pred_words
                        })
                elif extra_boundaries and missed_boundaries:
                    error_types['both_error'] += 1
                    if len(error_examples['both_error']) < 10:
                        error_examples['both_error'].append({
                            'text': result['text'],
                            'gold': gold_words,
                            'pred': pred_words
                        })
                else:
                    error_types['other_error'] += 1
                    if len(error_examples['other_error']) < 10:
                        error_examples['other_error'].append({
                            'text': result['text'],
                            'gold': gold_words,
                            'pred': pred_words
                        })
                        
        # 输出错误分析
        total_errors = sum(error_types.values())
        print("\n错误类型分析:")
        for error_type, count in error_types.items():
            percentage = (count / total_errors * 100) if total_errors > 0 else 0
            print(f"{error_type}: {count} ({percentage:.2f}%)")
            
        # 保存错误分析结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# HMM分词器错误分析\n\n")
                
                f.write("## 总体性能\n\n")
                f.write(f"精确率: {eval_result['precision']:.4f}\n")
                f.write(f"召回率: {eval_result['recall']:.4f}\n")
                f.write(f"F1值: {eval_result['f1']:.4f}\n\n")
                
                f.write("## 错误类型统计\n\n")
                for error_type, count in error_types.items():
                    percentage = (count / total_errors * 100) if total_errors > 0 else 0
                    f.write(f"{error_type}: {count} ({percentage:.2f}%)\n")
                    
                f.write("\n## 错误示例\n\n")
                for error_type, examples in error_examples.items():
                    if examples:
                        f.write(f"### {error_type}\n\n")
                        for i, example in enumerate(examples):
                            f.write(f"#### 示例 {i+1}\n\n")
                            f.write(f"原文: {example['text']}\n\n")
                            f.write(f"标准分词: {' '.join(example['gold'])}\n\n")
                            f.write(f"HMM分词: {' '.join(example['pred'])}\n\n")
                            
                print(f"错误分析结果已保存到 {output_file}")
                
        return {
            'error_types': error_types,
            'error_examples': error_examples,
            'performance': eval_result
        }


def main():
    """主函数，演示HMM分词模型的训练和使用"""
    trainer = HMMSegmentationTrainer()
    
    # 示例1：生成默认模型
    print("\n示例1：生成默认HMM模型")
    trainer.generate_default_model("default_hmm_model.json")
    
    # 示例2：从语料库训练模型
    print("\n示例2：从语料库训练HMM模型")
    print("（注意：如果没有提供语料库文件，将使用默认模型）")
    
    corpus_file = "corpus.txt"  # 替换为实际的语料库文件路径
    if os.path.exists(corpus_file):
        trainer.train([corpus_file], "trained_hmm_model.json", smoothing=True)
    else:
        print(f"语料库文件 {corpus_file} 不存在，使用默认模型")
        trainer.load_model("default_hmm_model.json")
    
    # 示例3：使用模型进行分词
    print("\n示例3：使用HMM模型进行分词")
    test_sentences = [
        "中华人民共和国成立于一九四九年",
        "自然语言处理是人工智能的一个重要分支",
        "高科技计算机网络信息技术的飞速发展促进了社会经济的腾飞",
        "今天天气真好，我们一起去公园散步吧"
    ]
    
    for sentence in test_sentences:
        words = trainer.segment(sentence)
        print(f"原文: {sentence}")
        print(f"分词结果: {' '.join(words)}")
        print()
    
    # 示例4：评估模型性能
    print("\n示例4：评估HMM模型性能")
    test_file = "test.txt"  # 替换为实际的测试文件路径
    if os.path.exists(test_file):
        trainer.evaluate(test_file, "hmm_evaluation.txt")
        trainer.analyze_errors(test_file, "hmm_error_analysis.md")
    else:
        print(f"测试文件 {test_file} 不存在，跳过评估")


if __name__ == "__main__":
    main()