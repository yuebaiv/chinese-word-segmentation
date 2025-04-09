#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
中文分词评测程序
===============
本程序用于测试中文分词工具的性能和准确率，
并提供详细的评测报告。
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from chinese_word_segmentation import ChineseWordSegmentation

class SegmentationEvaluator:
    def __init__(self, segmenter):
        """初始化评估器"""
        self.segmenter = segmenter
        
    def load_test_data(self, file_path=None):
        """
        加载测试数据
        如果没有提供文件路径，则使用内置的测试数据
        """
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get('texts', []), data.get('gold_standards', [])
            except:
                print(f"无法加载测试数据文件 {file_path}，使用内置测试数据")
                
        # 内置测试数据
        texts = [
            "我爱北京天安门",
            "中华人民共和国成立于一九四九年",
            "自然语言处理是人工智能的一个重要分支",
            "高科技计算机网络信息技术的飞速发展促进了社会经济的腾飞",
            "今天天气真好，我们一起去公园散步吧",
            "这个问题我们需要深入研究才能得出结论",
            "北京大学是中国最著名的高等学府之一",
            "机器学习和深度学习是当前人工智能领域的热门技术",
            "他说：\"我们要勇往直前，不畏艰难险阻！\""
            "中国的改革开放政策促进了经济的快速发展和人民生活水平的提高"
        ]
        
        gold_standards = [
            ["我", "爱", "北京", "天安门"],
            ["中华人民共和国", "成立", "于", "一九四九年"],
            ["自然语言处理", "是", "人工智能", "的", "一个", "重要", "分支"],
            ["高科技", "计算机", "网络", "信息", "技术", "的", "飞速", "发展", "促进", "了", "社会经济", "的", "腾飞"],
            ["今天", "天气", "真", "好", "，", "我们", "一起", "去", "公园", "散步", "吧"],
            ["这个", "问题", "我们", "需要", "深入", "研究", "才", "能", "得出", "结论"],
            ["北京大学", "是", "中国", "最", "著名", "的", "高等", "学府", "之一"],
            ["机器学习", "和", "深度学习", "是", "当前", "人工智能", "领域", "的", "热门", "技术"],
            ["他", "说", "：", """, "我们", "要", "勇往直前", "，", "不畏", "艰难险阻", "！", """],
            ["中国", "的", "改革开放", "政策", "促进", "了", "经济", "的", "快速", "发展", "和", "人民", "生活", "水平", "的", "提高"]
        ]
        
        return texts, gold_standards
        
    def run_comprehensive_evaluation(self, texts=None, gold_standards=None, save_path=None):
        """
        运行全面评估
        tests: 测试文本列表
        gold_standards: 标准分词结果列表
        save_path: 保存结果的路径
        """
        if texts is None or gold_standards is None:
            texts, gold_standards = self.load_test_data()
            
        if len(texts) != len(gold_standards):
            raise ValueError("测试文本和标准分词结果数量不匹配")
            
        results = {
            "individual_results": [],
            "method_performance": {},
            "speed_benchmark": {}
        }
        
        # 1. 对每个文本进行评估
        for i, (text, gold) in enumerate(zip(texts, gold_standards)):
            print(f"正在评估文本 {i+1}/{len(texts)}...")
            
            # 分词
            segmented_results = self.segmenter.segment(text, method="all")
            
            # 计算评估指标
            evaluation = {}
            for method, words in segmented_results.items():
                evaluation[method] = self.segmenter.evaluate(words, gold)
                
            # 添加到结果
            results["individual_results"].append({
                "text": text,
                "gold_standard": gold,
                "segmented_results": {k: v for k, v in segmented_results.items()},
                "evaluation": evaluation
            })
            
        # 2. 计算各方法在所有文本上的平均性能
        method_performance = {}
        first_result = results["individual_results"][0]["evaluation"]
        
        for method in first_result.keys():
            precision_values = [result["evaluation"][method]["precision"] for result in results["individual_results"]]
            recall_values = [result["evaluation"][method]["recall"] for result in results["individual_results"]]
            f1_values = [result["evaluation"][method]["f1"] for result in results["individual_results"]]
            
            method_performance[method] = {
                "avg_precision": np.mean(precision_values),
                "avg_recall": np.mean(recall_values),
                "avg_f1": np.mean(f1_values),
                "std_precision": np.std(precision_values),
                "std_recall": np.std(recall_values),
                "std_f1": np.std(f1_values),
                "min_f1": np.min(f1_values),
                "max_f1": np.max(f1_values)
            }
            
        results["method_performance"] = method_performance
        
        # 3. 性能基准测试
        print("正在进行性能基准测试...")
        all_text = " ".join(texts)
        benchmarks = self.segmenter.benchmark(all_text, repeat=3)
        results["speed_benchmark"] = benchmarks
        
        # 4. 保存结果
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"评估结果已保存到 {save_path}")
            except Exception as e:
                print(f"保存结果失败: {e}")
                
        return results
        
    def generate_report(self, results, output_file=None):
        """
        生成评估报告
        results: 评估结果
        output_file: 输出文件路径
        """
        report = []
        
        # 1. 报告标题
        report.append("# 中文分词工具评估报告")
        report.append("\n## 1. 总体性能")
        
        # 2. 总体性能表格
        report.append("\n### 各方法平均性能（按F1值降序排列）")
        
        # 按F1值降序排序
        methods = list(results["method_performance"].keys())
        methods.sort(key=lambda x: results["method_performance"][x]["avg_f1"], reverse=True)
        
        # 表头
        report.append("\n| 方法 | 平均精确率 | 平均召回率 | 平均F1值 | F1标准差 | 最小F1值 | 最大F1值 |")
        report.append("| --- | --- | --- | --- | --- | --- | --- |")
        
        # 表格内容
        for method in methods:
            perf = results["method_performance"][method]
            report.append(f"| {method} | {perf['avg_precision']:.4f} | {perf['avg_recall']:.4f} | {perf['avg_f1']:.4f} | {perf['std_f1']:.4f} | {perf['min_f1']:.4f} | {perf['max_f1']:.4f} |")
            
        # 3. 性能基准测试
        report.append("\n### 性能基准测试")
        report.append("\n| 方法 | 平均处理时间(秒) |")
        report.append("| --- | --- |")
        
        # 按处理时间排序
        benchmark_methods = list(results["speed_benchmark"].keys())
        benchmark_methods.sort(key=lambda x: results["speed_benchmark"][x])
        
        for method in benchmark_methods:
            report.append(f"| {method} | {results['speed_benchmark'][method]:.6f} |")
            
        # 4. 各文本的详细评估结果
        report.append("\n## 2. 各文本的详细评估结果")
        
        for i, individual in enumerate(results["individual_results"]):
            report.append(f"\n### 文本 {i+1}")
            report.append(f"\n原文: {individual['text']}")
            report.append(f"\n标准分词: {' '.join(individual['gold_standard'])}")
            
            report.append("\n各方法分词结果:")
            for method, words in individual["segmented_results"].items():
                report.append(f"\n- {method}: {' '.join(words)}")
                
            report.append("\n各方法评估指标:")
            report.append("\n| 方法 | 精确率 | 召回率 | F1值 |")
            report.append("| --- | --- | --- | --- |")
            
            for method, metrics in individual["evaluation"].items():
                report.append(f"| {method} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |")
                
        # 5. 生成结论
        report.append("\n## 3. 结论")
        
        # 找出F1值最高的方法
        best_method = max(methods, key=lambda x: results["method_performance"][x]["avg_f1"])
        best_f1 = results["method_performance"][best_method]["avg_f1"]
        
        # 找出速度最快的方法
        fastest_method = min(benchmark_methods, key=lambda x: results["speed_benchmark"][x])
        fastest_time = results["speed_benchmark"][fastest_method]
        
        report.append(f"\n在本次评测中，{best_method}方法取得了最高的平均F1值({best_f1:.4f})，表现最为准确。")
        report.append(f"\n而{fastest_method}方法具有最快的处理速度，平均处理时间为{fastest_time:.6f}秒。")
        
        # 总结各方法的优缺点
        report.append("\n各方法特点总结：")
        report.append("\n1. 正向最大匹配（forward_max_match）:")
        report.append("   - 优点：算法简单，执行速度快")
        report.append("   - 缺点：容易受词典质量影响，对歧义词处理能力有限")
        
        report.append("\n2. 逆向最大匹配（backward_max_match）:")
        report.append("   - 优点：在某些情况下比正向匹配有更好的歧义处理能力")
        report.append("   - 缺点：同样受词典质量影响")
        
        report.append("\n3. 双向最大匹配（bidirection_max_match）:")
        report.append("   - 优点：结合了正向和逆向匹配的优势，歧义处理能力有所提升")
        report.append("   - 缺点：速度较慢，仍然依赖于词典质量")
        
        report.append("\n4. 基于HMM的分词（hmm）:")
        report.append("   - 优点：具有一定的语义理解能力，对未登录词有一定处理能力")
        report.append("   - 缺点：需要训练数据，参数调优复杂，速度较慢")
        
        report.append("\n5. 基于jieba的分词:")
        report.append("   - 优点：综合了多种算法，准确率高，速度快，对未登录词有较好的识别能力")
        report.append("   - 缺点：黑盒使用，难以针对特定领域进行深度优化")
        
        # 最后的建议
        report.append("\n## 4. 建议")
        report.append("\n根据评测结果，我们提出以下建议：")
        report.append("\n1. 对于追求高准确率的应用场景，建议使用jieba分词或优化后的双向最大匹配算法。")
        report.append("\n2. 对于追求高速度的场景，可以考虑使用正向最大匹配算法，但需要保证词典的质量和覆盖率。")
        report.append("\n3. 对于特定领域的应用，建议：")
        report.append("   - 扩充领域专有词典")
        report.append("   - 针对领域特点调整算法参数")
        report.append("   - 考虑结合规则和统计的混合方法")
        report.append("\n4. 未来改进方向：")
        report.append("   - 引入深度学习方法，如BiLSTM-CRF或BERT等预训练模型")
        report.append("   - 构建更大规模的训练语料")
        report.append("   - 开发更精细的评估方法，如针对不同词性或领域的分词效果评估")
        
        # 合并为完整报告
        full_report = "\n".join(report)
        
        # 保存报告
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(full_report)
                print(f"评估报告已保存到 {output_file}")
            except Exception as e:
                print(f"保存报告失败: {e}")
                
        return full_report
        
    def visualize_results(self, results, output_dir=None):
        """
        可视化评估结果
        results: 评估结果
        output_dir: 输出目录
        """
        if not plt:
            print("未安装matplotlib，无法生成可视化结果")
            return
            
        # 创建输出目录
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 1. 方法平均性能对比图
        plt.figure(figsize=(12, 8))
        
        # 准备数据
        methods = list(results["method_performance"].keys())
        precision_values = [results["method_performance"][method]["avg_precision"] for method in methods]
        recall_values = [results["method_performance"][method]["avg_recall"] for method in methods]
        f1_values = [results["method_performance"][method]["avg_f1"] for method in methods]
        
        # 绘制柱状图
        x = np.arange(len(methods))
        width = 0.25
        
        plt.bar(x - width, precision_values, width, label='精确率')
        plt.bar(x, recall_values, width, label='召回率')
        plt.bar(x + width, f1_values, width, label='F1值')
        
        plt.xlabel('分词方法')
        plt.ylabel('得分')
        plt.title('各分词方法性能对比')
        plt.xticks(x, methods, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # 保存图像
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'method_performance.png'))
            print(f"性能对比图已保存到 {os.path.join(output_dir, 'method_performance.png')}")
            
        plt.close()
        
        # 2. 速度对比图
        plt.figure(figsize=(10, 6))
        
        # 准备数据
        benchmark_methods = list(results["speed_benchmark"].keys())
        times = [results["speed_benchmark"][method] for method in benchmark_methods]
        
        # 按时间排序
        sorted_idx = np.argsort(times)
        sorted_methods = [benchmark_methods[i] for i in sorted_idx]
        sorted_times = [times[i] for i in sorted_idx]
        
        # 绘制柱状图
        plt.barh(sorted_methods, sorted_times)
        plt.xlabel('平均处理时间(秒)')
        plt.ylabel('分词方法')
        plt.title('各分词方法速度对比')
        plt.tight_layout()
        
        # 保存图像
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'speed_benchmark.png'))
            print(f"速度对比图已保存到 {os.path.join(output_dir, 'speed_benchmark.png')}")
            
        plt.close()
        
        # 3. 各文本F1值对比图
        plt.figure(figsize=(14, 8))
        
        # 准备数据
        text_ids = [f"文本{i+1}" for i in range(len(results["individual_results"]))]
        method_f1_values = {}
        
        for method in methods:
            method_f1_values[method] = []
            
        for result in results["individual_results"]:
            for method in methods:
                method_f1_values[method].append(result["evaluation"][method]["f1"])
                
        # 绘制折线图
        for method in methods:
            plt.plot(text_ids, method_f1_values[method], marker='o', label=method)
            
        plt.xlabel('文本ID')
        plt.ylabel('F1值')
        plt.title('各方法在不同文本上的F1值')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # 保存图像
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'text_f1_comparison.png'))
            print(f"F1值对比图已保存到 {os.path.join(output_dir, 'text_f1_comparison.png')}")
            
        plt.close()
        

def main():
    """主函数，演示评测程序的使用"""
    # 1. 初始化分词器和评估器
    segmenter = ChineseWordSegmentation()
    evaluator = SegmentationEvaluator(segmenter)
    
    # 2. 加载测试数据
    texts, gold_standards = evaluator.load_test_data()
    
    # 3. 运行评估
    print("开始评估分词工具性能...")
    results = evaluator.run_comprehensive_evaluation(texts, gold_standards, save_path="evaluation_results.json")
    
    # 4. 生成评估报告
    print("生成评估报告...")
    report = evaluator.generate_report(results, output_file="evaluation_report.md")
    
    # 5. 可视化结果
    print("生成可视化结果...")
    evaluator.visualize_results(results, output_dir="evaluation_figures")
    
    print("评估完成！")
    

if __name__ == "__main__":
    main()