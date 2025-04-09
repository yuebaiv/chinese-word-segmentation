#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
中文分词可视化工具
==============
用于可视化展示不同分词方法的结果对比，
以及错误分析和分词结果差异。
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.font_manager import FontProperties
import json
import os

# 尝试设置中文字体
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except:
    print("警告：未找到合适的中文字体，图表中的中文可能无法正确显示")
    
class SegmentationVisualizer:
    def __init__(self, output_dir="visualization_output"):
        """初始化可视化工具"""
        self.output_dir = output_dir
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 设置颜色方案
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
            
    def load_results(self, results_file):
        """
        加载评估结果
        results_file: 评估结果JSON文件路径
        """
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载结果文件失败: {e}")
            return None
            
    def plot_performance_comparison(self, results, save=True):
        """
        绘制各方法性能对比图
        results: 评估结果
        save: 是否保存图像
        """
        plt.figure(figsize=(12, 8))
        
        # 准备数据
        methods = list(results["method_performance"].keys())
        precision_values = [results["method_performance"][method]["avg_precision"] for method in methods]
        recall_values = [results["method_performance"][method]["avg_recall"] for method in methods]
        f1_values = [results["method_performance"][method]["avg_f1"] for method in methods]
        
        # 按F1值降序排序
        sorted_indices = np.argsort(f1_values)[::-1]
        methods = [methods[i] for i in sorted_indices]
        precision_values = [precision_values[i] for i in sorted_indices]
        recall_values = [recall_values[i] for i in sorted_indices]
        f1_values = [f1_values[i] for i in sorted_indices]
        
        # 绘制柱状图
        x = np.arange(len(methods))
        width = 0.25
        
        plt.bar(x - width, precision_values, width, label='精确率', color=self.colors[0])
        plt.bar(x, recall_values, width, label='召回率', color=self.colors[1])
        plt.bar(x + width, f1_values, width, label='F1值', color=self.colors[2])
        
        # 添加数值标签
        for i, value in enumerate(precision_values):
            plt.text(i - width, value + 0.01, f'{value:.4f}', ha='center', va='bottom', fontsize=8)
            
        for i, value in enumerate(recall_values):
            plt.text(i, value + 0.01, f'{value:.4f}', ha='center', va='bottom', fontsize=8)
            
        for i, value in enumerate(f1_values):
            plt.text(i + width, value + 0.01, f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('分词方法')
        plt.ylabel('得分')
        plt.title('各分词方法性能对比')
        plt.xticks(x, methods, rotation=45)
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存图像
        if save:
            plt.savefig(os.path.join(self.output_dir, 'performance_comparison.png'), dpi=300)
            plt.savefig(os.path.join(self.output_dir, 'performance_comparison.pdf'))
            print(f"性能对比图已保存到 {self.output_dir}")
            
        plt.close()
        
    def plot_speed_comparison(self, results, save=True):
        """
        绘制各方法速度对比图
        results: 评估结果
        save: 是否保存图像
        """
        plt.figure(figsize=(10, 6))
        
        # 准备数据
        methods = list(results["speed_benchmark"].keys())
        times = [results["speed_benchmark"][method] for method in methods]
        
        # 按时间升序排序
        sorted_indices = np.argsort(times)
        methods = [methods[i] for i in sorted_indices]
        times = [times[i] for i in sorted_indices]
        
        # 绘制水平柱状图
        bars = plt.barh(methods, times, color=self.colors[3])
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2, 
                    f'{times[i]:.6f}s', ha='left', va='center')
        
        plt.xlabel('平均处理时间(秒)')
        plt.ylabel('分词方法')
        plt.title('各分词方法速度对比')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存图像
        if save:
            plt.savefig(os.path.join(self.output_dir, 'speed_comparison.png'), dpi=300)
            plt.savefig(os.path.join(self.output_dir, 'speed_comparison.pdf'))
            print(f"速度对比图已保存到 {self.output_dir}")
            
        plt.close()
        
    def plot_f1_by_text(self, results, save=True):
        """
        绘制各方法在不同文本上的F1值
        results: 评估结果
        save: 是否保存图像
        """
        plt.figure(figsize=(14, 8))
        
        # 准备数据
        methods = list(results["method_performance"].keys())
        text_ids = [f"文本{i+1}" for i in range(len(results["individual_results"]))]
        
        method_f1_values = {}
        for method in methods:
            method_f1_values[method] = []
            
        for result in results["individual_results"]:
            for method in methods:
                method_f1_values[method].append(result["evaluation"][method]["f1"])
                
        # 按平均F1值降序排序
        avg_f1_values = [results["method_performance"][method]["avg_f1"] for method in methods]
        sorted_indices = np.argsort(avg_f1_values)[::-1]
        methods = [methods[i] for i in sorted_indices]
        
        # 绘制折线图
        for i, method in enumerate(methods):
            plt.plot(text_ids, method_f1_values[method], marker='o', label=method, color=self.colors[i % len(self.colors)])
            
        plt.xlabel('测试文本')
        plt.ylabel('F1值')
        plt.title('各方法在不同文本上的F1值')
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存图像
        if save:
            plt.savefig(os.path.join(self.output_dir, 'f1_by_text.png'), dpi=300)
            plt.savefig(os.path.join(self.output_dir, 'f1_by_text.pdf'))
            print(f"各文本F1值对比图已保存到 {self.output_dir}")
            
        plt.close()
        
    def plot_method_stability(self, results, save=True):
        """
        绘制各方法稳定性对比图（箱线图）
        results: 评估结果
        save: 是否保存图像
        """
        plt.figure(figsize=(12, 8))
        
        # 准备数据
        methods = list(results["method_performance"].keys())
        method_f1_values = []
        
        for method in methods:
            f1_values = []
            for result in results["individual_results"]:
                f1_values.append(result["evaluation"][method]["f1"])
            method_f1_values.append(f1_values)
            
        # 按平均F1值降序排序
        avg_f1_values = [results["method_performance"][method]["avg_f1"] for method in methods]
        sorted_indices = np.argsort(avg_f1_values)[::-1]
        methods = [methods[i] for i in sorted_indices]
        method_f1_values = [method_f1_values[i] for i in sorted_indices]
            
        # 绘制箱线图
        box = plt.boxplot(method_f1_values, patch_artist=True, labels=methods)
        
        # 设置颜色
        for i, patch in enumerate(box['boxes']):
            patch.set_facecolor(self.colors[i % len(self.colors)])
            
        plt.xlabel('分词方法')
        plt.ylabel('F1值')
        plt.title('各分词方法稳定性对比')
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存图像
        if save:
            plt.savefig(os.path.join(self.output_dir, 'method_stability.png'), dpi=300)
            plt.savefig(os.path.join(self.output_dir, 'method_stability.pdf'))
            print(f"方法稳定性对比图已保存到 {self.output_dir}")
            
        plt.close()
        
    def plot_precision_recall_curve(self, results, save=True):
        """
        绘制精确率-召回率曲线
        results: 评估结果
        save: 是否保存图像
        """
        plt.figure(figsize=(10, 8))
        
        # 准备数据
        methods = list(results["method_performance"].keys())
        avg_precision = [results["method_performance"][method]["avg_precision"] for method in methods]
        avg_recall = [results["method_performance"][method]["avg_recall"] for method in methods]
        avg_f1 = [results["method_performance"][method]["avg_f1"] for method in methods]
        
        # 绘制散点图
        for i, method in enumerate(methods):
            plt.scatter(avg_recall[i], avg_precision[i], s=100, label=method, color=self.colors[i % len(self.colors)])
            plt.annotate(f"{method}\nF1={avg_f1[i]:.4f}", 
                        (avg_recall[i], avg_precision[i]), 
                        xytext=(10, 10), 
                        textcoords='offset points',
                        fontsize=9)
            
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('各分词方法的精确率-召回率')
        plt.xlim(0, 1.1)
        plt.ylim(0, 1.1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # 保存图像
        if save:
            plt.savefig(os.path.join(self.output_dir, 'precision_recall.png'), dpi=300)
            plt.savefig(os.path.join(self.output_dir, 'precision_recall.pdf'))
            print(f"精确率-召回率图已保存到 {self.output_dir}")
            
        plt.close()
        
    def visualize_all(self, results):
        """
        生成所有可视化图表
        results: 评估结果
        """
        print("正在生成性能对比图...")
        self.plot_performance_comparison(results)
        
        print("正在生成速度对比图...")
        self.plot_speed_comparison(results)
        
        print("正在生成各文本F1值对比图...")
        self.plot_f1_by_text(results)
        
        print("正在生成方法稳定性对比图...")
        self.plot_method_stability(results)
        
        print("正在生成精确率-召回率图...")
        self.plot_precision_recall_curve(results)
        
        print("所有可视化图表已生成完毕！")
        
    def visualize_segmentation_diff(self, text, gold_standard, segmented_results, method_names=None, save=True):
        """
        可视化不同分词方法与标准分词的差异
        text: 原文
        gold_standard: 标准分词结果
        segmented_results: 不同方法的分词结果
        method_names: 方法名称列表
        save: 是否保存图像
        """
        if method_names is None:
            method_names = [f"方法{i+1}" for i in range(len(segmented_results))]
            
        # 计算分词边界
        def get_boundaries(words):
            boundaries = [0]
            for word in words:
                boundaries.append(boundaries[-1] + len(word))
            return boundaries[1:]
            
        gold_boundaries = set(get_boundaries(gold_standard))
        
        # 准备图形
        n_methods = len(segmented_results)
        fig, axes = plt.subplots(n_methods+1, 1, figsize=(15, 2*(n_methods+1)), sharex=True)
        
        # 绘制标准分词
        ax = axes[0]
        ax.set_title("标准分词", fontsize=12)
        ax.text(0, 0.5, " | ".join(gold_standard), fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, len(text)+1)
        ax.set_yticks([])
        
        # 为标准分词的每个分词点添加垂直线
        for boundary in gold_boundaries:
            ax.axvline(x=boundary, color='green', linestyle='-', alpha=0.5)
            
        # 绘制每种方法的分词结果
        for i, (method_result, method_name) in enumerate(zip(segmented_results, method_names)):
            ax = axes[i+1]
            ax.set_title(method_name, fontsize=12)
            
            # 获取该方法的分词边界
            method_boundaries = set(get_boundaries(method_result))
            
            # 绘制分词结果文本
            ax.text(0, 0.5, " | ".join(method_result), fontsize=10)
            
            # 标记分词点
            for boundary in method_boundaries:
                if boundary in gold_boundaries:
                    # 正确的分词点
                    ax.axvline(x=boundary, color='green', linestyle='-', alpha=0.5)
                else:
                    # 错误的分词点
                    ax.axvline(x=boundary, color='red', linestyle='-', alpha=0.5)
                    
            # 标记缺失的分词点
            for boundary in gold_boundaries:
                if boundary not in method_boundaries:
                    # 缺失的分词点
                    ax.axvline(x=boundary, color='blue', linestyle='--', alpha=0.5)
            
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            
        # 添加图例
        custom_lines = [
            plt.Line2D([0], [0], color='green', lw=2),
            plt.Line2D([0], [0], color='red', lw=2),
            plt.Line2D([0], [0], color='blue', linestyle='--', lw=2)
        ]
        fig.legend(custom_lines, ['正确分词点', '错误分词点', '缺失分词点'], 
                  loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        # 保存图像
        if save:
            plt.savefig(os.path.join(self.output_dir, 'segmentation_diff.png'), dpi=300)
            plt.savefig(os.path.join(self.output_dir, 'segmentation_diff.pdf'))
            print(f"分词差异图已保存到 {self.output_dir}")
            
        plt.close()
        
    def generate_interactive_html(self, results, output_file="segmentation_report.html"):
        """
        生成交互式HTML报告
        results: 评估结果
        output_file: 输出文件路径
        """
        try:
            # 准备数据
            methods = list(results["method_performance"].keys())
            avg_precision = [results["method_performance"][method]["avg_precision"] for method in methods]
            avg_recall = [results["method_performance"][method]["avg_recall"] for method in methods]
            avg_f1 = [results["method_performance"][method]["avg_f1"] for method in methods]
            
            # 处理速度
            speed_methods = list(results["speed_benchmark"].keys())
            speeds = [results["speed_benchmark"][method] for method in speed_methods]
            
            # 生成HTML内容
            html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>中文分词评估报告</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1, h2, h3 {
            color: #333;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .method-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .method-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        .method-name {
            font-size: 18px;
            font-weight: bold;
        }
        .method-metrics {
            display: flex;
            gap: 15px;
        }
        .metric {
            text-align: center;
        }
        .metric-value {
            font-size: 20px;
            font-weight: bold;
        }
        .metric-label {
            font-size: 12px;
            color: #666;
        }
        .text-segment {
            margin-bottom: 30px;
            border-bottom: 1px solid #eee;
            padding-bottom: 20px;
        }
        .segment-comparison {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }
        .word {
            border: 1px solid #ddd;
            padding: 3px 8px;
            border-radius: 3px;
            display: inline-block;
            margin-right: 5px;
            margin-bottom: 5px;
        }
        .correct {
            background-color: #d4edda;
            border-color: #c3e6cb;
        }
        .incorrect {
            background-color: #f8d7da;
            border-color: #f5c6cb;
        }
        .speed-bar {
            height: 20px;
            background-color: #007bff;
            margin-bottom: 5px;
        }
        .tab {
            overflow: hidden;
            border: 1px solid #ccc;
            background-color: #f1f1f1;
        }
        .tab button {
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 16px;
            transition: 0.3s;
        }
        .tab button:hover {
            background-color: #ddd;
        }
        .tab button.active {
            background-color: #ccc;
        }
        .tabcontent {
            display: none;
            padding: 6px 12px;
            border: 1px solid #ccc;
            border-top: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>中文分词评估报告</h1>
        
        <div class="tab">
            <button class="tablinks active" onclick="openTab(event, 'Summary')">总体性能</button>
            <button class="tablinks" onclick="openTab(event, 'Details')">详细评估</button>
            <button class="tablinks" onclick="openTab(event, 'TextComparison')">文本对比</button>
        </div>
        
        <div id="Summary" class="tabcontent" style="display:block;">
            <h2>总体性能</h2>
            <table>
                <tr>
                    <th>方法</th>
                    <th>平均精确率</th>
                    <th>平均召回率</th>
                    <th>平均F1值</th>
                    <th>F1标准差</th>
                    <th>处理时间(秒)</th>
                </tr>
"""
            
            # 添加方法性能数据
            for method in methods:
                perf = results["method_performance"][method]
                speed = results["speed_benchmark"].get(method, 0)
                
                html_content += f"""
                <tr>
                    <td>{method}</td>
                    <td>{perf['avg_precision']:.4f}</td>
                    <td>{perf['avg_recall']:.4f}</td>
                    <td>{perf['avg_f1']:.4f}</td>
                    <td>{perf['std_f1']:.4f}</td>
                    <td>{speed:.6f}</td>
                </tr>"""
                
            html_content += """
            </table>
            
            <h3>速度对比</h3>
            <div style="width: 100%;">
"""
            
            # 添加速度条形图
            max_speed = max(speeds)
            for i, method in enumerate(speed_methods):
                speed = results["speed_benchmark"][method]
                width_percent = (speed / max_speed) * 100
                
                html_content += f"""
                <div style="margin-bottom: 15px;">
                    <div>{method}: {speed:.6f}秒</div>
                    <div class="speed-bar" style="width: {width_percent}%;"></div>
                </div>"""
                
            html_content += """
            </div>
        </div>
        
        <div id="Details" class="tabcontent">
            <h2>详细评估</h2>
"""
            
            # 添加各方法详细卡片
            for i, method in enumerate(methods):
                perf = results["method_performance"][method]
                speed = results["speed_benchmark"].get(method, 0)
                
                html_content += f"""
            <div class="method-card">
                <div class="method-header">
                    <div class="method-name">{method}</div>
                    <div class="method-metrics">
                        <div class="metric">
                            <div class="metric-value">{perf['avg_precision']:.4f}</div>
                            <div class="metric-label">精确率</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{perf['avg_recall']:.4f}</div>
                            <div class="metric-label">召回率</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{perf['avg_f1']:.4f}</div>
                            <div class="metric-label">F1值</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{speed:.6f}s</div>
                            <div class="metric-label">处理时间</div>
                        </div>
                    </div>
                </div>
                <div>
                    <h4>各文本F1值</h4>
                    <table>
                        <tr>
                            <th>文本ID</th>
                            <th>F1值</th>
                        </tr>"""
                
                # 添加各文本F1值
                for j, result in enumerate(results["individual_results"]):
                    f1 = result["evaluation"][method]["f1"]
                    html_content += f"""
                        <tr>
                            <td>文本{j+1}</td>
                            <td>{f1:.4f}</td>
                        </tr>"""
                    
                html_content += """
                    </table>
                </div>
            </div>"""
            
            html_content += """
        </div>
        
        <div id="TextComparison" class="tabcontent">
            <h2>文本分词对比</h2>
"""
            
            # 添加各文本分词对比
            for i, result in enumerate(results["individual_results"]):
                text = result["text"]
                gold = result["gold_standard"]
                
                html_content += f"""
            <div class="text-segment">
                <h3>文本{i+1}</h3>
                <p><strong>原文:</strong> {text}</p>
                <p><strong>标准分词:</strong> {'|'.join(gold)}</p>
                
                <h4>各方法分词结果</h4>"""
                
                for method in methods:
                    segments = result["segmented_results"][method]
                    html_content += f"""
                <div>
                    <p><strong>{method}:</strong></p>
                    <div class="segment-comparison">"""
                    
                    # 比较分词结果，标记正确和错误的分词
                    for word in segments:
                        if word in gold:
                            html_content += f'<span class="word correct">{word}</span>'
                        else:
                            html_content += f'<span class="word incorrect">{word}</span>'
                            
                    html_content += """
                    </div>
                </div>"""
                    
                html_content += """
            </div>"""
                
            html_content += """
        </div>
        
        <script>
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }
        </script>
    </div>
</body>
</html>"""
            
            # 保存HTML报告
            with open(os.path.join(self.output_dir, output_file), 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            print(f"交互式HTML报告已保存到 {os.path.join(self.output_dir, output_file)}")
            return True
        except Exception as e:
            print(f"生成HTML报告失败: {e}")
            return False


def main():
    """主函数，演示可视化工具的使用"""
    # 创建可视化工具
    visualizer = SegmentationVisualizer(output_dir="visualization")
    
    # 加载评估结果
    results = visualizer.load_results("evaluation_results.json")
    
    if results:
        # 生成所有可视化图表
        visualizer.visualize_all(results)
        
        # 生成交互式HTML报告
        visualizer.generate_interactive_html(results)
        
        # 可视化特定文本的分词差异
        if results["individual_results"]:
            first_result = results["individual_results"][0]
            text = first_result["text"]
            gold_standard = first_result["gold_standard"]
            
            # 选择几种方法进行对比
            methods = list(first_result["segmented_results"].keys())
            if len(methods) > 3:
                methods = methods[:3]  # 只对比前三种方法
                
            segmented_results = [first_result["segmented_results"][method] for method in methods]
            
            visualizer.visualize_segmentation_diff(text, gold_standard, segmented_results, method_names=methods)
    else:
        print("无法加载评估结果，请先运行评估程序")


if __name__ == "__main__":
    main()