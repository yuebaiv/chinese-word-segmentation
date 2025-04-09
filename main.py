#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
中文分词工具主程序
==============
本程序整合了所有分词模块，提供命令行界面，
便于用户使用和评测不同的分词方法。
"""

import os
import sys
import time
import argparse
import json
from chinese_word_segmentation import ChineseWordSegmentation
from test_and_evaluation import SegmentationEvaluator
from visualization_utils import SegmentationVisualizer
from hmm_model import HMMSegmentationTrainer

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description='中文分词工具')
    
    # 创建子命令
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # segment子命令 - 对文本进行分词
    segment_parser = subparsers.add_parser('segment', help='对文本进行分词')
    segment_parser.add_argument('--input', '-i', required=True, help='输入文件路径或文本')
    segment_parser.add_argument('--output', '-o', help='输出文件路径')
    segment_parser.add_argument('--method', '-m', default='all', 
                              choices=['forward', 'backward', 'bidirection', 'hmm', 'jieba', 'all'], 
                              help='分词方法')
    segment_parser.add_argument('--dict', '-d', help='自定义词典路径')
    segment_parser.add_argument('--hmm-model', help='HMM模型路径')
    segment_parser.add_argument('--remove-stopwords', action='store_true', help='是否去除停用词')
    
    # evaluate子命令 - 评估分词效果
    evaluate_parser = subparsers.add_parser('evaluate', help='评估分词效果')
    evaluate_parser.add_argument('--test-data', '-t', required=True, help='测试数据文件路径')
    evaluate_parser.add_argument('--gold-standard', '-g', required=True, help='标准分词结果文件路径')
    evaluate_parser.add_argument('--output-dir', '-o', default='evaluation_output', help='评估结果输出目录')
    evaluate_parser.add_argument('--visualize', '-v', action='store_true', help='是否生成可视化图表')
    
    # train子命令 - 训练HMM模型
    train_parser = subparsers.add_parser('train', help='训练HMM模型')
    train_parser.add_argument('--corpus', '-c', required=True, help='语料库文件路径')
    train_parser.add_argument('--output-model', '-o', required=True, help='模型输出路径')
    train_parser.add_argument('--smoothing', action='store_true', help='是否使用平滑')
    train_parser.add_argument('--alpha', type=float, default=0.01, help='平滑参数')
    
    # analyze子命令 - 分析错误类型
    analyze_parser = subparsers.add_parser('analyze', help='分析分词错误类型')
    analyze_parser.add_argument('--test-data', '-t', required=True, help='测试数据文件路径')
    analyze_parser.add_argument('--gold-standard', '-g', required=True, help='标准分词结果文件路径')
    analyze_parser.add_argument('--output-dir', '-o', default='analysis_output', help='分析结果输出目录')
    analyze_parser.add_argument('--hmm-model', help='HMM模型路径')
    
    # interactive子命令 - 交互式分词
    interactive_parser = subparsers.add_parser('interactive', help='交互式分词')
    interactive_parser.add_argument('--method', '-m', default='all', 
                                  choices=['forward', 'backward', 'bidirection', 'hmm', 'jieba', 'all'], 
                                  help='分词方法')
    interactive_parser.add_argument('--dict', '-d', help='自定义词典路径')
    interactive_parser.add_argument('--hmm-model', help='HMM模型路径')
    
    return parser

def process_segment_command(args):
    """处理segment子命令"""
    segmenter = ChineseWordSegmentation()
    
    # 加载自定义词典
    if args.dict and os.path.exists(args.dict):
        segmenter.word_dict = segmenter.load_dict(args.dict)
        print(f"已加载自定义词典: {args.dict}")
    
    # 加载HMM模型
    if args.hmm_model and os.path.exists(args.hmm_model):
        hmm_trainer = HMMSegmentationTrainer()
        hmm_trainer.load_model(args.hmm_model)
        segmenter.hmm_trainer = hmm_trainer
        print(f"已加载HMM模型: {args.hmm_model}")
    
    # 确定输入是文件还是文本
    if os.path.exists(args.input):
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"已读取输入文件: {args.input}")
        except Exception as e:
            print(f"读取输入文件时出错: {e}")
            return False
    else:
        text = args.input
        print("使用直接输入的文本")
    
    # 分词
    start_time = time.time()
    results = segmenter.segment(text, method=args.method, remove_stopwords=args.remove_stopwords)
    elapsed = time.time() - start_time
    
    # 输出结果
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                for method_name, words in results.items():
                    f.write(f"方法: {method_name}\n")
                    f.write(' '.join(words))
                    f.write('\n\n')
            print(f"分词结果已保存到: {args.output}")
        except Exception as e:
            print(f"保存分词结果时出错: {e}")
            return False
    else:
        for method_name, words in results.items():
            print(f"\n方法: {method_name}")
            print(' '.join(words))
    
    print(f"\n分词耗时: {elapsed:.6f} 秒")
    return True

def process_evaluate_command(args):
    """处理evaluate子命令"""
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"已创建输出目录: {args.output_dir}")
    
    # 读取测试数据
    try:
        with open(args.test_data, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"已读取测试数据: {args.test_data}")
    except Exception as e:
        print(f"读取测试数据时出错: {e}")
        return False
    
    # 读取标准分词结果
    try:
        with open(args.gold_standard, 'r', encoding='utf-8') as f:
            gold_standards = []
            for line in f:
                line = line.strip()
                if line:
                    gold_standards.append(line.split())
        print(f"已读取标准分词结果: {args.gold_standard}")
    except Exception as e:
        print(f"读取标准分词结果时出错: {e}")
        return False
    
    # 检查数据是否匹配
    if len(texts) != len(gold_standards):
        print(f"错误: 测试数据和标准分词结果数量不匹配: {len(texts)} vs {len(gold_standards)}")
        return False
    
    # 评估分词效果
    segmenter = ChineseWordSegmentation()
    evaluator = SegmentationEvaluator(segmenter)
    
    print("正在评估分词效果...")
    results = evaluator.run_comprehensive_evaluation(
        texts, 
        gold_standards, 
        save_path=os.path.join(args.output_dir, "evaluation_results.json")
    )
    
    # 生成评估报告
    print("正在生成评估报告...")
    evaluator.generate_report(results, output_file=os.path.join(args.output_dir, "evaluation_report.md"))
    
    # 生成可视化图表
    if args.visualize:
        print("正在生成可视化图表...")
        visualizer = SegmentationVisualizer(output_dir=os.path.join(args.output_dir, "visualization"))
        visualizer.visualize_all(results)
        visualizer.generate_interactive_html(results)
    
    print(f"评估完成! 结果已保存到目录: {args.output_dir}")
    return True

def process_train_command(args):
    """处理train子命令"""
    # 检查语料库文件
    if not os.path.exists(args.corpus):
        print(f"错误: 语料库文件不存在: {args.corpus}")
        return False
    
    # 训练HMM模型
    print("正在训练HMM模型...")
    trainer = HMMSegmentationTrainer()
    
    start_time = time.time()
    trainer.train(
        corpus_files=[args.corpus],
        model_path=args.output_model,
        smoothing=args.smoothing,
        alpha=args.alpha
    )
    elapsed = time.time() - start_time
    
    print(f"HMM模型训练完成! 模型已保存到: {args.output_model}")
    print(f"训练耗时: {elapsed:.6f} 秒")
    return True

def process_analyze_command(args):
    """处理analyze子命令"""
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"已创建输出目录: {args.output_dir}")
    
    # 初始化HMM模型
    trainer = HMMSegmentationTrainer()
    if args.hmm_model and os.path.exists(args.hmm_model):
        trainer.load_model(args.hmm_model)
        print(f"已加载HMM模型: {args.hmm_model}")
    else:
        print("使用默认HMM模型")
    
    # 合并测试数据和标准分词到一个文件
    temp_file = os.path.join(args.output_dir, "temp_test_data.txt")
    try:
        with open(args.test_data, 'r', encoding='utf-8') as f_test, \
             open(args.gold_standard, 'r', encoding='utf-8') as f_gold, \
             open(temp_file, 'w', encoding='utf-8') as f_out:
            
            for test_line, gold_line in zip(f_test, f_gold):
                test_line = test_line.strip()
                gold_line = gold_line.strip()
                if test_line and gold_line:
                    f_out.write(f"{gold_line}\n")
        
        print("已准备测试数据")
    except Exception as e:
        print(f"准备测试数据时出错: {e}")
        return False
    
    # 分析分词错误
    print("正在分析分词错误...")
    error_analysis = trainer.analyze_errors(
        temp_file,
        output_file=os.path.join(args.output_dir, "error_analysis.md")
    )
    
    # 保存错误分析结果为JSON
    try:
        with open(os.path.join(args.output_dir, "error_analysis.json"), 'w', encoding='utf-8') as f:
            json.dump(error_analysis, f, ensure_ascii=False, indent=2)
        print(f"错误分析结果已保存到: {args.output_dir}")
    except Exception as e:
        print(f"保存错误分析结果时出错: {e}")
    
    # 清理临时文件
    try:
        os.remove(temp_file)
    except:
        pass
    
    return True

def process_interactive_command(args):
    """处理interactive子命令"""
    segmenter = ChineseWordSegmentation()
    
    # 加载自定义词典
    if args.dict and os.path.exists(args.dict):
        segmenter.word_dict = segmenter.load_dict(args.dict)
        print(f"已加载自定义词典: {args.dict}")
    
    # 加载HMM模型
    if args.hmm_model and os.path.exists(args.hmm_model):
        hmm_trainer = HMMSegmentationTrainer()
        hmm_trainer.load_model(args.hmm_model)
        segmenter.hmm_trainer = hmm_trainer
        print(f"已加载HMM模型: {args.hmm_model}")
    
    print("\n===== 中文分词交互模式 =====")
    print("输入文本进行分词，输入'exit'或'quit'退出")
    
    while True:
        try:
            text = input("\n请输入要分词的文本: ")
            if text.lower() in ['exit', 'quit']:
                break
            
            if not text:
                continue
            
            # 分词
            start_time = time.time()
            results = segmenter.segment(text, method=args.method)
            elapsed = time.time() - start_time
            
            # 输出结果
            for method_name, words in results.items():
                print(f"\n方法: {method_name}")
                print(' '.join(words))
            
            print(f"\n分词耗时: {elapsed:.6f} 秒")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"分词时出错: {e}")
    
    print("谢谢使用!")
    return True

def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 分发子命令
    if args.command == 'segment':
        process_segment_command(args)
    elif args.command == 'evaluate':
        process_evaluate_command(args)
    elif args.command == 'train':
        process_train_command(args)
    elif args.command == 'analyze':
        process_analyze_command(args)
    elif args.command == 'interactive':
        process_interactive_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()