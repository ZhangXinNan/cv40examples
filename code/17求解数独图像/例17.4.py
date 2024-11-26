# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 11:41:18 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

def solveSudoku(puzzle):
    from sudoku import Sudoku             #是的，写这里了
    puzzle = Sudoku(3, 3, board=puzzle)   #初始化
    puzzle.show()                         #显示  
    solution = puzzle.solve()             #求解
    print("求解结果：")
    solution.show()                       #显示
    result = solution.board             #获取list形式
    return result                         #返回
#==================主程序=====================
puzzle=[[4, 0, 6, 0, 0, 0, 0, 9, 0],
 [0, 0, 0, 3, 1, 0, 0, 0, 6],
 [1, 0, 8, 0, 0, 7, 0, 0, 0],
 [0, 8, 0, 0, 4, 0, 6, 0, 0],
 [0, 6, 0, 7, 0, 3, 0, 2, 0],
 [0, 0, 7, 0, 9, 0, 0, 8, 0],
 [0, 0, 0, 8, 0, 0, 2, 0, 3],
 [3, 0, 0, 0, 5, 2, 0, 0, 0],
 [0, 4, 0, 0, 0, 0, 5, 0, 7]]
result=solveSudoku(puzzle)
print(result)