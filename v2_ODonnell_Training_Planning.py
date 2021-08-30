#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 10:48:40 2021

@author: franciso
"""

import pulp
import numpy as np

def surface(months, Dept1_HR, Dept2_HR, Dept3_HR, Emp, T_0, Resign, TH, EH):
    """
    Using numpy arrays, builds a 3D surface model which highligts where and when employees are needed to satisfy hourly work requirements.    
    Args:
        months: list of months in which trainee hiring will be planned.
        Dept1_HR: numpy array of hours required each month in department 1.
        Dept2_HR: numpy array of hours required each month in department 2.
        Dept3_HR: numpy array of hours required each month in department 3.
        Emp: list of number of employees in each department at start of planning.
        T_0: list of number of trainees in each department one month before start of planning.
        Resign: dictionary of resignations in each department for the given month known prior to planning.
        TH: integer, hours trainee can work per month.
        EH: integer, hours employee can work per month.
    Returns:
       numpy array representing employee requirements for all departments and months.

    """
    # Builds employee array for all departments and months.
    Dept1_E = np.array([Emp[0]]).repeat(len(months))
    Dept2_E = np.array([Emp[1]]).repeat(len(months))
    Dept3_E = np.array([Emp[2]]).repeat(len(months))
    
    # Builds employee resignation array for all departments and months.
    Dept1_R = np.concatenate((np.zeros(months.index(Resign[1][0]),dtype=int), np.repeat(Resign[1][1],len(months)-months.index(Resign[1][0]))))
    Dept2_R = np.concatenate((np.zeros(months.index(Resign[2][0]),dtype=int), np.repeat(Resign[2][1],len(months)-months.index(Resign[2][0]))))
    Dept3_R = np.concatenate((np.zeros(months.index(Resign[3][0]),dtype=int), np.repeat(Resign[3][1],len(months)-months.index(Resign[3][0]))))
    
    # Builds newly hired trainee hour availability array for all departments and months.
    Dept1_T = np.array([T_0[0]]).repeat(len(months)) * EH
    Dept1_T[0]*=(TH/EH)
    Dept2_T = np.array([T_0[1]]).repeat(len(months)) * EH
    Dept2_T[0]*=(TH/EH)
    Dept3_T = np.array([T_0[2]]).repeat(len(months)) * EH
    Dept3_T[0]*=(TH/EH)
    
    # Builds employee hours availability array for all departments and months.
    Dept1_HA = EH * (Dept1_E - Dept1_R) + Dept1_T
    Dept2_HA = EH * (Dept2_E - Dept2_R) + Dept2_T
    Dept3_HA = EH * (Dept3_E - Dept3_R) + Dept3_T
    
    # Builds employee requirements array for all departments and months.
    E1 = (Dept1_HA - Dept1_HR) / EH
    E2 = (Dept2_HA - Dept2_HR) / EH
    E3 = (Dept3_HA - Dept3_HR) / EH
    
    # Flattens (zeroes) array suface where employee availability exceeds requirements because there is no cost penalty for extras employees.
    var1 = np.where(E1>0, 0, E1)
    var2 = np.where(E2>0, 0, E2)
    var3 = np.where(E3>0, 0, E3)
    return np.array([var1, var2, var3])

def understaff(var, solution, months, TH, EH, UC):
    """
    Calculates the residual cost of understaffing, if any, for the optimal staffing solution.
    Args:
        var: numpy array representing employee requirements for all departments and months.
        solution: numpy array representing trainee hiring requirements by month and department as optimized by the pulp solver for the given parameters.
        months: list of months in which trainee hiring will be planned.
        TH: integer, hours trainee can work per month.
        EH: integer, hours employee can work per month.
        UC: integer, understaffing cost per employee.
    Returns:
       A float, the cost of understaffing.

    """
    newEmp=[]
    for i in np.copy(solution):
        h = i
        i=np.insert(i,0,0)
        i=np.delete(i,-1)
        h += i
        h *= TH/EH
        for j in range(len(months)-2):
            i=np.insert(i,0,0)
            i=np.delete(i,-1)
            h += i
        newEmp.append(h)
    newEmp = np.array(newEmp)
    uCost = -sum(sum(np.where(newEmp + var>0, 0, newEmp + var))) * UC
    return uCost

def Solver():
    """
    Based on given parameters and using a numpy array model, uses the PuLP solver to generate an optimal planning solution for the
    number of trainees required to satisfy staffing requirements at the lowest cost.
    
    Prints the objective function, constraints, variables, status, optimal solution if feasible and total cost.

    """
    dept = ['1','2','3'] # Departments for planning.
    months = ['JAN','FEB','MAR','APR','MAY','JUN'] # Months for planning.
    Dept1_HR = Dept2_HR = Dept3_HR = np.array([1900, 1700, 1600, 1900, 1500, 1800]) # Hours required per department.
    Emp = [25 , 20, 18] # Employees in each department at start of planning.
    T_0 = [2, 2, 2] # Trainees in each department one month before start of planning.
    Resign = {1:('APR', 3), 2:('MAY', 5), 3:('JUN', 6)}  # Resignations for each department.
    Train_Cap = 6 # Training capacity for each month.
    TH = 35 # Hours trainee can work per month.
    EH = 100 # Hours employee can work per month.
    TC = 50000 # Cost to train a trainee.
    UC = 60000 # Understaffing cost per employee.
    Min_Hires = Resign[1][1] + Resign[2][1] + Resign[3][1] - sum(T_0) # tallies minumum hiring requirements to satisfy min employee constraint
    
    # Invokes and saves the output of the surface function
    var = surface(months, Dept1_HR, Dept2_HR, Dept3_HR, Emp, T_0, Resign, TH, EH)
    
    # Builds variable name dict.
    T = {str(i+1)+str(j+1) : 0 for j in range(len(months)) for i in range(len(dept))}
    
    # Defines problem within pulp solver.
    prob = pulp.LpProblem('Employee_Training_Planning_Problem', pulp.LpMinimize)
    
    # Creates variables within pulp solver.
    vars = pulp.LpVariable.dicts('T', (T), lowBound=0, cat='Integer')
    
    # Defines objective function within pulp solver.
    prob += (pulp.lpSum([vars[i] for i in T]),'Total cost',)
    
    # Defines min hire constriants within pulp solver.
    prob += pulp.lpSum(vars[i] for i in T) >= Min_Hires
    
    # Finds max hiring requirment for each department and constrains hiring to a miinimum of two months earlier.
    for i in range(len(dept)):
        prob += pulp.lpSum(vars[(str(i+1)+str(j+1))] for j in range(max(1, np.argmin(var[i])-2))) + np.amin(var[i]) >= 0
    
    # Defines constriants within pulp solver that enourages early hiring to allow time to train. There's no cost penalty to hire early.
    for j in range(len(months)-1):
        prob += pulp.lpSum(vars[(str(i+1)+str(j+1))] - vars[(str(i+1)+str(j+2))] for i in range(len(dept))) >= 0
    
    # Defines max hire per month constriants within pulp solver.
    for j in range(len(months)):
        prob += pulp.lpSum(vars[(str(i+1)+str(j+1))] for i in range(len(dept))) <= Train_Cap
    
    # Prints problem.
    print(prob)
    
    # The problem is solved using a pulp solver.
    prob.solve()
    
    # Prints solution status.
    print('Status:', pulp.LpStatus[prob.status])
    
    solution = []
    # Prints variables and build solution array.
    for v in prob.variables():
        solution.append(v.varValue)
        print(v.name, '=', v.varValue)
    solution = np.reshape(np.array(solution), (3,6))
    
    # Invokes and saves value of understaff function, cost for understaffing.
    uCost = understaff(var, solution, months, TH, EH, UC)
    
    # Calculates training cost, product of number of trainees hired and cost to train a trainee.
    tCost = pulp.value(prob.objective) * TC
    
    # Prints objective function value.
    print('Total cost = $', int(tCost + uCost))

Solver() # Invokes Solver function
