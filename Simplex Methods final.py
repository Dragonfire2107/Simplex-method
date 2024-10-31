import numpy as np
import pandas as pd
from tabulate import tabulate
from ortools.linear_solver import pywraplp
from colorama import Fore, Style, Back, init

# Initialize colorama for color support
init()

# List of available CSV files for the custom simplex method
file_list = [
    "c:/Users/mattp/OneDrive - De Vinci/ESILV/Semestre 5 (Vic)/Optimization and operational research/EX5.csv",
    "path_to_EX9.csv",
    "path_to_EX6.csv", 
    "path_to_EX10.csv",
    "path_to_EX3.csv"
]

def select_file():
    """Prompt the user to select a file by entering a number."""
    print("Please select a CSV file by entering a number (1 to 5):")
    for i, filename in enumerate(file_list, start=1):
        print(f"{i}. {filename}")
    
    choice = int(input("Enter your choice: ")) - 1
    if 0 <= choice < len(file_list):
        return file_list[choice]
    else:
        print("Invalid choice. Please enter a number between 1 and 5.")
        return select_file()

# --- Method 1: Solving with OR-Tools ---

def solve_with_ortools_from_csv(filename):
    # Load CSV data
    data = pd.read_csv(filename)
    
    # Instantiate a solver
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        print("Solver not created.")
        return

    # Define Variables
    var_names = [col for col in data.columns if col not in ['Section', 'Inequality', 'RHS', 'Objective']]
    variables = [solver.NumVar(0, solver.infinity(), var_name) for var_name in var_names]
    
    print("Defined Variables:")
    for var in variables:
        print(f"{var.name()}")

    # Define Objective Function
    objective_row = data[data['Section'] == 'Objective'].iloc[0]
    objective = solver.Objective()
    objective_type = objective_row['Objective'].strip().lower()
    
    for var, coef in zip(variables, objective_row[var_names]):
        objective.SetCoefficient(var, float(coef))
    
    if objective_type == 'max':
        objective.SetMaximization()
    elif objective_type == 'min':
        objective.SetMinimization()
    else:
        print("Invalid objective type in CSV file. Please enter 'Max' or 'Min'.")
        return

    # Define Constraints
    constraints = data[data['Section'].str.contains("Constraint")]
    for _, row in constraints.iterrows():
        coefficients = [float(coef) for coef in row[var_names].values]
        rhs_value = float(row['RHS'])
        inequality = row['Inequality'].strip()
        
        if inequality == "<=":
            solver.Add(sum(coef * var for coef, var in zip(coefficients, variables)) <= rhs_value)
        elif inequality == ">=":
            solver.Add(sum(coef * var for coef, var in zip(coefficients, variables)) >= rhs_value)
        elif inequality == "=":
            solver.Add(sum(coef * var for coef, var in zip(coefficients, variables)) == rhs_value)
        else:
            print(f"Invalid inequality in constraint: {inequality}")
            return

    # Solve the problem
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print("\nOptimal solution found:")
        print(f"Objective value = {solver.Objective().Value():.2f}")
        for var in variables:
            print(f"{var.name()} = {var.solution_value():.2f}")
    else:
        print("No optimal solution found.")

def solve_with_ortools_from_csv(filename):
    # Load CSV data
    data = pd.read_csv(filename)
    
    # Instantiate a solver
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        print("Solver not created.")
        return

    # Step 1: Define Variables based on columns in CSV
    var_names = [col for col in data.columns if col not in ['Section', 'Inequality', 'RHS', 'Objective']]
    variables = [solver.NumVar(0, solver.infinity(), var_name) for var_name in var_names]
    
    print("Defined Variables:")
    for var in variables:
        print(f"{var.name()}")

    # Step 2: Define Objective Function
    objective_row = data[data['Section'] == 'Objective'].iloc[0]
    objective = solver.Objective()
    objective_type = objective_row['Objective'].strip().lower()
    
    # Convert coefficients to float and set them in the objective
    for var, coef in zip(variables, objective_row[var_names]):
        objective.SetCoefficient(var, float(coef))  # Ensure coef is a float
    
    if objective_type == 'max':
        objective.SetMaximization()
    elif objective_type == 'min':
        objective.SetMinimization()
    else:
        print("Invalid objective type in CSV file. Please enter 'Max' or 'Min'.")
        return

    # Step 3: Define Constraints
    constraints = data[data['Section'].str.contains("Constraint")]
    for _, row in constraints.iterrows():
        coefficients = [float(coef) for coef in row[var_names].values]  # Convert coefficients to float
        rhs_value = float(row['RHS'])  # Ensure RHS is a float
        inequality = row['Inequality'].strip()
        
        # Define the constraint based on the inequality type
        if inequality == "<=":
            solver.Add(sum(coef * var for coef, var in zip(coefficients, variables)) <= rhs_value)
        elif inequality == ">=":
            solver.Add(sum(coef * var for coef, var in zip(coefficients, variables)) >= rhs_value)
        elif inequality == "=":
            solver.Add(sum(coef * var for coef, var in zip(coefficients, variables)) == rhs_value)
        else:
            print(f"Invalid inequality in constraint: {inequality}")
            return

    # Solve the problem
    status = solver.Solve()

    # Check if the solution is optimal
    if status == pywraplp.Solver.OPTIMAL:
        print("\nOptimal solution found:")
        print(f"Objective value = {solver.Objective().Value():.2f}")
        for var in variables:
            print(f"{var.name()} = {var.solution_value():.2f}")
    else:
        print("No optimal solution found.")


def solve_with_ortools_manual():
    # Instantiate a solver
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        print("Solver not created.")
        return

    # Step 1: Define Variables
    while True:
        num_vars_input = input("Enter the number of variables (or type 'stop' to exit): ")
        if num_vars_input.lower() == "stop":
            print("Exiting the program.")
            return
        try:
            num_vars = int(num_vars_input)
            if num_vars > 0:
                break
            else:
                print("Please enter a positive integer for the number of variables.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    var_names = []
    variables = []

    for i in range(num_vars):
        while True:
            var_name = input(f"Enter the name for variable {i + 1} (or type 'stop' to exit): ")
            if var_name.lower() == "stop":
                print("Exiting the program.")
                return
            if var_name:
                var_names.append(var_name)
                var = solver.NumVar(0, solver.infinity(), var_name)
                variables.append(var)
                break
            else:
                print("Variable name cannot be empty.")

    print("\nDefined Variables:")
    for var in variables:
        print(f"{var.name()}")

    # Step 2: Define Constraints
    while True:
        num_constraints_input = input("\nEnter the number of constraints (or type 'stop' to exit): ")
        if num_constraints_input.lower() == "stop":
            print("Exiting the program.")
            return
        try:
            num_constraints = int(num_constraints_input)
            if num_constraints > 0:
                break
            else:
                print("Please enter a positive integer for the number of constraints.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    for i in range(num_constraints):
        print(f"\nDefining Constraint {i + 1}")
        coefficients = []

        for var in var_names:
            while True:
                coef_input = input(f"Enter the coefficient for {var} (or type 'stop' to exit): ")
                if coef_input.lower() == "stop":
                    print("Exiting the program.")
                    return
                try:
                    coef = float(coef_input)
                    coefficients.append(coef)
                    break
                except ValueError:
                    print("Invalid input. Please enter a valid number.")

        # Choose inequality type by number with validation
        while True:
            print("Choose inequality type:")
            print("1. <= ")
            print("2. >= ")
            print("3. = ")
            inequality_choice = input("Enter your choice (1, 2, or 3) or type 'stop' to exit: ").strip()
            if inequality_choice.lower() == "stop":
                print("Exiting the program.")
                return
            if inequality_choice in ["1", "2", "3"]:
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

        # Right-hand side value with validation
        while True:
            rhs_input = input("Enter the right-hand side (RHS) value (or type 'stop' to exit): ")
            if rhs_input.lower() == "stop":
                print("Exiting the program.")
                return
            try:
                rhs_value = float(rhs_input)
                break
            except ValueError:
                print("Invalid input. Please enter a valid number.")

        # Build the constraint based on the user input
        if inequality_choice == "1":
            constraint_expr = sum(coef * var for coef, var in zip(coefficients, variables)) <= rhs_value
        elif inequality_choice == "2":
            constraint_expr = sum(coef * var for coef, var in zip(coefficients, variables)) >= rhs_value
        elif inequality_choice == "3":
            constraint_expr = sum(coef * var for coef, var in zip(coefficients, variables)) == rhs_value

        # Add constraint to the solver
        solver.Add(constraint_expr)

    # Step 3: Define the objective function
    print("\nDefining the Objective Function")
    objective_coeffs = []
    for var in var_names:
        while True:
            coef_input = input(f"Enter the coefficient for {var} in the objective function (or type 'stop' to exit): ")
            if coef_input.lower() == "stop":
                print("Exiting the program.")
                return
            try:
                coef = float(coef_input)
                objective_coeffs.append(coef)
                break
            except ValueError:
                print("Invalid input. Please enter a valid number.")

    # Choose objective type with validation
    while True:
        objective_type = input("Do you want to Maximize or Minimize the objective function? (Max/Min or type 'stop' to exit): ").strip().lower()
        if objective_type == "stop":
            print("Exiting the program.")
            return
        if objective_type in ["max", "min"]:
            break
        else:
            print("Invalid input. Please enter 'Max' or 'Min'.")

    objective = solver.Objective()
    for coef, var in zip(objective_coeffs, variables):
        objective.SetCoefficient(var, coef)

    if objective_type == "max":
        objective.SetMaximization()
    elif objective_type == "min":
        objective.SetMinimization()

    # Solve the problem
    status = solver.Solve()

    # Check if the solution is optimal
    if status == pywraplp.Solver.OPTIMAL:
        print("\nOptimal solution found:")
        print(f"Objective value = {solver.Objective().Value():.2f}")
        for var in variables:
            print(f"{var.name()} = {var.solution_value():.2f}")
    else:
        print("No optimal solution found.")

# --- Method 2: Custom Simplex Solver ---

def create_tableau(objective, constraints):
    num_vars = len(objective)
    tableau = []
    
    for i, constraint in enumerate(constraints):
        row = constraint[:-1] + [0] * len(constraints) + [constraint[-1]]
        if constraint[-1] >= 0:
            row[num_vars + i] = 1
        tableau.append(row)

    tableau.append(objective + [0] * (len(constraints) + 1))
    basic_vars = [f"slack{i+1}" for i in range(len(constraints))]
    return np.array(tableau, dtype='float'), basic_vars

def display_tableau(tableau, num_vars, basic_vars, pivot_position=None):
    columns = ["Basic Var"] + [f"x{i+1}" for i in range(num_vars)]
    columns += [f"slack{i+1}" for i in range(len(tableau) - 1)]
    columns.append("RHS")

    tableau_data = []
    for row_idx, row in enumerate(tableau[:-1]):
        displayed_row = [basic_vars[row_idx]]
        for col_idx, value in enumerate(row):
            if pivot_position and row_idx == pivot_position[0] and col_idx == pivot_position[1]:
                displayed_row.append(f"{Back.YELLOW}{value}{Style.RESET_ALL}")
            elif pivot_position and col_idx == pivot_position[1]:
                displayed_row.append(f"{Back.CYAN}{value}{Style.RESET_ALL}")
            else:
                displayed_row.append(f"{value}")
        tableau_data.append(displayed_row)
    
    objective_row = ["Z"] + [f"{value}" for value in tableau[-1, :]]
    tableau_data.append(objective_row)
    
    print("\nCurrent Tableau:")
    print(tabulate(tableau_data, headers=columns, tablefmt="fancy_grid"))

def find_pivot(tableau):
    pivot_col = np.argmin(tableau[-1, :-1])
    if tableau[-1, pivot_col] >= 0:
        return None, None

    ratios = []
    for i in range(len(tableau) - 1):
        col_value = tableau[i, pivot_col]
        if col_value > 0:
            ratios.append(tableau[i, -1] / col_value)
        else:
            ratios.append(np.inf)

    if all(ratio == np.inf for ratio in ratios):
        raise ValueError("The problem is unbounded.")
    pivot_row = np.argmin(ratios)
    return pivot_row, pivot_col

def perform_pivot(tableau, row, col, basic_vars, entering_var):
    tableau[row, :] /= tableau[row, col]
    for i in range(len(tableau)):
        if i != row:
            tableau[i, :] -= tableau[i, col] * tableau[row, :]
    basic_vars[row] = entering_var

def simplex_solver(objective, constraints):
    tableau, basic_vars = create_tableau(objective, constraints)
    num_vars = len(objective)
    iteration = 0
    
    while True:
        print(f"\nIteration {iteration}")
        display_tableau(tableau, num_vars, basic_vars)
        
        pivot_row, pivot_col = find_pivot(tableau)
        
        if pivot_row is None or pivot_col is None:
            print("Optimal solution found.")
            break
        
        entering_var = f"x{pivot_col + 1}" if pivot_col < num_vars else f"slack{pivot_col - num_vars + 1}"
        print(f"Pivot on row {pivot_row}, column {pivot_col} (Entering variable: {entering_var})")
        perform_pivot(tableau, pivot_row, pivot_col, basic_vars, entering_var)
        iteration += 1

    display_tableau(tableau, num_vars, basic_vars)
    
    print("\nOptimal Solution:")
    solution = {f"x{i+1}": 0 for i in range(num_vars)}
    for i, var in enumerate(basic_vars):
        if var.startswith("x"):
            solution[var] = tableau[i, -1]

    for i in range(num_vars):
        var_name = f"x{i+1}"
        print(f"{var_name} = {solution.get(var_name, 0)}")
    
    optimal_value = tableau[-1, -1]
    print(f"Optimal Objective Value: {optimal_value}")

def main():
    print("Choose a solution method:")
    print("1. OR-Tools (manual or CSV)")
    print("2. Custom Simplex Solver")
    method_choice = input("Enter your choice (1 or 2): ").strip()
    
    if method_choice == "1":
        input_type = input("Do you want to input the problem manually or provide a CSV file? (manual/csv): ").strip().lower()
        if input_type == "manual":
            solve_with_ortools_manual()
        elif input_type == "csv":
            filename = input("Enter the name of the CSV file: ").strip()
            solve_with_ortools_from_csv(filename)
        else:
            print("Invalid choice.")
    elif method_choice == "2":
        file_path = select_file()
        data = pd.read_csv(file_path)
        objective_row = [-1 * val for val in data.loc[0, ['x1', 'x2']].values]
        constraints = [[data.loc[i, 'x1'], data.loc[i, 'x2'], data.loc[i, 'RHS']] for i in range(1, len(data))]
        simplex_solver(objective=objective_row, constraints=constraints)

# Run the main function to start the program
main()
