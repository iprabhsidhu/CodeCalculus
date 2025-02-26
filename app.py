from flask import Flask, request, render_template
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import solve_ivp
import io
import base64

app = Flask(__name__)

# Function to solve an ODE numerically using solve_ivp
def numerical_solver(ode_func, y0, t_range, method='RK45'):
    t = np.linspace(t_range[0], t_range[1], 100)
    sol_ivp = solve_ivp(ode_func, t_span=t_range, y0=y0, t_eval=t, method=method)
    return t, sol_ivp.y

# Function to solve an ODE symbolically
def symbolic_solver(equations, variables, initial_conditions):
    try:
        solutions = [sp.dsolve(eq, var) for eq, var in zip(equations, variables)]  # Solve individually
        return solutions
    except Exception as e:
        return f"⚠️ Error: {e}"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_eq = request.form.get('ode')
        initial_condition = request.form.get('initial_condition')
        time_end = float(request.form.get('time_end', 10))

        # Convert user input to a list of equations, multiple equation to be seperated by ;
        equations = user_eq.split(";") 
        y0 = [float(val) for val in initial_condition.split(",")] 

        # Define the numerical ODE function dynamically
        def example_ode(t, y):
            local_vars = {'t': t, 'np': np}  

            # Support both 'y1, y2,...' and 'v' if user inputs v
            for i, val in enumerate(y):
                local_vars[f'y{i+1}'] = val  # Define y1, y2, ...
                local_vars['v'] = y[0]  # If user enters 'v', assume it's the first variable

            try:
                return [eval(eq, local_vars) for eq in equations]  # Evaluate each equation safely
            except NameError as e:
                print(f"⚠️ ERROR: Undefined variable - {e}")
                return [0] * len(y)  # Return zero derivatives if an error occurs


        t_range = (0, time_end)
        t, sol_ivp = numerical_solver(example_ode, y0, t_range)

        # Symbolic solution
        t_sym = sp.Symbol('t')
        y_syms = [sp.Function(f'y{i+1}')(t_sym) for i in range(len(equations))]
        user_eq_sym = [eq.replace("np.", "sp.") for eq in equations]  
        local_vars = {'t': t_sym, 'sp': sp}
        for i, y_var in enumerate(y_syms):
            local_vars[f'y{i+1}'] = y_var  

        local_vars = {'t': t_sym, 'sp': sp}

        # Define all dependent variables dynamically (y1, y2, ...)
        for i, y_var in enumerate(y_syms):
            local_vars[f'y{i+1}'] = y_var
            local_vars['v'] = y_syms[0]  
        symbolic_eqs = [sp.Eq(y.diff(t_sym), eval(eq, local_vars)) for y, eq in zip(y_syms, user_eq_sym)]

        initial_conditions = {y_syms[i].subs(t_sym, 0): y0[i] for i in range(len(y_syms))}

        if len(initial_conditions) == len(y_syms):
            sym_solution = symbolic_solver(symbolic_eqs, y_syms, initial_conditions)
        else:
            sym_solution = "\u26a0\ufe0f Error: Mismatch between ODEs and initial conditions!"

        # Plot the numerical solutions
        plt.figure()
        for i, sol in enumerate(sol_ivp):
            plt.plot(t, sol, label=f'solve_ivp: y{i+1}')
        plt.xlabel('t')
        plt.ylabel('y(t)')
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        return render_template('result.html', sym_solution=sp.pretty(sym_solution), image_base64=image_base64)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)