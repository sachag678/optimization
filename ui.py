#!/usr/bin/env python

from optimization import *
import streamlit as st
import plotly.graph_objects as go

@st.cache
def generate_data(f, use_log, func_name):
    func_params = params[func_name]
    x = np.linspace(func_params['x_min'], func_params['x_max'], 1000)
    y = np.linspace(func_params['y_min'], func_params['y_max'], 1000)
    xv, yv = np.meshgrid(x, y)
    z = f(xv, yv)
    if use_log:
        z = np.log10(z)
    return x, y, z

def generate_figure(x, y, z):
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, opacity=0.8)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor='limegreen', project_z=True))
    fig.update_layout(autosize=False, height=800, margin=dict(l=0, r=0, b=0, t=0))
    fig.update_layout(scene_aspectmode='cube')
    return fig

st.set_page_config(layout="wide")
with st.sidebar:
    function_name = st.selectbox('Pick function', ['Rosenbrock', 'Paraboloid', 'Himmelblau'])
    method = st.selectbox('Pick optimization method', ['Steepest descent', 'Newton\'s method'])
    #step_size = st.selectbox('Pick step size selection algorithm', ['Backtracking', 'Custom'])
    if method == "Newton\'s method":
        st.write()
        mod_hess = st.radio(label='Modify Hessian to make it positive definite', options=['None', 'Add multiples of identity', 'Flip negative eigenvalues'])
    with st.form("x, y inputs"):
        col1, col2 = st.columns(2)
        with col1:
            input_x = st.number_input('Starting x value', value=params[function_name]['init_x'])
        with col2:
            input_y = st.number_input('Starting y value', value=params[function_name]['init_y'])
        st.form_submit_button("Submit")
    with st.form("Backtracking variables"):
        st.write("Backtracking line search parameters")
        col1, col2 = st.columns(2)
        with col1:
            c = st.number_input('Value of c', max_value=1.0, min_value=0.0, value=0.01, format='%e',
                                help='This parameter is used to scale the importance of the norm of the derivative in the armijo condition.')
        with col2:
            rho = st.number_input('Value of rho', value=0.75, max_value=1.0, min_value=0.01, format='%f',
                                  help='This parameter is used to modify the step_size during the backtracking search.')
        st.form_submit_button("Submit")

use_log = False
if function_name == "Paraboloid":
    f = parabaloid
elif function_name == "Himmelblau":
    f = himmelblau
    use_log = True
else:
    f = rosenbrock
    use_log = True

if method == 'Newton\'s method':

    xs, ys = newtons_method(f, x0=input_x, y0=input_y, multiple_identity=mod_hess=='Add multiples of identity',
                            flip_negative_eigs=mod_hess=='Flip negative eigenvalues', c=c, rho=rho)
else:
    xs, ys = steepest_descent(f, x0=input_x, y0=input_y,c=c, rho=rho)

x, y, z  = generate_data(f, use_log, function_name)
fig = generate_figure(x, y, z)

line_marker = dict(color='black', width=4)
zs = f(np.array(xs), np.array(ys))
if use_log:
    zs = np.log10(zs)
fig.add_scatter3d(x=xs, y=ys, z=zs, mode='markers+lines', line=line_marker, marker=dict(size=4))

st.plotly_chart(fig, use_container_width=True)
