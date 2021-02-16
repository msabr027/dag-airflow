# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 09:18:17 2021

@author: Mohamed Sabri
"""

from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from random import random
from random import randint
from random import seed
from numpy import mean
from numpy import std
from numpy import absolute
from textwrap import dedent

default_args = {
    'owner': 'MohamedSabri',
    'depends_on_past': False,
    'email': ['msabri@necando.com'],
    'email_on_failure': True,
    'email_on_retry': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'MachinelearningPipeline',
    default_args=default_args,
    description='A test for my ML pipelines',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(2),
    tags=['ML'],
)

def get_dataset(**kwargs):
    from sklearn.datasets import make_regression
    ti = kwargs['ti']
    X, y = make_regression(n_samples=100, n_features=1, tail_strength=0.9, effective_rank=1, n_informative=1, noise=3, bias=50, random_state=1)
	# add some artificial outliers
    seed(1)
    for i in range(10):
        factor = randint(2, 4)
        if random() > 0.5:
            X[i] += factor * X.std()
        else:
            X[i] -= factor * X.std()  
    ti.xcom_push('get_X', X)
    ti.xcom_push('get_y', y)

def evaluate_model(X, y, model):
	# define model evaluation method
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
	# force scores to be positive
	return absolute(scores)

def train(**kwargs):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedKFold
    ti = kwargs['ti']
    X = ti.xcom_pull(task_ids='get_dataset', key='get_X')
    y = ti.xcom_pull(task_ids='get_dataset', key='get_y')
    model = LinearRegression()
    # evaluate model
    results = evaluate_model(X, y, model)
    print('Mean MAE: %.3f (%.3f)' % (mean(results), std(results)))

task1 = BashOperator(
            task_id='install_pkg',
            bash_command='pip install scikit-learn',
        )

task2 = BashOperator(
            task_id='install_pkg',
            bash_command='pip install textwrap3',
        )

get_dataXy = PythonOperator(
    task_id='get_data',
    python_callable=get_dataset,
)
get_dataXy.doc_md = dedent()

get_train = PythonOperator(
    task_id='train_model',
    python_callable=train,
)
get_train.doc_md = dedent()

task1 >> task2 >>  get_dataXy >> get_train
