# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 09:18:17 2021

@author: Mohamed Sabri
"""

from datetime import timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import random
from sklearn.datasets import make_regression 
from numpy import mean
from numpy import std
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from numpy import absolute
from numpy import mean
from numpy import std
import json
from numpyencoder import NumpyEncoder

default_args = {
    'owner': 'MohamedSabri',
    'provide_context': True,
}

with DAG(
    'MachinelearningPipeline',
    default_args=default_args,
    description='A test for my ML pipelines',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(2),
    tags=['ML'],
) as dag:

	def get_dataset(**kwargs):
		ti = kwargs['ti']
		X, y = make_regression(n_samples=100, n_features=1, tail_strength=0.9, effective_rank=1, n_informative=1, noise=3, bias=50, random_state=1)
		# add some artificial outliers
		random.seed(1)
		for i in range(10):
			factor = random.randint(2, 4)
			if random.random() > 0.5:
				X[i] += factor * X.std()
			else:
				X[i] -= factor * X.std() 
		X = json.dumps(X,cls=NumpyEncoder)
		y = json.dumps(y,cls=NumpyEncoder)
		ti.xcom_push('get_X', X)
		ti.xcom_push('get_y', y)



	def train(**kwargs):
		ti = kwargs['ti']
		def evaluate_model(X, y, model):
			# define model evaluation method
			cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
			# evaluate model
			scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
			# force scores to be positive
			return absolute(scores)
		X = ti.xcom_pull(task_ids='get_dataset', key='get_X')
		y = ti.xcom_pull(task_ids='get_dataset', key='get_y')
		print(X)
		print(y)

	get_dataXy = PythonOperator(
	    task_id="get_data",
	    python_callable=get_dataset,
	    provide_context=True,
	    dag=dag,
	)

	get_train = PythonOperator(
	    task_id="train_model",
	    python_callable=train,
	    provide_context=True,
	    dag=dag,
	)


	get_dataXy >> get_train
