# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 09:18:17 2021

@author: Mohamed Sabri
"""

from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator
from airflow.utils.dates import days_ago



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

	def get_dataset(**context):
		import random
		from sklearn.datasets import make_regression 
		from numpy import mean
		from numpy import std
		X, y = make_regression(n_samples=100, n_features=1, tail_strength=0.9, effective_rank=1, n_informative=1, noise=3, bias=50, random_state=1)
		# add some artificial outliers
		random.seed(1)
		for i in range(10):
			factor = random.randint(2, 4)
			if random.random() > 0.5:
				X[i] += factor * X.std()
			else:
				X[i] -= factor * X.std()  
		context['ti'].xcom_push('get_X', X)
		context['ti'].xcom_push('get_y', y)



	def train(**context):
		from sklearn.linear_model import LinearRegression
		from sklearn.model_selection import cross_val_score
		from sklearn.model_selection import RepeatedKFold
		from numpy import absolute
		from numpy import mean
		from numpy import std
		def evaluate_model(X, y, model):
			# define model evaluation method
			cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
			# evaluate model
			scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
			# force scores to be positive
			return absolute(scores)
		X = context['ti'].xcom_pull(task_ids='get_dataset', key='get_X')
		y = context['ti'].xcom_pull(task_ids='get_dataset', key='get_y')
		model = LinearRegression()
		# evaluate model
		results = evaluate_model(X, y, model)
		print('Mean MAE: %.3f (%.3f)' % (mean(results), std(results)))

	get_dataXy = PythonVirtualenvOperator(
	    task_id="get_data",
	    python_callable=get_dataset,
	    do_xcom_push=True,
	    requirements=["scikit-learn"],
	    system_site_packages=False,
	    dag=dag,
	    provide_context=True
	)

	get_train = PythonVirtualenvOperator(
	    task_id="train_model",
	    python_callable=train,
	    requirements=["scikit-learn"],
	    system_site_packages=False,
	    dag=dag,
	    provide_context=True
	)


	get_dataXy >> get_train
