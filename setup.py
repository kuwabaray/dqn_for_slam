from setuptools import setup, find_packages


NAME = 'dqn_for_slam'
REQUIRES_PYTHON = '>=3.5.0'

setup(
    name=NAME,
    version='0.0.1',
    description='reinforce learning for the robot',
    packages=find_packages(),
    python_requires=REQUIRES_PYTHON,
    install_requires=[
        'gym==0.17.3',
        'matplotlib==3.3.3',
        'tensorflow==2.3.1',
        'keras-rl2==1.0.4',
        'numpy==1.18.5',
        'rospkg==1.2.9'
    ],
    entry_points={
        'console_scripts': [
        	'run_agent=dqn_for_slam.rl_worker:main'
        ],
    }
)  
