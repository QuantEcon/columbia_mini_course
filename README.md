# Computational Economics with Python

## Graduate Mini Course at Columbia University


* Instructor: John Stachurski
* Dates: 26--28th March 2018
* Times and location: see [here](http://econ.columbia.edu/mini-course-john-stachurski-part-i-iii)


### Summary

This mini course will provide a fast paced introduction to Python for
computational economic modeling, from basic scripting to high performance
computing.  The course is aimed at graduate students with proficiency in at
least one scientific computing platform (e.g, MATLAB, Fortran, STATA, R, C or
Julia).

No Python knowledge is assumed.  

Please **be sure to bring your laptop**


### Instructions

Get Python + scientific libraries

* Install [Anaconda Python](https://www.anaconda.com/download/)

Update Numba (still necessary as of 25th March 2018)

* At terminal (Mac / Linux) or Anaconda Prompt (Windows), type `conda install numba=0.37`

Get files from this repo

* Use `git clone` if you know git or download [the zip file](https://github.com/QuantEcon/columbia_mini_course/archive/master.zip)


### Schedule


#### Day 1

* Python vs MATLAB vs Julia vs Fortran vs others
* The Python language: syntax and semantics
* Object oriented vs procedural programming
* [Jupyter notebooks](http://jupyter.org/)

#### Day 2

* The major scientific libraries ( [SciPy](http://www.scipy.org/) / NumPy / [Matplotlib](http://matplotlib.org/) / etc.)
* [Numba](http://numba.pydata.org/) and other JIT compilers
* Parallelization
* Distributed and cloud computing

#### Day 3

* Applications (asset pricing, optimal savings, optimal stopping)




### Links:

* [Anaconda](https://www.anaconda.com/)
* [AWS](https://aws.amazon.com/)
* [Accessing AWS via SSH](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html)


### Notes on AWS


#### To get an instance running

1. Login to Amazon AWS Console 
2. Navigate to EC2 Service
3. Choose your region for setting up an instance
6. Create security key-pair for the region if you don't have one
4. Launch & Configure an instance and choose Ubuntu 64-bit
5. enable access through Port 8000 (in addition to Port 22 for ssh)
6. Choose security key you've set up

#### Connecting and set up 

Use `ssh -i /path/to/pem-key ubuntu@hostname`

Here `hostname` is your Public DNS, as shown in the instance information from AWS console

Now run `sudo apt-get update` so you can install things you might need using `apt-get`


#### Configure instance to run Jupyter

1. ssh into the running instance using IP from AWS Console
2. Install Anaconda using wget and the latest download link for python36
3. Run: jupyter notebook --generate-config
4. For Automatic Password Setup run: jupyter notebook password
5. Edit .jupyter/jupyter_notebook_config.py and set the following

```
# Set ip to '*' to bind on all interfaces (ips) for the public server
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8000
```
