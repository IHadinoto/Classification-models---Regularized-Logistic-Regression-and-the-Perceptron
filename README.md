# COMP9417 - Machine Learning

# Homework 2: Classification models - Regularized Logistic

# Regression and the Perceptron

**Introduction** In this homework we first look at a regularized version of logistic regression. You will im-
plement the algorithm from scratch and compare it to the existingsklearnimplementation. Special care
will be taken to ensure that our implementation is numerically stable. We then move on to consider the
important issue of hyperparameter tuning. In the second question we shift our focus to the perceptron and
dual perceptron algorithms. We will implement this algorithm from scratch and compare it to a variant
known as the rPerceptron.
**Points Allocation** There are a total of 28 marks.

- Question 1 a): 1 mark
- Question 1 b): 2 marks
- Question 1 c): 1 mark
- Question 1 d): 1 mark
- Question 1 e): 3 marks
- Question 1 f): 3 marks
- Question 1 g): 3 marks
- Question 1 h): 3 marks
- Question 2 a): 3 marks
- Question 2 b): 2 marks
- Question 2 c): 2 marks
- Question 2 d): 3 marks
- Question 2 e): 1 mark

**What to Submit**

- A **single PDF** file which contains solutions to each question. For each question, provide your solution
    in the form of text and requested plots. For some questions you will be requested to provide screen
    shots of code used to generate your answer — only include these when they are explicitly asked for.

## 1


- **.py file(s) containing all code you used for the project, which should be provided in a separate .zip**
    **file.** This code must match the code provided in the report.
- You may be deducted points for not following these instructions.
- You may be deducted points for poorly presented/formatted work. Please be neat and make your
    solutions clear. Start each question on a new page if necessary.
- You **cannot** submit a Jupyter notebook; this will receive a mark of zero. This does not stop you from
    developing your code in a notebook and then copying it into a .py file though, or using a tool such as
    **nbconvert** or similar.
- We will set up a Moodle forum for questions about this homework. Please read the existing questions
    before posting new questions. Please do some basic research online before posting questions. Please
    only post clarification questions. Any questions deemed to befishingfor answers will be ignored
    and/or deleted.
- Please check Moodle announcements for updates to this spec. It is your responsibility to check for
    announcements about the spec.
- Please complete your homework on your own, do not discuss your solution with other people in the
    course. General discussion of the problems is fine, but you must write out your own solution and
    acknowledge if you discussed any of the problems in your submission (including their name(s) and
    zID).
- As usual, we monitor all online forums such as Chegg, StackExchange, etc. Posting homework ques-
    tions on these site is equivalent to plagiarism and will result in a case of academic misconduct.

**When and Where to Submit**

- Due date: Week 7, Monday **March 28th** , 2022 by **5pm**. Please note that the forum will not be actively
    monitored on weekends.
- Late submissions will incur a penalty of 5% per day **from the maximum achievable grade**. For ex-
    ample, if you achieve a grade of 80/100 but you submitted 3 days late, then your final grade will be
    80 − 3 ×5 = 65. Submissions that are more than 5 days late will receive a mark of zero.
- Submission must be done through Moodle, no exceptions.


**Question 1. Regularized Logistic Regression
Note: throughout this question do not use any existing implementations of any of the algorithms
discussed unless explicitly asked to in the question. Using existing implementations can result in
a grade of zero for the entire question.** In this question we will work with the Regularized Logistic
Regression model for binary classification (i.e. we are trying to predict a binary target). Instead of using
mean squared error loss as in regression problems with a continuous target (such as linear regression),
we instead minimize the log-loss, also referred to as the cross entropy loss. Recall that for a parameter
vectorβ= (β 1 ,...,βp)∈Rp,yi∈{ 0 , 1 },xi∈Rpfori= 1,...,n, the log-loss is

```
L(β,β 0 ) =
```
```
∑n
```
```
i=
```
```
yiln
```
## (

## 1

```
σ(β 0 +βTxi)
```
## )

```
+ (1−yi) ln
```
## (

## 1

```
1 −σ(β 0 +βTxi)
```
## )

## ,

```
whereσ(z) = (1 +e−z)−^1 is the logistic sigmoid. In practice, we will usually add a penalty term, and
solve the optimization:
```
```
(βˆ 0 ,βˆ) = arg min
β 0 ,β
```
```
{CL(β 0 ,β) +penalty(β)} (1)
```
```
where the penalty is usually not applied to the bias termβ 0 , andCis a hyper-parameter. For example,
in the` 2 regularization case, we take penalty(β) =^12 ‖β‖^22 (a Ridge version of logistic regression).
(a) Consider thesklearnlogistic regression implementation (section 1.1.11), which claims to mini-
mize the following objective:
```
```
w,ˆˆc= arg min
w,c
```
## {

## 1

## 2

```
wTw+C
```
```
∑n
```
```
i=
```
```
log(1 + exp(− ̃yi(wTxi+c)))
```
## }

## . (2)

```
It turns out that this objective is identical to our objective above, but only after re-coding the binary
variables to be in{− 1 , 1 }instead of binary values{ 0 , 1 }. That is, ̃yi∈{− 1 , 1 }, whereasyi∈{ 0 , 1 }.
Argue rigorously that the two objectives (1) and (2) are identical, in that they give us the same
solutions (βˆ 0 = ˆcandβˆ= ˆw). Further, describe the role ofCin the objectives, how does it compare
to the standard Ridge parameterλ?What to submit: some commentary/your working.
(b) In the logistic regression loss, and indeed in many machine learning problems, we often deal with
expressions of the form
```
```
LogSumExp(x 1 ,...,xn) = log(ex^1 +···+exn).
```
```
This can be problematic from a computational perspective because it will often lead to numerical
overflow. (To see this, note thatlog(exp(x)) =xfor anyx, but try computing it naively with a large
x, e.g. runningnp.log(np.exp(1200))). In this question we will explore a smart trick to avoid
such issues.
(I) Letx∗= max(x 1 ,...,xn)and show that
```
```
LogSumExp(x 1 ,...,xn) =x∗+ log
```
## (

```
ex^1 −x∗+···+exn−x∗
```
## )

## .

```
(II) Show each term inside of the log is a number greater than zero and less than equal to 1.
(III) Hence, explain why rewriting the LogSumExp function in this way avoids numerical overflow.
What to submit: some commentary/your working.
```

```
(c) For the remainder of this question we will work with thesongs.csvdataset. The data contains
information about various songs, and also contains a class variable outlining the genre of the song.
If you are interested, you can read more about the data here, though a deep understanding of each
of the features will not be crucial for the purposes of this assessment. Load in the data and preform
the follwing preprocessing:
(I) Remove the following features: ”Artist Name”, ”Track Name”, ”key”, ”mode”, ”timesignature”,
”instrumentalness”
(II) The current dataset has 10 classes, but logistic regression in the form we have described it here
only works for binary classification. We will restrict the data to classes 5 (hiphop) and 9 (pop).
After removing the other classes, re-code the variables so that the target variable isy= 1for
hiphop andy=− 1 for pop.
(III) Remove any remaining rows that have missing values for any of the features. Your remaining
dataset should have a total of 3886 rows.
(IV) Use thesklearn.modelselection.traintestsplitfunction to split your data into
Xtrain, Xtest, Ytrain and Ytest. Use atestsizeof 0. 3 and arandomstateof 23 for
reproducibility.
(V) Fit thesklearn.preprocessing.StandardScalerto the resulting training data, and
then use this object to scale both your train and test datasets.
(VI) Print out the first and last row of Xtrain, Xtest, ytrain, ytest (but only the first three columns
of Xtrain, Xtest).
What to submit: the print out of the rows requested in (VI). A copy of your code in solutions.py
```
(d) In homework 1 we fit thesklearn.preprocessing.StandardScalerto the entire dataset yet
here we are being more careful by only fitting it to the training data. In the context of a real-world
prediction problem, explain why what we are doing here is the proper thing to do.What to submit:
some commentary.
(e) Write a function (in Python)reglogloss(W, C, X, y)that computes the loss achieved by a
modelW= (c,w)with regularization parameterCon dataX,y(consiting ofnobservations, and
y∈{− 1 , 1 }), where the loss is defined by:

```
1
2
```
```
‖w‖^22 +C
```
```
∑n
```
```
i=
```
```
log(1 +e−yi(w
```
```
Txi+c)
).
```
```
Be careful to write your function in such a way to avoid overflow issues like the one described
in (b). To do so, make use ofnp.logaddexpfunction when writing your implementation. Hint:
1 =e^0. For comparison, you should get:
1 c = -1.
2 w = 0.1*np.ones(X_train.shape[1])
3 W = np.insert(w, 0, c)
4 reg_log_loss(W=W, C=0.001, X=X_train, y=y_train) # returns 1.
5
```
```
Print out the result of running the above code but with parameters set to:
w = 0.35 * np.ones(Xtrain.shape[1]), c=1.2.
Note that combining the intercept and weight parameters into a single array(W)is important for
getting your code to work in the next question.
What to submit: Your computed loss withw = 0.35 * np.ones(Xtrain.shape[1]), c=1.2,
a screen shot of your code, and a copy of your code in solutions.py.
```

```
(f) Now that we have a way of quantifying the loss of any given modelW= (w,c)in a stable way, we
are in a position to fit the logistic regression model to the data. Implement a function (in Python)
reglogfit(X, y, C)that returns fitted parametersWˆ = ( ˆw,ˆc). Your function should only
make use of thescipy.optimize.minimizefunction which performs numerical optimization
and can be applied directly to yourregloglossfunction from the previous question. Do this
minimization withC= 0. 4 and initial parameters set to the same values as in the code snippet in
(e), i.e.
1 w = 0.1*np.ones(X_train.shape[1])
2 W0 = np.insert(w,0, -1.4)
3
```
```
Further, usemethod=’Nelder-Mead’andtol=1e-6. Report the following quantities:
(I) Use thesklearn.metrics.loglossto compute the train and test losses of your resulting
model. Recall that the predictions of your model can be calculated using the formula:σ( ˆwTxi+
ˆc)wherexiis a single feature vector.
(II) Fit a logistic regression model usingsklearn.linearmodel.LogisticRegressionand
use the parameters: C=1, tol=1e-6, penalty=’l2’, solver=’liblinear’. Com-
pute the train and test losses for this model.
Optional hint: you may find it easier to usescipy.optimize.minimizewith a lambda func-
tion rather than directly applied to yourregloglossimplementation. This would mean
defining: g = lambda x: reglogloss(x,*args)
What to submit: train and test losses from your implementation as well as the ‘sklearn’ model, a screen shot
of your code, and a copy of your code in solutions.py.
```
(g) Up to this point, we have chosen the hyperparameterCarbitrarly. In this question we will study
the effect thatChas on the fitted model. We will focus on the` 1 penalized version of the problem:

```
‖w‖ 1 +C
```
```
∑n
```
```
i=
```
```
log(1 +e−yi(w
```
```
Txi+c)
),
```
```
and we will use thesklearnimplementation:
LogisticRegression(penalty=’l1’, solver=’liblinear’, C=c).
Use the following codeCs = np.linspace(0.001, 0.2, num=100)to generate a list ofC
values. For each value ofC, fit the model and store the coefficients of each feature. Create a plot
withlog(C)on thex-axis, and the coefficient value on they-axis for each feature in each of the
fitted models. In other words, the plot should describe what happens to each of the coefficients
in your model for different choices ofC. Based on this plot, explain why` 1 regularization can be
thought of as performingfeature selection, and further comment on which features seem the most
important to you (based on the plot) and why.
What to submit: a single plot, some commentary, a screen shot of your code and a copy of your code in
solutions.py. Your plot must have a legend clearly representing the different features using the following
color scheme: [’red’, ’brown’, ’green’, ’blue’, ’orange’, ’pink’, ’purple’, ’grey’, ’black’, ’y’]
```
(h) We now work through an example of using cross validation to choose the best choice ofCbased
on the data. Specifically, we will use Leave-One-Out Cross Validation (LOOCV). Create a grid of
Cvalues using the codeCs = np.linspace(0.0001, 0.8, num=25). LOOCV is computa-
tionally intesive so we will only work with the first20%of the training data (the firstn= 544
observations). For each data point in the training seti= 1,...,n:


```
(I) For a givenCvalue, fit the logistic regression model with` 1 penalty on the dataset with point
iremoved. You should have a total ofnmodels for eachCvalue.
(II) For youri-th model (the one trained by leaving out pointi) compute the leave-one-out error
of predictingyi(the log-loss of predicting the left out point).
(III) Average the losses over allnchoics ofito get your CV score for that particular choice ofC.
(IV) Repeat for all values ofC.
Plot the leave-one-out error againstCand report the bestCvalue. Note that for this question you
are not permitted to use any existing packages that implement cross validation though you are per-
mitted to usesklearn.LogisticRegressionto fit the models, andsklearn.metrics.logloss
to compute the losses - you must write the LOOCV implementatio code yourself from scratch and
you must create the plot yourself from scratch using basic matplotlib functionality.
What to submit: a single plot, some commentary, a screen shot of any code used for this section.
```
**Question 2. Perceptron Learning Variants**
In this question we will take a closer look at the perceptron algorithm. We will work with the fol-
lowing data filesPerceptronX.csvandPerceptrony.csv, which are theX,yfor our problem,
respectively. In this question (all parts), you are only permitted to use the following import statements:
1 import numpy as np
2 import pandas as pd # not really needed, only for preference
3 import matplotlib.pyplot as plt
4 from utils import*

```
Inutils.pywe have provided you with theplotperceptronfunction that automatically plots a
scatter of the data as well as your perceptron model.
(a) Write a function that implements the perceptron algorithm and run it onX,y. Your implementa-
tion should be based on the following pseudo-code:
```
```
input:(x 1 ,y 1 ),...,(xn,yn)
initialise:w(0)= (0, 0 ,...,0)∈Rp
fort= 1,...,maxiter:
if there is an indexisuch thatyi〈w(t),xi〉≤0 :
w(t+1)=w(t)+yixi; t=t+ 1
else:
outputw(t),t
```
```
Note that at each iteration of your model, you need to identify all the points that are misclassified
by your current weight vector, and then sample one of these points randomly and use it to per-
form your update. For consistency in this process, please set the random seed to 1 inside of your
function, i.e.,
1 def perceptron(X, y, max_iter=100):
2 np.random.seed(1)
3 # your code here
4
5 return w, nmb_iter
```

```
Themaxiterparameter here is used to control the maximum number of iterations your algorithm
is allowed to make before terminating and should be set to 100. Provide a plot of your final model
superimposed on a scatter of the data, and print out the final model parameters and number of
iterations taken for convergence in the title. For example, you can use the following code for
plotting:
1 w = ## your trained perceptron
2 fig, ax = plt.subplots()
3 plot_perceptron(ax, X, y, w) # your implementation
4 ax.set_title(f"w={w}, iterations={nmb_iter}")
5 plt.savefig("name.png", dpi=300) # if you want to save your plot as a png
6 plt.show()
```
```
What to submit: a single plot, a screen shot of your code used for this section, a copy of your code in
solutions.py
```
(b) In this section, we will implement and run the dual perceptron algorithm on the sameX,y. Recall
the dual perceptron pseudo-code is:

```
input:(x 1 ,y 1 ),...,(xn,yn)
initialise:α(0)= (0, 0 ,...,0)∈Rn
fort= 1,...,maxiter:
```
```
if there is an indexisuch thatyi
```
```
∑n
```
```
j=
```
```
yjαj〈xj,xi〉≤0 :
```
```
α
(t+1)
i =α
```
```
(t)
i + 1; t=t+ 1
else:
outputα(t),t
```
```
In your implementation, use the same method as described in part (a) to choose the point to update
on, using the same random seed inside your function. Provide a plot of your final perceptron as in
the previous part (using the same title format), and further, provide a plot withx-axis representing
each of thei= 1,...,npoints, andy-axis representing the valueαi. Briefly comment on your
results relative to the previous part.What to submit: two plots, some commentary, a screen shot of your
code used for this section, a copy of your code in solutions.py
(c) We now consider a slight variant of the perceptron algorithm outlined in part (a), known as the
rPerceptron. We introduce the following indicator variable fori= 1,...,n:
```
```
Ii=
```
## {

```
1 if(xi,yi)was already used in a previous update
0 otherwise.
```

```
Then we have the following pseudo-code:
```
```
input:(x 1 ,y 1 ),...,(xn,yn),r > 0
initialise:w(0)= (0, 0 ,...,0)∈Rp, I= (0, 0 ,...,0)∈Rn
fort= 1,...,maxiter:
if there is an indexisuch thatyi〈w(t),xi〉+Iir≤0 :
w(t+1)=w(t)+yixi; t=t+ 1; Ii= 1
else:
outputw(t),t
```
```
Implement the rPerceptron and run it onX,ytakingr= 2. Use the same randomization step as in
the previous parts to pick the point to update on, using a random seed of 1. Provide a plot of your
final results as in part (a), and print out the final weight vectors and number of iterations taken in
the title of your plot.What to submit: a single plot, a screen shot of your code used for this section, a copy
of your code in solutions.py
```
(d) Derive a dual version of the rPerceptron algorithm and describe it using pseudo-code (use the
template pseudocode from the previous parts to get an idea of what is expected here). Implement
your algorithm in code (using the same randomization steps as above) and run it onX,ywith
r= 2. Produce the same two plots as requested in part (b).What to submit: a pseudocode description of
your algorithm, two plots, a screen shot of your code used for this section, a copy of your code in solutions.py
(e) What role does the additive term introduced in the rPerceptron play? In what situations (different
types of datasets) could this algorithm give you an advantage over the standard perceptron? What
is a disadvantage of the rPerceptron relative to the standard perceptron? Refer to your results in
the previous parts to support your arguments; please be specific.What to submit: some commentary.
(f) (Optional Reading) The following post by Ben Recht gives a nice history of the perceptron and its
importance in the development of machine learning theory.


